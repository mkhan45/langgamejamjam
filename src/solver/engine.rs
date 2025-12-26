use std::collections::VecDeque;

#[cfg(feature = "profile")]
use std::cell::RefCell;

use im::{HashMap, Vector};

use crate::solver::ir::{Arena, Clause, Program, Prop, PropId, RelId, RelKind, Term, TermId, Var, VarId};

#[cfg(feature = "profile")]
thread_local! {
    static PROFILE_STATS: RefCell<ProfileStats> = RefCell::new(ProfileStats::new());
}

#[cfg(feature = "profile")]
#[derive(Debug, Default)]
struct ProfileStats {
    walk_calls: usize,
    walk_chains: usize,
    unify_calls: usize,
    unify_failures: usize,
}

#[cfg(feature = "profile")]
impl ProfileStats {
    fn new() -> Self {
        Self::default()
    }

    fn record_walk(&mut self, depth: usize) {
        self.walk_calls += 1;
        if depth > 1 {
            self.walk_chains += 1;
        }
    }

    fn record_unify(&mut self, success: bool) {
        self.unify_calls += 1;
        if !success {
            self.unify_failures += 1;
        }
    }
}

macro_rules! record_profile {
    ($stats:ident => $body:expr) => {
        #[cfg(feature = "profile")]
        {
            PROFILE_STATS.with(|_stats| {
                let mut $stats = _stats.borrow_mut();
                $body;
            });
        }
    };
}

#[derive(Clone, Default)]
pub struct Subst {
    map: HashMap<VarId, TermId>,
}

impl Subst {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn walk(&self, t: TermId, terms: &Arena<Term>) -> TermId {
        self.walk_impl(t, terms, 0)
    }

    fn walk_impl(&self, t: TermId, terms: &Arena<Term>, depth: usize) -> TermId {
        match terms.get(t) {
            Term::Var(v) => {
                if let Some(&t2) = self.map.get(v) {
                    let result = self.walk_impl(t2, terms, depth + 1);
                    record_profile!(stats => stats.record_walk(depth + 1));
                    result
                } else {
                    record_profile!(stats => stats.record_walk(depth));
                    t
                }
            }
            _ => {
                record_profile!(stats => stats.record_walk(depth));
                t
            }
        }
    }

    pub fn extend(&self, v: VarId, t: TermId) -> Self {
        Self {
            map: self.map.update(v, t),
        }
    }

    pub fn get(&self, v: VarId) -> Option<TermId> {
        self.map.get(&v).copied()
    }

    pub fn unify(&self, t1: TermId, t2: TermId, terms: &Arena<Term>) -> Option<Self> {
        let t1 = self.walk(t1, terms);
        let t2 = self.walk(t2, terms);

        if t1 == t2 {
            record_profile!(stats => stats.record_unify(true));
            return Some(self.clone());
        }

        let term1 = terms.get(t1);
        let term2 = terms.get(t2);

        let result = match (term1, term2) {
            (Term::Var(v1), _) => Some(self.extend(*v1, t2)),
            (_, Term::Var(v2)) => Some(self.extend(*v2, t1)),
            (Term::Atom(s1), Term::Atom(s2)) if s1 == s2 => Some(self.clone()),
            (Term::Int(i1), Term::Int(i2)) if i1 == i2 => Some(self.clone()),
            (Term::Float(f1), Term::Float(f2)) if f1 == f2 => Some(self.clone()),
            (Term::App { sym: s1, args: a1 }, Term::App { sym: s2, args: a2 }) if s1 == s2 => {
                self.unify_args(a1, a2, terms)
            }
            _ => None,
        };
        record_profile!(stats => stats.record_unify(result.is_some()));
        result
    }

    pub fn unify_args(
        &self,
        args1: &[TermId],
        args2: &[TermId],
        terms: &Arena<Term>,
    ) -> Option<Self> {
        if args1.len() != args2.len() {
            return None;
        }
        let mut subst = self.clone();
        for (&a1, &a2) in args1.iter().zip(args2.iter()) {
            subst = subst.unify(a1, a2, terms)?;
        }
        Some(subst)
    }
}

#[derive(Debug, Clone)]
pub enum ArithConstraint {
    IntEq(TermId, TermId),
    IntLt(TermId, TermId),
    IntLe(TermId, TermId),
    IntGt(TermId, TermId),
    IntGe(TermId, TermId),
    IntNeq(TermId, TermId),
    IntAdd(TermId, TermId, TermId),
    IntSub(TermId, TermId, TermId),
    IntMul(TermId, TermId, TermId),
    IntDiv(TermId, TermId, TermId),

    RealEq(TermId, TermId),
    RealLt(TermId, TermId),
    RealLe(TermId, TermId),
    RealGt(TermId, TermId),
    RealGe(TermId, TermId),
    RealNeq(TermId, TermId),
    RealAdd(TermId, TermId, TermId),
    RealSub(TermId, TermId, TermId),
    RealMul(TermId, TermId, TermId),
    RealDiv(TermId, TermId, TermId),
}

#[derive(Clone, Default)]
pub struct ConstraintStore {
    constraints: Vector<ArithConstraint>,
}

impl ConstraintStore {
    pub fn new() -> Self {
        Self {
            constraints: Vector::new(),
        }
    }

    pub fn add(&self, c: ArithConstraint) -> Self {
        Self {
            constraints: self.constraints.clone() + Vector::unit(c),
        }
    }

    pub fn is_ground_constraint(c: &ArithConstraint, subst: &Subst, program: &Program) -> bool {
        let terms = match c {
            ArithConstraint::IntEq(a, b)
            | ArithConstraint::IntNeq(a, b)
            | ArithConstraint::IntLt(a, b)
            | ArithConstraint::IntLe(a, b)
            | ArithConstraint::IntGt(a, b)
            | ArithConstraint::IntGe(a, b)
            | ArithConstraint::RealEq(a, b)
            | ArithConstraint::RealNeq(a, b)
            | ArithConstraint::RealLt(a, b)
            | ArithConstraint::RealLe(a, b)
            | ArithConstraint::RealGt(a, b)
            | ArithConstraint::RealGe(a, b) => vec![*a, *b],
            ArithConstraint::IntAdd(a, b, c)
            | ArithConstraint::IntSub(a, b, c)
            | ArithConstraint::IntMul(a, b, c)
            | ArithConstraint::IntDiv(a, b, c)
            | ArithConstraint::RealAdd(a, b, c)
            | ArithConstraint::RealSub(a, b, c)
            | ArithConstraint::RealMul(a, b, c)
            | ArithConstraint::RealDiv(a, b, c) => vec![*a, *b, *c],
        };
        terms.iter().all(|t| {
            let walked = subst.walk(*t, &program.terms);
            !matches!(program.terms.get(walked), Term::Var(_))
        })
    }

    /// Partition constraints into ground (fully determined) and non-ground (contains variables).
    /// Used during SLD resolution for eager constraint propagation.
    fn partition_ground_constraints(&self, subst: &Subst, program: &Program) -> (Vec<ArithConstraint>, Vec<ArithConstraint>) {
        self.iter()
            .cloned()
            .partition(|c| Self::is_ground_constraint(c, subst, program))
    }

    /// SLD resolution constraint propagation: solve ground constraints, defer non-ground.
    /// Returns refined substitution and remaining (non-ground) constraints.
    /// Used during search to eagerly prune infeasible branches.
    pub fn propagate_ground(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> Option<(Subst, ConstraintStore)> {
        let (ground, non_ground) = self.partition_ground_constraints(subst, program);
        
        let ground_store = ConstraintStore {
            constraints: ground.into_iter().collect(),
        };
        
        let new_subst = if ground_store.is_empty() {
            subst.clone()
        } else {
            ground_store.solve_constraints(subst, program, z3_solver)?
        };
        
        let remaining = non_ground.into_iter()
            .fold(ConstraintStore::new(), |acc, c| acc.add(c));
        
        Some((new_subst, remaining))
    }

    pub fn iter(&self) -> impl Iterator<Item = &ArithConstraint> {
        self.constraints.iter()
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Final constraint validation: solve all constraints (ground and non-ground).
    /// Used after proof search completes to verify the solution satisfies all constraints.
    /// Critical for negation: ensures phantom proofs with unsatisfiable constraints are rejected.
    pub fn solve_all(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> Option<Subst> {
        self.solve_constraints(subst, program, z3_solver)
    }

    fn solve_constraints(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> Option<Subst> {
        if self.is_empty() {
            return Some(subst.clone());
        }

        z3_solver.reset();

        let mut int_vars: std::collections::HashMap<VarId, z3::ast::Int> =
            std::collections::HashMap::new();
        let mut real_vars: std::collections::HashMap<VarId, z3::ast::Real> =
            std::collections::HashMap::new();

        for constraint in self.iter() {
            let assertion = Self::constraint_to_z3(
                constraint, subst, &program.terms, &mut int_vars, &mut real_vars
            )?;
            z3_solver.assert(&assertion);
        }

        match z3_solver.check() {
            z3::SatResult::Sat => {
                let model = z3_solver.get_model()?;
                Some(Self::extract_bindings(&model, &int_vars, &real_vars, subst, program))
            }
            _ => None,
        }
    }

    fn constraint_to_z3(
        constraint: &ArithConstraint,
        subst: &Subst,
        terms: &Arena<Term>,
        int_vars: &mut std::collections::HashMap<VarId, z3::ast::Int>,
        real_vars: &mut std::collections::HashMap<VarId, z3::ast::Real>,
    ) -> Option<z3::ast::Bool> {
        let mut to_int = |t: TermId| -> Option<z3::ast::Int> {
            let walked = subst.walk(t, terms);
            terms.get(walked).to_z3_int(int_vars)
        };

        let mut to_real = |t: TermId| -> Option<z3::ast::Real> {
            let walked = subst.walk(t, terms);
            terms.get(walked).to_z3_real(real_vars)
        };

        match constraint {
            ArithConstraint::IntAdd(a, b, c) => {
                Some(z3::ast::Int::add(&[&to_int(*a)?, &to_int(*b)?]).eq(&to_int(*c)?))
            }
            ArithConstraint::IntSub(a, b, c) => {
                Some(z3::ast::Int::sub(&[&to_int(*a)?, &to_int(*b)?]).eq(&to_int(*c)?))
            }
            ArithConstraint::IntMul(a, b, c) => {
                Some(z3::ast::Int::mul(&[&to_int(*a)?, &to_int(*b)?]).eq(&to_int(*c)?))
            }
            ArithConstraint::IntDiv(a, b, c) => {
                Some(to_int(*a)?.div(&to_int(*b)?).eq(&to_int(*c)?))
            }
            ArithConstraint::IntEq(a, b) => Some(to_int(*a)?.eq(&to_int(*b)?)),
            ArithConstraint::IntNeq(a, b) => Some(to_int(*a)?.eq(&to_int(*b)?).not()),
            ArithConstraint::IntLt(a, b) => Some(to_int(*a)?.lt(&to_int(*b)?)),
            ArithConstraint::IntLe(a, b) => Some(to_int(*a)?.le(&to_int(*b)?)),
            ArithConstraint::IntGt(a, b) => Some(to_int(*a)?.gt(&to_int(*b)?)),
            ArithConstraint::IntGe(a, b) => Some(to_int(*a)?.ge(&to_int(*b)?)),
            ArithConstraint::RealAdd(a, b, c) => {
                Some(z3::ast::Real::add(&[&to_real(*a)?, &to_real(*b)?]).eq(&to_real(*c)?))
            }
            ArithConstraint::RealSub(a, b, c) => {
                Some(z3::ast::Real::sub(&[&to_real(*a)?, &to_real(*b)?]).eq(&to_real(*c)?))
            }
            ArithConstraint::RealMul(a, b, c) => {
                Some(z3::ast::Real::mul(&[&to_real(*a)?, &to_real(*b)?]).eq(&to_real(*c)?))
            }
            ArithConstraint::RealDiv(a, b, c) => {
                Some(to_real(*a)?.div(&to_real(*b)?).eq(&to_real(*c)?))
            }
            ArithConstraint::RealEq(a, b) => Some(to_real(*a)?.eq(&to_real(*b)?)),
            ArithConstraint::RealNeq(a, b) => Some(to_real(*a)?.eq(&to_real(*b)?).not()),
            ArithConstraint::RealLt(a, b) => Some(to_real(*a)?.lt(&to_real(*b)?)),
            ArithConstraint::RealLe(a, b) => Some(to_real(*a)?.le(&to_real(*b)?)),
            ArithConstraint::RealGt(a, b) => Some(to_real(*a)?.gt(&to_real(*b)?)),
            ArithConstraint::RealGe(a, b) => Some(to_real(*a)?.ge(&to_real(*b)?)),
        }
    }

    fn extract_bindings(
        model: &z3::Model,
        int_vars: &std::collections::HashMap<VarId, z3::ast::Int>,
        real_vars: &std::collections::HashMap<VarId, z3::ast::Real>,
        subst: &Subst,
        program: &mut Program,
    ) -> Subst {
        let mut new_subst = subst.clone();

        for (var_id, z3_var) in int_vars {
            if let Some(val) = model.eval(z3_var, true)
                && let Some(i) = val.as_i64()
            {
                let term_id = program.terms.alloc(Term::Int(i as i32));
                new_subst = new_subst.extend(*var_id, term_id);
            }
        }

        for (var_id, z3_var) in real_vars {
            if let Some(val) = model.eval(z3_var, true)
                && let Some((num, den)) = val.as_rational()
            {
                let f = num as f32 / den as f32;
                let term_id = program.terms.alloc(Term::Float(f));
                new_subst = new_subst.extend(*var_id, term_id);
            }
        }

        new_subst
    }
}

#[derive(Clone)]
pub struct State {
    pub subst: Subst,
    pub constraints: ConstraintStore,
    pub goals: Vector<PropId>,
}

impl State {
    pub fn new(initial_goal: PropId) -> Self {
        Self {
            subst: Subst::new(),
            constraints: ConstraintStore::new(),
            goals: Vector::unit(initial_goal),
        }
    }

    pub fn empty() -> Self {
        Self {
            subst: Subst::new(),
            constraints: ConstraintStore::new(),
            goals: Vector::new(),
        }
    }

    pub fn with_subst(&self, subst: Subst) -> Self {
        Self {
            subst,
            constraints: self.constraints.clone(),
            goals: self.goals.clone(),
        }
    }

    pub fn with_constraint(&self, c: ArithConstraint) -> Self {
        Self {
            subst: self.subst.clone(),
            constraints: self.constraints.add(c),
            goals: self.goals.clone(),
        }
    }

    pub fn with_goal(&self, goal: PropId) -> Self {
        Self {
            subst: self.subst.clone(),
            constraints: self.constraints.clone(),
            goals: self.goals.clone() + Vector::unit(goal),
        }
    }

    pub fn with_goals(&self, new_goals: impl IntoIterator<Item = PropId>) -> Self {
        let mut goals = self.goals.clone();
        for g in new_goals {
            goals.push_back(g);
        }
        Self {
            subst: self.subst.clone(),
            constraints: self.constraints.clone(),
            goals,
        }
    }

    pub fn pop_goal(&self) -> Option<(PropId, Self)> {
        if self.goals.is_empty() {
            None
        } else {
            let mut goals = self.goals.clone();
            let goal = goals.pop_front().unwrap();
            Some((
                goal,
                Self {
                    subst: self.subst.clone(),
                    constraints: self.constraints.clone(),
                    goals,
                },
            ))
        }
    }

    pub fn is_solved(&self) -> bool {
        self.goals.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchStrategy {
    #[default]
    BFS,
    DFS,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    LimitReached,
    SearchExhausted,
    MaxStepsReached,
}

#[derive(Clone)]
pub struct SolutionSet {
    pub solutions: Vec<State>,
    pub reason: TerminationReason,
}

impl SolutionSet {
    pub fn solutions(&self) -> &[State] {
        &self.solutions
    }
}

pub struct SearchQueue {
    pub queue: VecDeque<State>,
    pub strategy: SearchStrategy,
}

impl SearchQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            strategy: SearchStrategy::default(),
        }
    }

    pub fn with_strategy(strategy: SearchStrategy) -> Self {
        Self {
            queue: VecDeque::new(),
            strategy,
        }
    }

    pub fn push(&mut self, state: State) {
        self.queue.push_back(state);
    }

    pub fn pop(&mut self) -> Option<State> {
        match self.strategy {
            SearchStrategy::BFS => self.queue.pop_front(),
            SearchStrategy::DFS => self.queue.pop_back(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn strategy(&self) -> SearchStrategy {
        self.strategy
    }
}

impl Default for SearchQueue {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Solver<'p> {
    pub program: &'p mut Program,
    fresh_counter: u32,
    z3_solver: z3::Solver,
}

impl<'p> Solver<'p> {
    pub fn new(program: &'p mut Program) -> Self {
        Self {
            program,
            fresh_counter: 0,
            z3_solver: z3::Solver::new(),
        }
    }

    fn fresh_var(&mut self) -> (VarId, TermId) {
        let name = format!("_S{}", self.fresh_counter);
        self.fresh_counter += 1;
        let var = Var { name };
        let var_id = self.program.vars.alloc(var);
        let term_id = self.program.terms.alloc(Term::Var(var_id));
        (var_id, term_id)
    }

    fn instantiate_clause(&mut self, clause: &Clause) -> (Vec<TermId>, PropId) {
        let mut var_map: HashMap<VarId, TermId> = HashMap::new();

        let new_head_args: Vec<TermId> = clause
            .head_args
            .iter()
            .map(|&t| self.rename_term(t, &mut var_map))
            .collect();

        let new_body = self.rename_prop(clause.body, &mut var_map);

        (new_head_args, new_body)
    }

    fn rename_term(&mut self, term_id: TermId, var_map: &mut HashMap<VarId, TermId>) -> TermId {
        match self.program.terms.get(term_id).clone() {
            Term::Var(v) => {
                let var_name = &self.program.vars.get(v).name;
                if let Some(&original_state_var_term_id) = self.program.state_var_term_ids.get(var_name)
                    && original_state_var_term_id == term_id
                {
                    return term_id;
                }
                
                if let Some(&new_term) = var_map.get(&v) {
                    new_term
                } else {
                    let (_, new_term_id) = self.fresh_var();
                    var_map.insert(v, new_term_id);
                    new_term_id
                }
            }
            Term::App { sym, args } => {
                let new_args: Vec<TermId> = args
                    .iter()
                    .map(|&t| self.rename_term(t, var_map))
                    .collect();
                if new_args == args {
                    term_id
                } else {
                    self.program.terms.alloc(Term::App { sym, args: new_args })
                }
            }
            Term::Atom(_) | Term::Int(_) | Term::Float(_) => term_id,
        }
    }

    fn rename_prop(&mut self, prop_id: PropId, var_map: &mut HashMap<VarId, TermId>) -> PropId {
        let prop = self.program.props.get(prop_id).clone();
        match prop {
            Prop::True | Prop::False => prop_id,
            Prop::Eq(t1, t2) => {
                let new_t1 = self.rename_term(t1, var_map);
                let new_t2 = self.rename_term(t2, var_map);
                if new_t1 == t1 && new_t2 == t2 {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::Eq(new_t1, new_t2))
                }
            }
            Prop::And(p1, p2) => {
                let new_p1 = self.rename_prop(p1, var_map);
                let new_p2 = self.rename_prop(p2, var_map);
                if new_p1 == p1 && new_p2 == p2 {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::And(new_p1, new_p2))
                }
            }
            Prop::Or(p1, p2) => {
                let new_p1 = self.rename_prop(p1, var_map);
                let new_p2 = self.rename_prop(p2, var_map);
                if new_p1 == p1 && new_p2 == p2 {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::Or(new_p1, new_p2))
                }
            }
            Prop::Cond(c, p1, p2) => {
                let new_c = self.rename_prop(c, var_map);
                let new_p1 = self.rename_prop(p1, var_map);
                let new_p2 = self.rename_prop(p2, var_map);
                if new_c == c && new_p1 == p1 && new_p2 == p2 {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::Cond(new_c, new_p1, new_p2))
                }
            }
            Prop::Not(p) => {
                let new_p = self.rename_prop(p, var_map);
                if new_p == p {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::Not(new_p))
                }
            }
            Prop::App { rel, ref args } => {
                let new_args: Vec<TermId> = args
                    .iter()
                    .map(|&t| self.rename_term(t, var_map))
                    .collect();
                if new_args == *args {
                    prop_id
                } else {
                    self.program.props.alloc(Prop::App {
                        rel,
                        args: new_args,
                    })
                }
            }
        }
    }

    fn step_prop(&mut self, state: State, prop_id: PropId, queue: &mut SearchQueue) {
        let prop = self.program.props.get(prop_id).clone();
        match prop {
            Prop::True => {
                queue.push(state);
            }
            Prop::False => {
                queue.pop();
            }
            Prop::Eq(t1, t2) => {
                if let Some(new_subst) = state.subst.unify(t1, t2, &self.program.terms) {
                    queue.push(state.with_subst(new_subst));
                }
            }
            Prop::And(p1, p2) => {
                let new_state = state.with_goals([p1, p2]);
                queue.push(new_state);
            }
            Prop::Or(p1, p2) => {
                queue.push(state.with_goal(p1));
                queue.push(state.with_goal(p2));
            }
            Prop::Cond(c, p1, p2) => {
                let t_prop = self.program.props.alloc(Prop::And(c, p1));
                let e_prop = {
                    let n_prop = self.program.props.alloc(Prop::Not(c));
                    self.program.props.alloc(Prop::And(n_prop, p2))
                };
                queue.push(state.with_goal(t_prop));
                queue.push(state.with_goal(e_prop));
            }
            Prop::Not(p) => {
                let mut neg_queue = SearchQueue::new();
                neg_queue.push(state.with_goal(p));
                
                let mut found_valid_solution = false;

                while let Some(neg_state) = neg_queue.pop() {
                    if let Some((goal, remaining)) = neg_state.pop_goal() {
                        self.step_prop(remaining, goal, &mut neg_queue);
                    } else {
                        // Found a complete proof state - verify constraints are satisfiable
                        if let Some(_solved_subst) = neg_state.constraints.solve_all(&neg_state.subst, self.program, &self.z3_solver) {
                             found_valid_solution = true;
                             break;
                         }
                    }
                }
                
                if !found_valid_solution {
                    queue.push(state);
                }
            }
            Prop::App { rel, args } => {
                let rel_info = self.program.rels.get(rel).clone();
                match rel_info.kind {
                    RelKind::User => {
                        self.step_user_rel(&state, rel, &args, queue);
                    }
                    RelKind::SMTInt | RelKind::SMTReal => {
                        let constraint = match rel_info.kind {
                            RelKind::SMTInt => self.make_int_constraint(&rel_info.name, &args),
                            RelKind::SMTReal => self.make_real_constraint(&rel_info.name, &args),
                            RelKind::User => unreachable!(),
                        };
                        if let Some(c) = constraint {
                            let new_state = state.with_constraint(c);
                            if let Some((solved_subst, remaining)) = new_state
                                .constraints
                                .propagate_ground(&new_state.subst, self.program, &self.z3_solver)
                            {
                                queue.push(State {
                                    subst: solved_subst,
                                    constraints: remaining,
                                    goals: new_state.goals,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    fn step_user_rel(
        &mut self,
        state: &State,
        rel: RelId,
        args: &[TermId],
        queue: &mut SearchQueue,
    ) {
        let matching_facts: Vec<Vec<TermId>> = self
            .program
            .facts
            .iter()
            .filter_map(|&prop_id| {
                match self.program.props.get(prop_id) {
                    Prop::App { rel: fact_rel, args: fact_args } if *fact_rel == rel => {
                        Some(fact_args.clone())
                    }
                    _ => None,
                }
            })
            .collect();

        for fact_args in matching_facts {
            if let Some(new_subst) = state.subst.unify_args(args, &fact_args, &self.program.terms) {
                queue.push(state.with_subst(new_subst));
            }
        }

        let clauses: Vec<Clause> = self
            .program
            .global_rules
            .iter()
            .filter(|c| c.head_rel == rel)
            .cloned()
            .collect();

        for clause in clauses {
            let (new_head_args, new_body) = self.instantiate_clause(&clause);

            if let Some(new_subst) =
                state.subst.unify_args(args, &new_head_args, &self.program.terms)
            {
                queue.push(state.with_subst(new_subst).with_goal(new_body));
            }
        }
    }

    fn make_int_constraint(&self, name: &str, args: &[TermId]) -> Option<ArithConstraint> {
        match (name, args) {
            ("int_eq", [a, b]) => Some(ArithConstraint::IntEq(*a, *b)),
            ("int_neq", [a, b]) => Some(ArithConstraint::IntNeq(*a, *b)),
            ("int_lt", [a, b]) => Some(ArithConstraint::IntLt(*a, *b)),
            ("int_le", [a, b]) => Some(ArithConstraint::IntLe(*a, *b)),
            ("int_gt", [a, b]) => Some(ArithConstraint::IntGt(*a, *b)),
            ("int_ge", [a, b]) => Some(ArithConstraint::IntGe(*a, *b)),
            ("int_add", [a, b, c]) => Some(ArithConstraint::IntAdd(*a, *b, *c)),
            ("int_sub", [a, b, c]) => Some(ArithConstraint::IntSub(*a, *b, *c)),
            ("int_mul", [a, b, c]) => Some(ArithConstraint::IntMul(*a, *b, *c)),
            ("int_div", [a, b, c]) => Some(ArithConstraint::IntDiv(*a, *b, *c)),
            _ => None,
        }
    }

    fn make_real_constraint(&self, name: &str, args: &[TermId]) -> Option<ArithConstraint> {
        match (name, args) {
            ("real_eq", [a, b]) => Some(ArithConstraint::RealEq(*a, *b)),
            ("real_neq", [a, b]) => Some(ArithConstraint::RealNeq(*a, *b)),
            ("real_lt", [a, b]) => Some(ArithConstraint::RealLt(*a, *b)),
            ("real_le", [a, b]) => Some(ArithConstraint::RealLe(*a, *b)),
            ("real_gt", [a, b]) => Some(ArithConstraint::RealGt(*a, *b)),
            ("real_ge", [a, b]) => Some(ArithConstraint::RealGe(*a, *b)),
            ("real_add", [a, b, c]) => Some(ArithConstraint::RealAdd(*a, *b, *c)),
            ("real_sub", [a, b, c]) => Some(ArithConstraint::RealSub(*a, *b, *c)),
            ("real_mul", [a, b, c]) => Some(ArithConstraint::RealMul(*a, *b, *c)),
            ("real_div", [a, b, c]) => Some(ArithConstraint::RealDiv(*a, *b, *c)),
            _ => None,
        }
    }



    pub fn step_until_solution(
        &mut self,
        mut queue: SearchQueue,
        max_steps: usize,
    ) -> (Option<State>, SearchQueue) {
        let mut steps = 0;

        while let Some(state) = queue.pop() {
            steps += 1;
            if steps > max_steps {
                queue.push(state);
                return (None, queue);
            }

            if let Some((goal, remaining)) = state.pop_goal() {
                self.step_prop(remaining, goal, &mut queue);
            } else if let Some(solved_subst) =
                state.constraints.solve_all(&state.subst, self.program, &self.z3_solver)
            {
                #[cfg(feature = "profile")]
                PROFILE_STATS.with(|stats| {
                    dbg!(&*stats.borrow());
                });
                return (
                    Some(State {
                        subst: solved_subst,
                        constraints: ConstraintStore::new(),
                        goals: Vector::new(),
                    }),
                    queue,
                );
            }
        }
        (None, queue)
    }

    pub fn init_query(&mut self, goal: PropId, strategy: SearchStrategy) -> SearchQueue {
        let mut state = State::new(goal);

        for &fact_prop in &self.program.facts {
            state = state.with_goal(fact_prop);
        }

        let mut queue = SearchQueue::with_strategy(strategy);
        queue.push(state);
        queue
    }

    /// Collect solutions from a query up to a given limit and step count.
    /// Unifies the batch query API into a single canonical path.
    pub fn collect_solutions(
        &mut self,
        goal: PropId,
        strategy: SearchStrategy,
        limit: usize,
        max_steps: usize,
    ) -> SolutionSet {
        let mut queue = self.init_query(goal, strategy);
        let mut solutions = Vec::new();

        loop {
            if solutions.len() >= limit {
                return SolutionSet {
                    solutions,
                    reason: TerminationReason::LimitReached,
                };
            }

            let (solution, remaining_queue) = self.step_until_solution(queue, max_steps);
            
            let hit_max_steps = solution.is_none() && !remaining_queue.is_empty();
            queue = remaining_queue;

            if let Some(state) = solution {
                solutions.push(state);
            } else if hit_max_steps {
                return SolutionSet {
                    solutions,
                    reason: TerminationReason::MaxStepsReached,
                };
            } else {
                return SolutionSet {
                    solutions,
                    reason: TerminationReason::SearchExhausted,
                };
            }
        }
    }
}



pub fn reify_term(term_id: TermId, subst: &Subst, program: &Program) -> String {
    let walked = subst.walk(term_id, &program.terms);
    match program.terms.get(walked) {
        Term::Var(v) => {
            let var = program.vars.get(*v);
            format!("?{}", var.name)
        }
        Term::Atom(s) => program.symbols.get(*s).clone(),
        Term::Int(i) => i.to_string(),
        Term::Float(f) => f.to_string(),
        Term::App { sym, args } => {
            let name = program.symbols.get(*sym).clone();
            let arg_strs: Vec<String> = args
                .iter()
                .map(|a| reify_term(*a, subst, program))
                .collect();
            format!("{}({})", name, arg_strs.join(", "))
        }
    }
}

pub fn format_solution(
    query_vars: &[(String, TermId)],
    state: &State,
    program: &Program,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (name, term_id) in query_vars {
        if !name.starts_with('_') {
            let value = reify_term(*term_id, &state.subst, program);
            parts.push(format!("{} = {}", name, value));
        }
    }
    if !state.constraints.is_empty() {
        parts.push(format!("[{} constraints]", state.constraints.len()));
    }
    if parts.is_empty() {
        "yes".to_string()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod engine_tests;
