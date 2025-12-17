use std::collections::VecDeque;

use im::{HashMap, Vector};

use crate::ir::{Arena, Clause, Program, Prop, PropId, RelKind, Term, TermId, VarId};

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
        match terms.get(t) {
            Term::Var(v) => {
                if let Some(&t2) = self.map.get(v) {
                    self.walk(t2, terms)
                } else {
                    t
                }
            }
            _ => t,
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
            return Some(self.clone());
        }

        let term1 = terms.get(t1);
        let term2 = terms.get(t2);

        match (term1, term2) {
            (Term::Var(v1), _) => Some(self.extend(*v1, t2)),
            (_, Term::Var(v2)) => Some(self.extend(*v2, t1)),
            (Term::Atom(s1), Term::Atom(s2)) if s1 == s2 => Some(self.clone()),
            (Term::Int(i1), Term::Int(i2)) if i1 == i2 => Some(self.clone()),
            (Term::Float(f1), Term::Float(f2)) if f1 == f2 => Some(self.clone()),
            (Term::App { sym: s1, args: a1 }, Term::App { sym: s2, args: a2 }) if s1 == s2 => {
                self.unify_args(a1, a2, terms)
            }
            _ => None,
        }
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

    pub fn check_ground_constraints(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> bool {
        let ground: Vec<_> = self
            .constraints
            .iter()
            .filter(|c| Self::is_ground_constraint(c, subst, program))
            .cloned()
            .collect();

        if ground.is_empty() {
            return true;
        }

        let ground_store = ConstraintStore {
            constraints: ground.into_iter().collect(),
        };
        ground_store.solve(subst, program, z3_solver).is_some()
    }

    pub fn try_solve_and_propagate(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> Option<(Subst, ConstraintStore)> {
        let (ground, non_ground): (Vec<_>, Vec<_>) = self.iter()
            .cloned()
            .partition(|c| Self::is_ground_constraint(c, subst, program));
        
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

    pub fn solve(&self, subst: &Subst, program: &mut Program, z3_solver: &z3::Solver) -> Option<Subst> {
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
            if let Some(val) = model.eval(z3_var, true) {
                if let Some(i) = val.as_i64() {
                    let term_id = program.terms.alloc(Term::Int(i as i32));
                    new_subst = new_subst.extend(*var_id, term_id);
                }
            }
        }

        for (var_id, z3_var) in real_vars {
            if let Some(val) = model.eval(z3_var, true) {
                if let Some((num, den)) = val.as_rational() {
                    let f = num as f32 / den as f32;
                    let term_id = program.terms.alloc(Term::Float(f));
                    new_subst = new_subst.extend(*var_id, term_id);
                }
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
        let var = crate::ir::Var { name };
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
            Prop::False => {}
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
                queue.push(state.clone().with_goal(p1));
                queue.push(state.with_goal(p2));
            }
            Prop::Not(p) => {
                let mut neg_queue = SearchQueue::new();
                neg_queue.push(state.clone().with_goal(p));

                while let Some(neg_state) = neg_queue.pop() {
                    if let Some((goal, remaining)) = neg_state.pop_goal() {
                        self.step_prop(remaining, goal, &mut neg_queue);
                    } else {
                        return;
                    }
                }
                queue.push(state);
            }
            Prop::App { rel, args } => {
                let rel_info = self.program.rels.get(rel).clone();
                match rel_info.kind {
                    RelKind::User => {
                        self.step_user_rel(&state, rel, &args, queue);
                    }
                    RelKind::SMTInt => {
                        if let Some(constraint) = self.make_int_constraint(&rel_info.name, &args) {
                            let new_state = state.with_constraint(constraint);
                            if let Some((solved_subst, remaining)) = new_state
                                .constraints
                                .try_solve_and_propagate(&new_state.subst, self.program, &self.z3_solver)
                            {
                                queue.push(State {
                                    subst: solved_subst,
                                    constraints: remaining,
                                    goals: new_state.goals,
                                });
                            }
                        }
                    }
                    RelKind::SMTReal => {
                        if let Some(constraint) = self.make_real_constraint(&rel_info.name, &args) {
                            let new_state = state.with_constraint(constraint);
                            if let Some((solved_subst, remaining)) = new_state
                                .constraints
                                .try_solve_and_propagate(&new_state.subst, self.program, &self.z3_solver)
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
        rel: crate::ir::RelId,
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

    pub fn query(&mut self, goal: PropId) -> SolutionIter<'_, 'p> {
        self.query_with_strategy(goal, SearchStrategy::default())
    }

    pub fn query_with_strategy(
        &mut self,
        goal: PropId,
        strategy: SearchStrategy,
    ) -> SolutionIter<'_, 'p> {
        let mut state = State::new(goal);

        for &fact_prop in &self.program.facts {
            state = state.with_goal(fact_prop);
        }

        let mut queue = SearchQueue::with_strategy(strategy);
        queue.push(state);

        SolutionIter {
            solver: self,
            queue,
            max_steps: 100_000,
            steps: 0,
            max_solutions: None,
            solutions_found: 0,
        }
    }

    pub fn query_from_state(&mut self, state: State) -> SolutionIter<'_, 'p> {
        self.query_from_state_with_strategy(state, SearchStrategy::default())
    }

    pub fn query_from_state_with_strategy(
        &mut self,
        state: State,
        strategy: SearchStrategy,
    ) -> SolutionIter<'_, 'p> {
        let mut queue = SearchQueue::with_strategy(strategy);
        queue.push(state);

        SolutionIter {
            solver: self,
            queue,
            max_steps: 10000,
            steps: 0,
            max_solutions: None,
            solutions_found: 0,
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
            } else {
                if let Some(solved_subst) =
                    state.constraints.solve(&state.subst, self.program, &self.z3_solver)
                {
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
}

pub struct SolutionIter<'s, 'p> {
    solver: &'s mut Solver<'p>,
    queue: SearchQueue,
    max_steps: usize,
    steps: usize,
    max_solutions: Option<usize>,
    solutions_found: usize,
}

impl<'s, 'p> SolutionIter<'s, 'p> {
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_solutions = Some(limit);
        self
    }
}

impl<'s, 'p> Iterator for SolutionIter<'s, 'p> {
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(max) = self.max_solutions {
            if self.solutions_found >= max {
                return None;
            }
        }

        while let Some(state) = self.queue.pop() {
            self.steps += 1;
            if self.steps > self.max_steps {
                return None;
            }

            if let Some((goal, remaining)) = state.pop_goal() {
                self.solver.step_prop(remaining, goal, &mut self.queue);
            } else {
                if let Some(solved_subst) =
                    state.constraints.solve(&state.subst, self.solver.program, &self.solver.z3_solver)
                {
                    self.solutions_found += 1;
                    return Some(State {
                        subst: solved_subst,
                        constraints: ConstraintStore::new(),
                        goals: Vector::new(),
                    });
                }
            }
        }
        None
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
    if state.constraints.len() > 0 {
        parts.push(format!("[{} constraints]", state.constraints.len()));
    }
    if parts.is_empty() {
        "yes".to_string()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::compile::compile;
    use crate::ast::parser;
    use nom::Finish;

    const ALL_STRATEGIES: [SearchStrategy; 2] = [SearchStrategy::BFS, SearchStrategy::DFS];

    fn parse_and_compile(input: &str) -> Program {
        let result = parser::parse_module(input.into()).finish();
        let (_, module) = result.expect("parse failed");
        compile(&module)
    }

    fn for_each_strategy(test_fn: impl Fn(SearchStrategy)) {
        for strategy in ALL_STRATEGIES {
            test_fn(strategy);
        }
    }

    #[test]
    fn test_fact_query() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
    position(player, 0, 0)
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let position_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "position")
                .map(|(id, _)| id)
                .unwrap();

            let player_sym = program.symbols.intern("player".to_string());
            let player_term = program.terms.alloc(Term::Atom(player_sym));
            let x_var = program.vars.alloc(crate::ir::Var {
                name: "X".to_string(),
            });
            let y_var = program.vars.alloc(crate::ir::Var {
                name: "Y".to_string(),
            });
            let x_term = program.terms.alloc(Term::Var(x_var));
            let y_term = program.terms.alloc(Term::Var(y_var));

            let query_prop = program.props.alloc(Prop::App {
                rel: position_rel,
                args: vec![player_term, x_term, y_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let solution = &solutions[0];

            let x_val = solution.subst.walk(x_term, &solver.program.terms);
            let y_val = solution.subst.walk(y_term, &solver.program.terms);

            match solver.program.terms.get(x_val) {
                Term::Int(0) => {}
                other => panic!("Expected Int(0), got {:?} (strategy: {:?})", other, strategy),
            }
            match solver.program.terms.get(y_val) {
                Term::Int(0) => {}
                other => panic!("Expected Int(0), got {:?} (strategy: {:?})", other, strategy),
            }
        });
    }

    #[test]
    fn test_rule_backchain() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
    base(1)
End Facts

Begin Global:
Rule Derive:
    base(X)
    -------
    derived(X)
End Global
"#;
            let mut program = parse_and_compile(input);

            let derived_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "derived")
                .map(|(id, _)| id)
                .unwrap();

            let var = program.vars.alloc(crate::ir::Var {
                name: "Q".to_string(),
            });
            let var_term = program.terms.alloc(Term::Var(var));

            let query_prop = program.props.alloc(Prop::App {
                rel: derived_rel,
                args: vec![var_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let solution = &solutions[0];

            let result = solution.subst.walk(var_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Int(1) => {}
                other => panic!("Expected Int(1), got {:?} (strategy: {:?})", other, strategy),
            }
        });
    }

    #[test]
    fn test_multiple_facts() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
    item(sword)
    item(shield)
    item(potion)
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let item_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "item")
                .map(|(id, _)| id)
                .unwrap();

            let var = program.vars.alloc(crate::ir::Var {
                name: "I".to_string(),
            });
            let var_term = program.terms.alloc(Term::Var(var));

            let query_prop = program.props.alloc(Prop::App {
                rel: item_rel,
                args: vec![var_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 3, "strategy: {:?}", strategy);
        });
    }

    #[test]
    fn test_simple_graph() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
    edge(1, 2)
    edge(2, 3)
End Facts

Begin Global:
    Rule ConnT:
    and(edge(A, B), edge(B, C))
    ----------------------------
    connected(A, C)

    Rule ConnE:
    edge(A, B)
    ----------
    connected(A, B)
End Global
"#;
            let mut program = parse_and_compile(input);

            let connected_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "connected")
                .map(|(id, _)| id)
                .unwrap();

            let query_prop = program.props.alloc(Prop::App {
                rel: connected_rel,
                args: vec![
                    program.terms.alloc(Term::Int(1)),
                    program.terms.alloc(Term::Int(3)),
                ],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
        });
    }

    #[test]
    fn test_smt_constraint() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
    value(5)
    true()
End Facts

Begin Global:
Rule AddOne:
    and(value(X), int_add(X, 1, Y))
    -------------------------------
    next(Y)
End Global
"#;
            let mut program = parse_and_compile(input);

            let next_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "next")
                .map(|(id, _)| id)
                .unwrap();

            let var = program.vars.alloc(crate::ir::Var {
                name: "R".to_string(),
            });
            let var_term = program.terms.alloc(Term::Var(var));

            let query_prop = program.props.alloc(Prop::App {
                rel: next_rel,
                args: vec![var_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            // Constraints should be solved by Z3
            assert!(
                solutions[0].constraints.is_empty(),
                "constraints should be solved, strategy: {:?}",
                strategy
            );

            let result = solutions[0].subst.walk(var_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Int(6) => {}
                other => panic!(
                    "Expected next(6) for int_add(5, 1, Y), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_z3_int_add_forward() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let int_add_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "int_add")
                .map(|(id, _)| id)
                .unwrap();

            let var_b = program.vars.alloc(crate::ir::Var {
                name: "B".to_string(),
            });
            let var_b_term = program.terms.alloc(Term::Var(var_b));
            let one_term = program.terms.alloc(Term::Int(1));

            // int_add(1, 1, B) should give B = 2
            let query_prop = program.props.alloc(Prop::App {
                rel: int_add_rel,
                args: vec![one_term, one_term, var_b_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let result = solutions[0].subst.walk(var_b_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Int(2) => {}
                other => panic!(
                    "Expected B=2 for int_add(1, 1, B), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_z3_int_add_backward() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let int_add_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "int_add")
                .map(|(id, _)| id)
                .unwrap();

            let var_b = program.vars.alloc(crate::ir::Var {
                name: "B".to_string(),
            });
            let var_b_term = program.terms.alloc(Term::Var(var_b));
            let two_term = program.terms.alloc(Term::Int(2));
            let five_term = program.terms.alloc(Term::Int(5));

            // int_add(2, B, 5) should give B = 3
            let query_prop = program.props.alloc(Prop::App {
                rel: int_add_rel,
                args: vec![two_term, var_b_term, five_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let result = solutions[0].subst.walk(var_b_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Int(3) => {}
                other => panic!(
                    "Expected B=3 for int_add(2, B, 5), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_eq_conjunction_fails_on_conflict() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let var = program.vars.alloc(crate::ir::Var {
                name: "X".to_string(),
            });
            let var_term = program.terms.alloc(Term::Var(var));
            let one_term = program.terms.alloc(Term::Int(1));
            let two_term = program.terms.alloc(Term::Int(2));

            let eq1 = program.props.alloc(Prop::Eq(var_term, one_term));
            let eq2 = program.props.alloc(Prop::Eq(var_term, two_term));
            let query_prop = program.props.alloc(Prop::And(eq1, eq2));

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(
                solutions.len(),
                0,
                "and(eq(X, 1), eq(X, 2)) should fail (strategy: {:?})",
                strategy
            );
        });
    }

    #[test]
    fn test_eq_conjunction_succeeds_when_compatible() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let var_x = program.vars.alloc(crate::ir::Var {
                name: "X".to_string(),
            });
            let var_y = program.vars.alloc(crate::ir::Var {
                name: "Y".to_string(),
            });
            let var_x_term = program.terms.alloc(Term::Var(var_x));
            let var_y_term = program.terms.alloc(Term::Var(var_y));
            let one_term = program.terms.alloc(Term::Int(1));

            let eq1 = program.props.alloc(Prop::Eq(var_x_term, one_term));
            let eq2 = program.props.alloc(Prop::Eq(var_x_term, var_y_term));
            let query_prop = program.props.alloc(Prop::And(eq1, eq2));

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(
                solutions.len(),
                1,
                "and(eq(X, 1), eq(X, Y)) should succeed with Y=1 (strategy: {:?})",
                strategy
            );

            let y_val = solutions[0]
                .subst
                .walk(var_y_term, &solver.program.terms);
            match solver.program.terms.get(y_val) {
                Term::Int(1) => {}
                other => panic!("Expected Y=1, got {:?} (strategy: {:?})", other, strategy),
            }
        });
    }

    #[test]
    fn test_z3_real_add_forward() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let real_add_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "real_add")
                .map(|(id, _)| id)
                .unwrap();

            let var_c = program.vars.alloc(crate::ir::Var {
                name: "C".to_string(),
            });
            let var_c_term = program.terms.alloc(Term::Var(var_c));
            let one_term = program.terms.alloc(Term::Float(1.5));
            let two_term = program.terms.alloc(Term::Float(2.5));

            // real_add(1.5, 2.5, C) should give C = 4.0
            let query_prop = program.props.alloc(Prop::App {
                rel: real_add_rel,
                args: vec![one_term, two_term, var_c_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let result = solutions[0].subst.walk(var_c_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Float(f) if (*f - 4.0).abs() < 0.0001 => {}
                other => panic!(
                    "Expected C=4.0 for real_add(1.5, 2.5, C), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_z3_real_add_backward() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let real_add_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "real_add")
                .map(|(id, _)| id)
                .unwrap();

            let var_b = program.vars.alloc(crate::ir::Var {
                name: "B".to_string(),
            });
            let var_b_term = program.terms.alloc(Term::Var(var_b));
            let two_term = program.terms.alloc(Term::Float(2.0));
            let five_term = program.terms.alloc(Term::Float(5.0));

            // real_add(2.0, B, 5.0) should give B = 3.0
            let query_prop = program.props.alloc(Prop::App {
                rel: real_add_rel,
                args: vec![two_term, var_b_term, five_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let result = solutions[0].subst.walk(var_b_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Float(f) if (*f - 3.0).abs() < 0.0001 => {}
                other => panic!(
                    "Expected B=3.0 for real_add(2.0, B, 5.0), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_z3_real_div() {
        for_each_strategy(|strategy| {
            let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
            let mut program = parse_and_compile(input);

            let real_div_rel = program
                .rels
                .iter()
                .find(|(_, r)| r.name == "real_div")
                .map(|(id, _)| id)
                .unwrap();

            let var_c = program.vars.alloc(crate::ir::Var {
                name: "C".to_string(),
            });
            let var_c_term = program.terms.alloc(Term::Var(var_c));
            let ten_term = program.terms.alloc(Term::Float(10.0));
            let four_term = program.terms.alloc(Term::Float(4.0));

            // real_div(10.0, 4.0, C) should give C = 2.5
            let query_prop = program.props.alloc(Prop::App {
                rel: real_div_rel,
                args: vec![ten_term, four_term, var_c_term],
            });

            let mut solver = Solver::new(&mut program);
            let solutions: Vec<_> = solver
                .query_with_strategy(query_prop, strategy)
                .collect();

            assert_eq!(solutions.len(), 1, "strategy: {:?}", strategy);
            let result = solutions[0].subst.walk(var_c_term, &solver.program.terms);
            match solver.program.terms.get(result) {
                Term::Float(f) if (*f - 2.5).abs() < 0.0001 => {}
                other => panic!(
                    "Expected C=2.5 for real_div(10.0, 4.0, C), got {:?} (strategy: {:?})",
                    other, strategy
                ),
            }
        });
    }

    #[test]
    fn test_eager_constraint_pruning() {
        let input = std::fs::read_to_string("sample/inventory.l")
            .expect("Failed to read sample/inventory.l");
        let mut program = parse_and_compile(&input);

        let cartcost_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "cartCost")
            .map(|(id, _)| id)
            .unwrap();

        let var_c = program.vars.alloc(crate::ir::Var {
            name: "C".to_string(),
        });
        let var_c_term = program.terms.alloc(Term::Var(var_c));
        let var_a = program.vars.alloc(crate::ir::Var {
            name: "A".to_string(),
        });
        let var_a_term = program.terms.alloc(Term::Var(var_a));
        let zero_term = program.terms.alloc(Term::Int(0));

        // cartCost(C, A, 0) - with MaxSize=0, only empty cart should match
        let query_prop = program.props.alloc(Prop::App {
            rel: cartcost_rel,
            args: vec![var_c_term, var_a_term, zero_term],
        });

        let mut solver = Solver::new(&mut program);
        let solutions: Vec<_> = solver
            .query(query_prop)
            .with_limit(5)
            .with_max_steps(1000)
            .collect();

        assert_eq!(solutions.len(), 1, "Should find exactly one solution (empty cart)");
        
        let cart_result = solutions[0].subst.walk(var_c_term, &solver.program.terms);
        match solver.program.terms.get(cart_result) {
            Term::Atom(s) => {
                let name = solver.program.symbols.get(*s);
                assert_eq!(name, "nil", "Cart should be nil");
            }
            other => panic!("Expected nil cart, got {:?}", other),
        }

        let cost_result = solutions[0].subst.walk(var_a_term, &solver.program.terms);
        match solver.program.terms.get(cost_result) {
            Term::Int(0) => {}
            other => panic!("Expected cost 0, got {:?}", other),
        }
    }

    #[test]
    fn test_cartcost_with_items() {
        let input = std::fs::read_to_string("sample/inventory.l")
            .expect("Failed to read sample/inventory.l");
        let mut program = parse_and_compile(&input);

        let cartcost_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "cartCost")
            .map(|(id, _)| id)
            .unwrap();

        let var_c = program.vars.alloc(crate::ir::Var {
            name: "C".to_string(),
        });
        let var_c_term = program.terms.alloc(Term::Var(var_c));
        let var_t = program.vars.alloc(crate::ir::Var {
            name: "T".to_string(),
        });
        let var_t_term = program.terms.alloc(Term::Var(var_t));
        let two_term = program.terms.alloc(Term::Int(2));

        // cartCost(C, T, 2) - find carts with up to 2 items
        let query_prop = program.props.alloc(Prop::App {
            rel: cartcost_rel,
            args: vec![var_c_term, var_t_term, two_term],
        });

        let mut solver = Solver::new(&mut program);
        let solutions: Vec<_> = solver
            .query(query_prop)
            .with_limit(10)
            .collect();

        // Should find: nil(0), apple(10), banana(5), apple+apple(20), apple+banana(15), banana+apple(15), banana+banana(10)
        assert!(solutions.len() >= 5, "Should find multiple cart combinations, got {}", solutions.len());
    }

    #[test]
    fn test_cartcost_specific_total() {
        let input = std::fs::read_to_string("sample/inventory.l")
            .expect("Failed to read sample/inventory.l");
        let mut program = parse_and_compile(&input);

        let cartcost_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "cartCost")
            .map(|(id, _)| id)
            .unwrap();

        let var_c = program.vars.alloc(crate::ir::Var {
            name: "C".to_string(),
        });
        let var_c_term = program.terms.alloc(Term::Var(var_c));
        let ten_term = program.terms.alloc(Term::Int(10));
        let one_term = program.terms.alloc(Term::Int(1));

        // cartCost(C, 10, 1) - find single-item carts costing exactly 10
        let query_prop = program.props.alloc(Prop::App {
            rel: cartcost_rel,
            args: vec![var_c_term, ten_term, one_term],
        });

        let mut solver = Solver::new(&mut program);
        let solutions: Vec<_> = solver
            .query(query_prop)
            .with_limit(5)
            .with_max_steps(1000)
            .collect();

        // Should find cons(apple, nil) since apple costs 10
        assert_eq!(solutions.len(), 1, "Should find exactly one cart costing 10 with max 1 item");
    }

    #[test]
    fn test_cartcost_unbound_maxsize() {
        // Regression test: cartCost(A, 25, B) should work even when B (MaxSize) is unbound
        let input = std::fs::read_to_string("sample/inventory.l")
            .expect("Failed to read sample/inventory.l");
        let mut program = parse_and_compile(&input);

        let cartcost_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "cartCost")
            .map(|(id, _)| id)
            .unwrap();

        let var_a = program.vars.alloc(crate::ir::Var {
            name: "A".to_string(),
        });
        let var_a_term = program.terms.alloc(Term::Var(var_a));
        let var_b = program.vars.alloc(crate::ir::Var {
            name: "B".to_string(),
        });
        let var_b_term = program.terms.alloc(Term::Var(var_b));
        let twenty_five_term = program.terms.alloc(Term::Int(25));

        // cartCost(A, 25, B) - find carts costing 25 with any max size
        let query_prop = program.props.alloc(Prop::App {
            rel: cartcost_rel,
            args: vec![var_a_term, twenty_five_term, var_b_term],
        });

        let mut solver = Solver::new(&mut program);
        let solutions: Vec<_> = solver
            .query(query_prop)
            .with_limit(5)
            .with_max_steps(5000)
            .collect();

        // Should find at least one solution (e.g., cons(banana, cons(banana, cons(banana, cons(apple, nil)))) = 5+5+5+10 = 25)
        assert!(!solutions.is_empty(), "Should find carts costing 25 with unbound MaxSize");
    }
}
