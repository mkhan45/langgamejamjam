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

    pub fn iter(&self) -> impl Iterator<Item = &ArithConstraint> {
        self.constraints.iter()
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
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

pub struct SearchQueue {
    queue: VecDeque<State>,
}

impl SearchQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    pub fn push(&mut self, state: State) {
        self.queue.push_back(state);
    }

    pub fn pop(&mut self) -> Option<State> {
        self.queue.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
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
}

impl<'p> Solver<'p> {
    pub fn new(program: &'p mut Program) -> Self {
        Self {
            program,
            fresh_counter: 0,
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
                            queue.push(state.with_constraint(constraint));
                        }
                    }
                    RelKind::SMTReal => {
                        if let Some(constraint) = self.make_real_constraint(&rel_info.name, &args) {
                            queue.push(state.with_constraint(constraint));
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
        let mut state = State::new(goal);
        
        for &fact_prop in &self.program.facts {
            state = state.with_goal(fact_prop);
        }

        let mut queue = SearchQueue::new();
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

    pub fn query_from_state(&mut self, state: State) -> SolutionIter<'_, 'p> {
        let mut queue = SearchQueue::new();
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
                self.solutions_found += 1;
                return Some(state);
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

    fn parse_and_compile(input: &str) -> Program {
        let result = parser::parse_module(input.into()).finish();
        let (_, module) = result.expect("parse failed");
        compile(&module)
    }

    #[test]
    fn test_fact_query() {
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];

        let x_val = solution.subst.walk(x_term, &solver.program.terms);
        let y_val = solution.subst.walk(y_term, &solver.program.terms);

        match solver.program.terms.get(x_val) {
            Term::Int(0) => {}
            other => panic!("Expected Int(0), got {:?}", other),
        }
        match solver.program.terms.get(y_val) {
            Term::Int(0) => {}
            other => panic!("Expected Int(0), got {:?}", other),
        }
    }

    #[test]
    fn test_rule_backchain() {
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];

        let result = solution.subst.walk(var_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Int(1) => {}
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_facts() {
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 3);
    }

    #[test]
    fn test_simple_graph() {
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
            args: vec![program.terms.alloc(Term::Int(1)), program.terms.alloc(Term::Int(3))],
        });

        let mut solver = Solver::new(&mut program);
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 1);
    }

    #[test]
    fn test_smt_constraint() {
        let input = r#"Begin Facts:
    value(5)
End Facts

Begin Global:
Rule AddOne:
    value(X)
    --------
    next(int_add(X, 1))
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 1);
        assert!(!solutions[0].constraints.is_empty());
    }

    #[test]
    fn test_eq_conjunction_fails_on_conflict() {
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 0, "and(eq(X, 1), eq(X, 2)) should fail");
    }

    #[test]
    fn test_eq_conjunction_succeeds_when_compatible() {
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
        let solutions: Vec<_> = solver.query(query_prop).collect();

        assert_eq!(solutions.len(), 1, "and(eq(X, 1), eq(X, Y)) should succeed with Y=1");
        
        let y_val = solutions[0].subst.walk(var_y_term, &solver.program.terms);
        match solver.program.terms.get(y_val) {
            Term::Int(1) => {}
            other => panic!("Expected Y=1, got {:?}", other),
        }
    }
}
