use std::collections::VecDeque;

use im::{HashMap, Vector};

use crate::ir::{Arena, PropId, Term, TermId, VarId};

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

    pub fn with_goal(&self, goal: PropId) -> Self {
        Self {
            subst: self.subst.clone(),
            constraints: self.constraints.clone(),
            goals: self.goals.clone() + Vector::unit(goal),
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
}

impl Default for SearchQueue {
    fn default() -> Self {
        Self::new()
    }
}
