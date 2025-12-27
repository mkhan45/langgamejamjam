mod engine;
pub mod ir;

pub use engine::{
    format_solution, reify_term, ArithConstraint, ConstraintStore, SearchQueue, SearchStrategy,
    Solver, State, Subst, SolutionSet, TerminationReason,
};

#[cfg(test)]
mod state_tests;
