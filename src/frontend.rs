pub mod ffi;
pub mod frontend_tests;

use std::collections::HashMap;

use nom::Finish;

use crate::solver::ir::{Program, PropId, Prop, TermId};
use crate::solver::{format_solution, Solver, SearchStrategy, SearchQueue, Subst, State, reify_term};

use crate::ast::parser;
use crate::ast::compile::Compiler;

pub struct Frontend {
    pub program: Program,
    pub var_map: HashMap<String, TermId>,
    pub strategy: SearchStrategy,
    pub max_steps: usize,
    pending_queue: Option<SearchQueue>,
    pending_query_vars: Vec<(String, TermId)>,
}

impl Default for Frontend {
    fn default() -> Self {
        Self {
            program: Program::default(),
            var_map: HashMap::new(),
            strategy: SearchStrategy::default(),
            max_steps: 10_000,
            pending_queue: None,
            pending_query_vars: Vec::new(),
        }
    }
}

impl Frontend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self, source: &str) -> Result<(), String> {
        let result = parser::parse_module(source.into()).finish();
        match result {
            Ok((_, module)) => {
                self.program = Program::default();
                let mut compiler = Compiler::new(&mut self.program);
                compiler.compile_module(&module);
                self.var_map = compiler.into_var_map();
                Ok(())
            }
            Err(e) => Err(format!("Parse error: {:?}", e)),
        }
    }

    /// Execute a batch query, returning up to `limit` solutions all at once.
    ///
    /// For incremental solution retrieval, use `query_start` / `query_next` instead.
    pub fn query_batch(&mut self, query_str: &str, limit: usize) -> Result<Vec<String>, String> {
        self.query_batch_with_steps(query_str, limit, self.max_steps)
    }

    fn query_batch_with_steps(
        &mut self,
        query_str: &str,
        limit: usize,
        max_steps: usize,
    ) -> Result<Vec<String>, String> {
        let term_result = parser::parse_term(query_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => return Err(format!("Query parse error: {:?}", e)),
        };

        let (goal, query_vars) = Compiler::with_var_map(&mut self.program, self.var_map.clone())
            .compile_query(&term);

        // State variables are synchronized via two mechanisms:
        // 1. var_map: Runtime tracking of current state variable term IDs
        // 2. program.facts: Contains current state as equality constraints
        //
        // When the compiler compiles a query, it uses var_map to resolve state variables
        // to their current term IDs. The solver then unifies these with facts, which already
        // contain the current state values. No additional constraints needed.
        //
        // facts are always the single source of truth for the solver.
        let mut solver = Solver::new(&mut self.program);
        let solutions: Vec<_> = solver.query_with_strategy(goal, self.strategy)
            .with_limit(limit)
            .with_max_steps(max_steps)
            .collect();

        Ok(solutions
            .iter()
            .map(|s| format_solution(&query_vars, s, solver.program))
            .collect())
    }

    /// Start an incremental query, returning the first solution if one exists.
    ///
    /// Use `query_next()` to retrieve subsequent solutions.
    /// Use `query_stop()` to abandon the query.
    /// Use `has_more_solutions()` to check if more results are available.
    pub fn query_start(&mut self, query_str: &str) -> Result<Option<String>, String> {
        let term_result = parser::parse_term(query_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => return Err(format!("Query parse error: {:?}", e)),
        };

        let (goal, query_vars) = Compiler::with_var_map(&mut self.program, self.var_map.clone())
            .compile_query(&term);

        self.pending_query_vars = query_vars;

        let mut solver = Solver::new(&mut self.program);
        let queue = solver.init_query(goal, self.strategy);
        let (solution, remaining_queue) = solver.step_until_solution(queue, self.max_steps);

        self.pending_queue = Some(remaining_queue);

        match solution {
            Some(state) => Ok(Some(format_solution(&self.pending_query_vars, &state, &self.program))),
            None => {
                if self.pending_queue.as_ref().is_none_or(|q| q.is_empty()) {
                    self.pending_queue = None;
                }
                Ok(None)
            }
        }
    }

    /// Retrieve the next solution from an ongoing incremental query.
    ///
    /// Returns None if no more solutions are available.
    pub fn query_next(&mut self) -> Option<String> {
        let queue = self.pending_queue.take()?;

        if queue.is_empty() {
            return None;
        }

        let mut solver = Solver::new(&mut self.program);
        let (solution, remaining_queue) = solver.step_until_solution(queue, self.max_steps);

        if remaining_queue.is_empty() {
            self.pending_queue = None;
        } else {
            self.pending_queue = Some(remaining_queue);
        }

        solution.map(|state| format_solution(&self.pending_query_vars, &state, &self.program))
    }

    /// Check if more solutions are available from the current incremental query.
    pub fn has_more_solutions(&self) -> bool {
        self.pending_queue.as_ref().is_some_and(|q| !q.is_empty())
    }

    /// Abandon the current incremental query and free its state.
    pub fn query_stop(&mut self) {
        self.pending_queue = None;
        self.pending_query_vars.clear();
    }

    pub fn run_stage(&mut self, stage_index: usize) -> Result<(), String> {
        if stage_index >= self.program.stages.len() {
            return Err(format!("Stage index {} out of bounds", stage_index));
        }

        let stage = &self.program.stages[stage_index];
        if stage.state_constraints.is_empty() {
            return Ok(());
        }

        // State constraint solving always has access to:
        // 1. Current facts (base knowledge, including current state variable values)
        // 2. Global rules (from Begin Global section - available everywhere)
        // 3. Stage-specific state constraints
        // Global rules are accessible via back-chaining in the solver
        let constraints = stage.state_constraints.clone();
        let stage_name = stage.name.clone();
        let next_var_map = stage.next_var_map.clone();

        let resolved_state_values: Vec<(String, TermId)> = {
            let true_prop = self.program.props.alloc(Prop::True);
            let mut solver = Solver::new(&mut self.program);
            let solutions: Vec<_> = solver.query_with_strategy(true_prop, self.strategy)
                .with_limit(1)
                .with_max_steps(1000)
                .collect();
            
            if let Some(solution) = solutions.first() {
                self.program.state_vars.clone().into_iter().filter_map(|name| {
                    let term_id = self.var_map.get(&name)?;
                    let resolved = solution.subst.walk(*term_id, &self.program.terms);
                    Some((name, resolved))
                }).collect()
            } else {
                Vec::new()
            }
        };

        let mut all_constraints = Vec::new();
        for (name, resolved_val) in &resolved_state_values {
            if let Some(&original_term) = self.program.state_var_term_ids.get(name) {
                let eq_prop = self.program.props.alloc(Prop::Eq(original_term, *resolved_val));
                all_constraints.push(eq_prop);
            }
        }
        all_constraints.extend(constraints.iter().copied());
        let combined_goal = self.conjoin_props(&all_constraints);

        let mut solver = Solver::new(&mut self.program);
        let state = State::new(combined_goal);
        let solutions: Vec<_> = solver.query_from_state_with_strategy(state, self.strategy)
            .with_limit(2)
            .with_max_steps(self.max_steps)
            .collect();

        match solutions.len() {
            0 => Err(format!(
                "State constraint failure in stage '{}': no solutions found",
                stage_name
            )),
            1 => {
                let solution = &solutions[0];
                // Collect new values
                let mut new_values = Vec::new();
                for (name, next_term_id) in &next_var_map {
                    let new_value = solution.subst.walk(*next_term_id, &solver.program.terms);
                    new_values.push((name.clone(), new_value));
                }
                
                // Update var_map with new values
                for (name, new_value) in &new_values {
                    self.var_map.insert(name.clone(), *new_value);
                }
                
                // Update facts to reflect new state variable values
                // Find and update fact constraints for state variables
                let mut updated_facts = Vec::new();
                for &fact_prop_id in &self.program.facts {
                    let fact_prop = self.program.props.get(fact_prop_id).clone();
                    if let Prop::Eq(term_a, term_b) = fact_prop {
                        // Check if this fact involves a state variable
                        let mut is_state_var_fact = false;
                        for (name, original_term_id) in &self.program.state_var_term_ids {
                            if term_a == *original_term_id || term_b == *original_term_id {
                                // This fact involves a state variable, check if we have a new value for it
                                if let Some((_, new_value)) = new_values.iter().find(|(n, _)| n == name) {
                                    // Replace this fact with the new value
                                    let updated_fact = if term_a == *original_term_id {
                                        self.program.props.alloc(Prop::Eq(*original_term_id, *new_value))
                                    } else {
                                        self.program.props.alloc(Prop::Eq(term_b, *new_value))
                                    };
                                    updated_facts.push(updated_fact);
                                    is_state_var_fact = true;
                                    break;
                                }
                            }
                        }
                        if !is_state_var_fact {
                            updated_facts.push(fact_prop_id);
                        }
                    } else {
                        updated_facts.push(fact_prop_id);
                    }
                }
                self.program.facts = updated_facts;
                
                Ok(())
            }
            _ => {
                let mut diff_vars = Vec::new();
                for (name, next_term_id) in &next_var_map {
                    let val1 = reify_term(
                        solutions[0].subst.walk(*next_term_id, &solver.program.terms),
                        &solutions[0].subst,
                        solver.program,
                    );
                    let val2 = reify_term(
                        solutions[1].subst.walk(*next_term_id, &solver.program.terms),
                        &solutions[1].subst,
                        solver.program,
                    );
                    if val1 != val2 {
                        diff_vars.push(format!("{}: {} vs {}", name, val1, val2));
                    }
                }
                Err(format!(
                    "Ambiguous state update in stage '{}': multiple solutions found. Differing state vars: [{}]",
                    stage_name,
                    diff_vars.join(", ")
                ))
            }
        }
    }

    pub fn run_stage_by_name(&mut self, name: &str) -> Result<(), String> {
        let stage_index = self.program.stages
            .iter()
            .position(|s| s.name == name)
            .ok_or_else(|| format!("Stage '{}' not found", name))?;
        self.run_stage(stage_index)
    }

    pub fn get_state_var(&mut self, name: &str) -> Option<String> {
        let term_id = *self.var_map.get(name)?;
        
        let true_prop = self.program.props.alloc(Prop::True);
        let mut solver = Solver::new(&mut self.program);
        let solutions: Vec<_> = solver.query_with_strategy(true_prop, self.strategy)
            .with_limit(1)
            .with_max_steps(self.max_steps)
            .collect();
        
        if let Some(solution) = solutions.first() {
            Some(reify_term(term_id, &solution.subst, solver.program))
        } else {
            Some(reify_term(term_id, &Subst::new(), &self.program))
        }
    }

    pub fn state_vars(&mut self) -> Vec<(String, String)> {
        let true_prop = self.program.props.alloc(Prop::True);
        let state_var_names = self.program.state_vars.clone();
        let var_map_snapshot: Vec<(String, TermId)> = state_var_names
            .iter()
            .filter_map(|name| {
                self.var_map.get(name).map(|&tid| (name.clone(), tid))
            })
            .collect();
        
        let mut solver = Solver::new(&mut self.program);
        let solutions: Vec<_> = solver.query_with_strategy(true_prop, self.strategy)
            .with_limit(1)
            .with_max_steps(self.max_steps)
            .collect();
        
        let subst = solutions.first().map(|s| &s.subst);
        
        var_map_snapshot
            .into_iter()
            .map(|(name, term_id)| {
                let value = if let Some(s) = subst {
                    reify_term(term_id, s, solver.program)
                } else {
                    reify_term(term_id, &Subst::new(), solver.program)
                };
                (name, value)
            })
            .collect()
    }

    fn conjoin_props(&mut self, props: &[PropId]) -> PropId {
        if props.is_empty() {
            self.program.props.alloc(Prop::True)
        } else if props.len() == 1 {
            props[0]
        } else {
            let mut result = props[0];
            for &p in &props[1..] {
                result = self.program.props.alloc(Prop::And(result, p));
            }
            result
        }
    }
}
