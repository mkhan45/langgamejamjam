pub mod ffi;
pub mod frontend_tests;

use std::collections::HashMap;

use nom::Finish;

use crate::solver::ir::{Program, PropId, Prop, Term, TermId};
use crate::solver::{format_solution, Solver, SearchStrategy, SearchQueue, Subst, reify_term, TerminationReason};

use crate::ast::parser;
use crate::ast::compile::Compiler;

#[derive(Debug, Clone, PartialEq)]
pub struct DrawCommand {
    pub name: String,
    pub args: Vec<f32>,
}

pub struct Frontend {
    pub program: Program,
    pub var_map: HashMap<String, TermId>,
    pub strategy: SearchStrategy,
    pub max_steps: usize,
    pending_queue: Option<SearchQueue>,
    pending_query_vars: Vec<(String, TermId)>,
    pub last_query_reason: Option<TerminationReason>,
    active_stage: Option<usize>,
    pub draw_cache: Vec<DrawCommand>,
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
            last_query_reason: None,
            active_stage: None,
            draw_cache: Vec::new(),
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
                self.active_stage = None;
                let mut compiler = Compiler::new(&mut self.program);
                compiler.compile_module(&module);
                self.var_map = compiler.into_var_map();
                Ok(())
            }
            Err(e) => Err(format!("Parse error: {:?}", e)),
        }
    }

    fn push_stage_rules(&mut self, stage_index: usize) {
        if stage_index < self.program.stages.len() {
            let rules = self.program.stages[stage_index].rules.clone();
            self.program.global_rules.extend(rules);
            self.active_stage = Some(stage_index);
        }
    }

    fn pop_stage_rules(&mut self) {
        if let Some(stage_index) = self.active_stage.take() {
            if stage_index < self.program.stages.len() {
                let count = self.program.stages[stage_index].rules.len();
                let new_len = self.program.global_rules.len().saturating_sub(count);
                self.program.global_rules.truncate(new_len);
            }
        }
    }

    pub fn query_batch(&mut self, query_str: &str, limit: usize) -> Result<Vec<String>, String> {
        self.query_batch_in_stage(query_str, limit, None)
    }

    pub fn query_batch_in_stage(
        &mut self,
        query_str: &str,
        limit: usize,
        stage_index: Option<usize>,
    ) -> Result<Vec<String>, String> {
        let max_steps = self.max_steps;
        if let Some(idx) = stage_index {
            self.push_stage_rules(idx);
        }

        let term_result = parser::parse_term(query_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => {
                if stage_index.is_some() {
                    self.pop_stage_rules();
                }
                return Err(format!("Query parse error: {:?}", e));
            }
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
        let solution_set = {
            let mut solver = Solver::new(&mut self.program);
            solver.collect_solutions(goal, self.strategy, limit, max_steps)
        };

        self.last_query_reason = Some(solution_set.reason.clone());

        let results = solution_set
            .solutions()
            .iter()
            .map(|s| format_solution(&query_vars, s, &self.program))
            .collect();

        if stage_index.is_some() {
            self.pop_stage_rules();
        }

        Ok(results)
    }

    pub fn query_start_global(&mut self, query_str: &str) -> Result<Option<String>, String> {
        self.query_start(query_str, None)
    }

    /// Start an incremental query, returning the first solution if one exists.
    ///
    /// Use `query_next()` to retrieve subsequent solutions.
    /// Use `query_stop()` to abandon the query.
    /// Use `has_more_solutions()` to check if more results are available.
    pub fn query_start(&mut self, query_str: &str, stage_index: Option<usize>) -> Result<Option<String>, String> {
        self.query_stop();

        if let Some(idx) = stage_index {
            self.push_stage_rules(idx);
        }

        let term_result = parser::parse_term(query_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => {
                self.pop_stage_rules();
                return Err(format!("Query parse error: {:?}", e));
            }
        };

        let (goal, query_vars) = Compiler::with_var_map(&mut self.program, self.var_map.clone())
            .compile_query(&term);

        self.pending_query_vars = query_vars;

        let mut solver = Solver::new(&mut self.program);
        let queue = solver.init_query(goal, self.strategy);
        let (solution, remaining_queue) = solver.step_until_solution(queue, self.max_steps);

        let found_solution = solution.is_some();
        let queue_exhausted = remaining_queue.is_empty();

        self.update_incremental_state(found_solution, queue_exhausted, remaining_queue);

        match solution {
            Some(state) => Ok(Some(format_solution(&self.pending_query_vars, &state, &self.program))),
            None => {
                if self.pending_queue.as_ref().is_none_or(|q| q.is_empty()) {
                    self.pending_queue = None;
                    self.pop_stage_rules();
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
            self.pop_stage_rules();
            return None;
        }

        let mut solver = Solver::new(&mut self.program);
        let (solution, remaining_queue) = solver.step_until_solution(queue, self.max_steps);

        let found_solution = solution.is_some();
        let queue_exhausted = remaining_queue.is_empty();

        self.update_incremental_state(found_solution, queue_exhausted, remaining_queue);

        if queue_exhausted {
            self.pop_stage_rules();
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
        self.pop_stage_rules();
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
            let solution_set = solver.collect_solutions(true_prop, self.strategy, 1, 1000);
            
            if let Some(solution) = solution_set.solutions().first() {
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
        // Collect up to 2 solutions to verify state constraints are deterministic (exactly 1 solution)
        let solution_set = solver.collect_solutions(combined_goal, self.strategy, 2, self.max_steps);
        let solutions = solution_set.solutions();

        if solutions.len() == 1 {
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
        } else if solutions.is_empty() {
            match solution_set.reason {
                TerminationReason::SearchExhausted => {
                    Err(format!(
                        "State constraint failure in stage '{}': no solutions found",
                        stage_name
                    ))
                }
                TerminationReason::MaxStepsReached => {
                    Err(format!(
                        "State constraint search hit step limit in stage '{}': inconclusive result",
                        stage_name
                    ))
                }
                TerminationReason::LimitReached => {
                    unreachable!("LimitReached with 0 solutions")
                }
            }
        } else {
            // solutions.len() >= 2
            match solution_set.reason {
                TerminationReason::LimitReached => {
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
                TerminationReason::MaxStepsReached => {
                    Err(format!(
                        "State constraint search hit step limit in stage '{}': non-determinism check inconclusive",
                        stage_name
                    ))
                }
                TerminationReason::SearchExhausted => {
                    unreachable!("SearchExhausted with 2+ solutions")
                }
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
        let solution_set = solver.collect_solutions(true_prop, self.strategy, 1, self.max_steps);
        
        if let Some(solution) = solution_set.solutions().first() {
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
        let solution_set = solver.collect_solutions(true_prop, self.strategy, 1, self.max_steps);
        
        let subst = solution_set.solutions().first().map(|s| &s.subst);
        
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

    fn update_incremental_state(&mut self, found_solution: bool, queue_exhausted: bool, remaining_queue: SearchQueue) {
        self.last_query_reason = Some(if !found_solution && !queue_exhausted {
            TerminationReason::MaxStepsReached  // No solution, queue has more work
        } else if found_solution && !queue_exhausted {
            TerminationReason::LimitReached     // Found solution, more available
        } else {
            TerminationReason::SearchExhausted  // No solution or found solution with search complete
        });

        self.pending_queue = if queue_exhausted {
            None
        } else {
            Some(remaining_queue)
        };
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

    pub fn add_fact(&mut self, fact_str: &str) -> Result<(), String> {
        let term_result = parser::parse_term(fact_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => return Err(format!("Fact parse error: {:?}", e)),
        };

        let prop = Compiler::with_var_map(&mut self.program, self.var_map.clone())
            .compile_fact(&term);
        self.program.facts.push(prop);
        Ok(())
    }

    pub fn clear_facts_by_relation(&mut self, relation_name: &str) {
        self.program.facts.retain(|&fact_id| {
            let prop = self.program.props.get(fact_id);
            if let Prop::App { rel, .. } = prop {
                let rel_data = self.program.rels.get(*rel);
                return rel_data.name != relation_name;
            }
            true
        });
    }

    pub fn collect_draws(&mut self, stage_index: usize) -> Result<Vec<DrawCommand>, String> {
        if stage_index >= self.program.stages.len() {
            return Err(format!("Stage index {} out of bounds", stage_index));
        }

        let directives = self.program.stages[stage_index].draw_directives.clone();
        let mut results = Vec::new();

        self.push_stage_rules(stage_index);

        for directive in &directives {
            let solution_set = {
                let mut solver = Solver::new(&mut self.program);
                solver.collect_solutions(directive.condition, self.strategy, 1000, self.max_steps)
            };

            for solution in solution_set.solutions() {
                for &draw_term in &directive.draws {
                    if let Some(cmd) = self.term_to_draw_command(draw_term, &solution.subst) {
                        results.push(cmd);
                    }
                }
            }
        }

        self.pop_stage_rules();
        Ok(results)
    }

    pub fn collect_draws_by_name(&mut self, name: &str) -> Result<Vec<DrawCommand>, String> {
        let stage_index = self
            .program
            .stages
            .iter()
            .position(|s| s.name == name)
            .ok_or_else(|| format!("Stage '{}' not found", name))?;
        self.collect_draws(stage_index)
    }

    fn term_to_draw_command(&self, term_id: TermId, subst: &Subst) -> Option<DrawCommand> {
        let walked = subst.walk(term_id, &self.program.terms);
        let term = self.program.terms.get(walked);

        if let Term::App { sym, args } = term {
            let name = self.program.symbols.get(*sym).clone();
            let float_args: Option<Vec<f32>> = args
                .iter()
                .map(|&arg_id| {
                    let walked_arg = subst.walk(arg_id, &self.program.terms);
                    self.extract_float(walked_arg)
                })
                .collect();

            float_args.map(|args| DrawCommand { name, args })
        } else {
            None
        }
    }

    fn extract_float(&self, term_id: TermId) -> Option<f32> {
        let term = self.program.terms.get(term_id);
        match term {
            Term::Float(f) => Some(*f),
            Term::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Find what state conditions would produce a specific draw command.
    /// Returns the With conditions (with variable bindings) that would emit the target.
    pub fn query_draw_condition(
        &mut self,
        _stage_index: usize,
        _target_draw: &str,
    ) -> Result<Vec<String>, String> {
        // Scan draw directives, unify target with each draw term,
        // return the conditions that would produce it
        todo!()
    }
}
