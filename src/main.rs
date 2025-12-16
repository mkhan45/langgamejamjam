pub mod ast;
pub mod ir;
pub mod solver;
mod test_samelength;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nom::Finish;
use ast::parser;
use ast::compile::Compiler;

use ast::Module;
use ir::Program;
use solver::{format_solution, Solver};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn parse_module(input: *const c_char) -> *mut Module {
    unsafe {
        let inp = CStr::from_ptr(input).to_str().unwrap_or("");
        let res = parser::parse_module(inp.into()).finish();
        match res {
            Ok((_rest, module)) => Box::leak(Box::new(module)) as *mut _,
            Err(_e) => std::ptr::null::<Module>() as *mut _,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn module_to_string(module: *mut Module) -> *mut c_char {
    unsafe {
        let s = CString::new((*module).to_string()).unwrap();
        s.into_raw()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_module(module: *mut Module) {
    unsafe { std::ptr::drop_in_place(module) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn module_stage_count(module: *mut Module) -> i32 {
    unsafe { (*module).stages.len() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn module_get_stage_name(module: *mut Module, index: i32) -> *mut c_char {
    unsafe {
        if let Some(stage) = (&(*module).stages).get(index as usize) {
            CString::new(stage.name.clone()).unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }
}

pub struct Frontend {
    pub program: Program,
    pub var_map: std::collections::HashMap<String, ir::TermId>,
}

impl Frontend {
    pub fn new() -> Self {
        Self {
            program: Program::default(),
            var_map: std::collections::HashMap::new(),
        }
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

    pub fn query(&mut self, query_str: &str) -> Result<Vec<String>, String> {
        self.query_with_limit(query_str, 10)
    }

    pub fn query_with_limit(&mut self, query_str: &str, limit: usize) -> Result<Vec<String>, String> {
        let term_result = parser::parse_term(query_str.into()).finish();
        let term = match term_result {
            Ok((_, term)) => term,
            Err(e) => return Err(format!("Query parse error: {:?}", e)),
        };

        let (goal, query_vars) = Compiler::with_var_map(&mut self.program, self.var_map.clone())
            .compile_query(&term);

        let mut solver = Solver::new(&mut self.program);
        let solutions: Vec<_> = solver.query(goal).with_limit(limit).collect();

        Ok(solutions
            .iter()
            .map(|s| format_solution(&query_vars, s, solver.program))
            .collect())
    }
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn create_frontend() -> *mut Frontend {
    Box::leak(Box::new(Frontend::new()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_frontend(frontend: *mut Frontend) {
    unsafe { std::ptr::drop_in_place(frontend) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_load(frontend: *mut Frontend, source: *const c_char) -> i32 {
    unsafe {
        let source_str = CStr::from_ptr(source).to_str().unwrap_or("");
        match (*frontend).load(source_str) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_query(frontend: *mut Frontend, query: *const c_char) -> *mut c_char {
    unsafe {
        let query_str = CStr::from_ptr(query).to_str().unwrap_or("");
        let result = (*frontend).query(query_str);
        let output = match result {
            Ok(solutions) => {
                if solutions.is_empty() {
                    "no".to_string()
                } else {
                    solutions.join("\n")
                }
            }
            Err(e) => format!("Error: {}", e),
        };
        CString::new(output).unwrap().into_raw()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_fact_count(frontend: *mut Frontend) -> i32 {
    unsafe { (*frontend).program.facts.len() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_rule_count(frontend: *mut Frontend) -> i32 {
    unsafe { (*frontend).program.global_rules.len() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_stage_count(frontend: *mut Frontend) -> i32 {
    unsafe { (*frontend).program.stages.len() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_stage_name(frontend: *mut Frontend, index: i32) -> *mut c_char {
    unsafe {
        if let Some(stage) = (&(*frontend).program.stages).get(index as usize) {
            CString::new(stage.name.clone()).unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }
}

fn main() {}

#[cfg(test)]
mod frontend_tests {
    use super::*;

    #[test]
    fn test_position_query() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    position(player, 0, 0)
End Facts

Begin Global:
End Global
"#).unwrap();

        eprintln!("Facts: {}", frontend.program.facts.len());
        for &fact_prop_id in &frontend.program.facts {
            let fact_prop = frontend.program.props.get(fact_prop_id);
            eprintln!("  fact: {:?}", fact_prop);
        }
        eprintln!("Rels:");
        for (id, rel) in frontend.program.rels.iter() {
            eprintln!("  {:?}: {:?}", id, rel);
        }
        
        let result = frontend.query("position(player, X, Y)").unwrap();
        eprintln!("Query result: {:?}", result);
        assert!(!result.is_empty(), "Expected at least one solution");
    }

    #[test]
    fn test_eq_fact_constrains_query() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    eq(X, 1)
End Facts

Begin Global:
End Global
"#).unwrap();

        let result = frontend.query("eq(X, 2)").unwrap();
        assert!(result.is_empty(), "eq(X, 2) should fail when eq(X, 1) is a fact, got: {:?}", result);
    }

    #[test]
    fn test_eq_fact_allows_compatible_query() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    eq(X, 1)
End Facts

Begin Global:
End Global
"#).unwrap();

        let result = frontend.query("eq(X, Y)").unwrap();
        assert!(!result.is_empty(), "eq(X, Y) should succeed with Y=1 when eq(X, 1) is a fact");
    }

    #[test]
    fn test_eq_with_cons_term() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    eq(X, cons(A, B))
End Facts

Begin Global:
End Global
"#).unwrap();

        let result = frontend.query("eq(X, Y)").unwrap();
        assert!(!result.is_empty(), "eq(X, cons(A, B)) should work with relational solver");
    }
}
