#![allow(clippy::needless_borrow)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nom::Finish;

use crate::ast::Module;
use crate::ast::parser;
use crate::solver::SearchStrategy;

use super::Frontend;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn create_frontend() -> *mut Frontend {
    Box::leak(Box::new(Frontend::new()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_frontend(frontend: *mut Frontend) {
    unsafe { std::ptr::drop_in_place(frontend) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_set_strategy(frontend: *mut Frontend, strategy: i32) {
    unsafe {
        (*frontend).strategy = match strategy {
            0 => SearchStrategy::BFS,
            1 => SearchStrategy::DFS,
            _ => SearchStrategy::BFS,
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_get_strategy(frontend: *mut Frontend) -> i32 {
    unsafe {
        match (*frontend).strategy {
            SearchStrategy::BFS => 0,
            SearchStrategy::DFS => 1,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_set_max_steps(frontend: *mut Frontend, max_steps: i32) {
    unsafe {
        (*frontend).max_steps = max_steps as usize;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_get_max_steps(frontend: *mut Frontend) -> i32 {
    unsafe {
        (*frontend).max_steps as i32
    }
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
        let result = (*frontend).query_batch(query_str, 10);
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
pub unsafe extern "C" fn frontend_query_start(frontend: *mut Frontend, query: *const c_char, stage_index: i32) -> *mut c_char {
    unsafe {
        let query_str = CStr::from_ptr(query).to_str().unwrap_or("");
        let stage = if stage_index >= 0 { Some(stage_index as usize) } else { None };
        let result = (*frontend).query_start(query_str, stage);
        let output = match result {
            Ok(Some(solution)) => solution,
            Ok(None) => "no".to_string(),
            Err(e) => format!("Error: {}", e),
        };
        CString::new(output).unwrap().into_raw()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_query_next(frontend: *mut Frontend) -> *mut c_char {
    unsafe {
        let result = (*frontend).query_next();
        let output = match result {
            Some(solution) => solution,
            None => "no".to_string(),
        };
        CString::new(output).unwrap().into_raw()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_has_more(frontend: *mut Frontend) -> i32 {
    unsafe {
        if (*frontend).has_more_solutions() { 1 } else { 0 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_query_reason(frontend: *mut Frontend) -> i32 {
    unsafe {
        match (*frontend).last_query_reason {
            Some(crate::solver::TerminationReason::LimitReached) => 0,
            Some(crate::solver::TerminationReason::SearchExhausted) => 1,
            Some(crate::solver::TerminationReason::MaxStepsReached) => 2,
            None => -1,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_query_stop(frontend: *mut Frontend) {
    unsafe {
        (*frontend).query_stop();
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_run_stage(frontend: *mut Frontend, stage_index: i32) -> i32 {
    unsafe {
        match (*frontend).run_stage(stage_index as usize) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_run_stage_by_name(frontend: *mut Frontend, name: *const c_char) -> i32 {
    unsafe {
        let name_str = CStr::from_ptr(name).to_str().unwrap_or("");
        match (*frontend).run_stage_by_name(name_str) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_get_state_var(frontend: *mut Frontend, name: *const c_char) -> *mut c_char {
    unsafe {
        let name_str = CStr::from_ptr(name).to_str().unwrap_or("");
        let value = (*frontend).get_state_var(name_str).unwrap_or_default();
        CString::new(value).unwrap().into_raw()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_state_var_count(frontend: *mut Frontend) -> i32 {
    unsafe { (*frontend).program.state_vars.len() as i32 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_state_var_name(frontend: *mut Frontend, index: i32) -> *mut c_char {
    unsafe {
        if let Some(name) = (&(*frontend).program.state_vars).get(index as usize) {
            CString::new(name.clone()).unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_state_var_value(frontend: *mut Frontend, index: i32) -> *mut c_char {
    unsafe {
        if let Some(name) = (&(*frontend).program.state_vars).get(index as usize).cloned() {
            let value = (*frontend).get_state_var(&name).unwrap_or_default();
            CString::new(value).unwrap().into_raw()
        } else {
            CString::new("").unwrap().into_raw()
        }
    }
}

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_query_batch(
    frontend: *mut Frontend,
    query: *const c_char,
    stage_index: i32,
    limit: i32,
) -> *mut c_char {
    unsafe {
        let query_str = CStr::from_ptr(query).to_str().unwrap_or("");
        let stage = if stage_index >= 0 { Some(stage_index as usize) } else { None };
        let result = (*frontend).query_batch_in_stage(query_str, limit as usize, stage);
        let output = match result {
            Ok(solutions) => {
                if solutions.is_empty() {
                    "".to_string()
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
pub unsafe extern "C" fn frontend_add_fact(frontend: *mut Frontend, fact: *const c_char) -> i32 {
    unsafe {
        let fact_str = CStr::from_ptr(fact).to_str().unwrap_or("");
        match (*frontend).add_fact(fact_str) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_clear_facts_by_relation(frontend: *mut Frontend, relation: *const c_char) {
    unsafe {
        let relation_str = CStr::from_ptr(relation).to_str().unwrap_or("");
        (*frontend).clear_facts_by_relation(relation_str);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_collect_draws(frontend: *mut Frontend, stage_index: i32) -> i32 {
    unsafe {
        match (*frontend).collect_draws(stage_index as usize) {
            Ok(draws) => {
                let count = draws.len() as i32;
                (*frontend).draw_cache = draws;
                count
            }
            Err(_) => -1,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_collect_draws_by_name(frontend: *mut Frontend, name: *const c_char) -> i32 {
    unsafe {
        let name_str = CStr::from_ptr(name).to_str().unwrap_or("");
        match (*frontend).collect_draws_by_name(name_str) {
            Ok(draws) => {
                let count = draws.len() as i32;
                (*frontend).draw_cache = draws;
                count
            }
            Err(_) => -1,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_draw_command_name(frontend: *mut Frontend, index: i32) -> *mut c_char {
    unsafe {
        if let Some(cmd) = (&(*frontend).draw_cache).get(index as usize) {
            CString::new(cmd.name.clone()).unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_draw_command_arg_count(frontend: *mut Frontend, index: i32) -> i32 {
    unsafe {
        if let Some(cmd) = (&(*frontend).draw_cache).get(index as usize) {
            cmd.args.len() as i32
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn frontend_draw_command_arg(frontend: *mut Frontend, index: i32, arg_index: i32) -> f32 {
    unsafe {
        if let Some(cmd) = (&(*frontend).draw_cache).get(index as usize) {
            if let Some(&arg) = cmd.args.get(arg_index as usize) {
                return arg;
            }
        }
        0.0
    }
}
