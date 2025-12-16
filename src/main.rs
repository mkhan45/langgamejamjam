pub mod ast;
pub mod ir;
pub mod solver;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nom::Finish;
use ast::parser;

use z3::SatResult;

use ast::Module;
use solver::{GameState, QueryResult};

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
pub extern "C" fn create_game_state() -> *mut GameState {
    Box::leak(Box::new(GameState::new()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_game_state(state: *mut GameState) {
    unsafe { std::ptr::drop_in_place(state) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_state_reset(state: *mut GameState) {
    unsafe { (*state).reset() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_state_load_facts(state: *mut GameState, module: *mut Module) {
    unsafe {
        for fact in &(*module).facts {
            (*state).assert_fact(fact);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_state_load_global(state: *mut GameState, module: *mut Module) {
    unsafe {
        (*state).load_stage(&(*module).global_stage);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_state_load_stage(state: *mut GameState, module: *mut Module, index: i32) {
    unsafe {
        if let Some(stage) = (&(*module).stages).get(index as usize) {
            (*state).load_stage(stage);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_state_check(state: *mut GameState) -> i32 {
    unsafe {
        match (*state).check() {
            SatResult::Sat => 0,
            SatResult::Unsat => 1,
            SatResult::Unknown => 2,
        }
    }
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
pub unsafe extern "C" fn game_state_query(state: *mut GameState, query: *const c_char) -> *mut c_char {
    unsafe {
        let query_str = CStr::from_ptr(query).to_str().unwrap_or("");
        let term_result = parser::parse_term(query_str.into()).finish();

        let term = match term_result {
            Ok((_, term)) => term,
            Err(_) => {
                return CString::new("Parse error").unwrap().into_raw();
            }
        };

        let result = (*state).query(&term);
        let output = match result {
            QueryResult::Sat { model: Some(m) } => format!("Sat\n{}", m),
            QueryResult::Sat { model: None } => "Sat (no model)".to_string(),
            QueryResult::Unsat => "Unsat".to_string(),
            QueryResult::Unknown => "Unknown".to_string(),
        };

        CString::new(output).unwrap().into_raw()
    }
}

fn main() {
    // let args: Vec<String> = env::args().collect();

    // if args.len() != 2 {
    //     eprintln!("Usage: {} <input_file>", args[0]);
    //     process::exit(1);
    // }

    // let filename = &args[1];

    // let contents = fs::read_to_string(filename)
    //     .unwrap_or_else(|err| {
    //         eprintln!("Error reading file '{}': {}", filename, err);
    //         process::exit(1);
    //     });

    // let input = LocatedSpan::new(contents.as_str());

    // match parse_module(input) {
    //     Ok((remaining, module)) => {
    //         println!("Successfully parsed module:");
    //         println!();
    //         print!("{}", module);
    //         println!();

    //         if !remaining.fragment().trim().is_empty() {
    //             println!("Remaining input: {:?}", remaining.fragment());
    //         }
    //     }
    //     Err(err) => {
    //         eprintln!("Error parsing module: {:?}", err);
    //         process::exit(1);
    //     }
    // }
}
