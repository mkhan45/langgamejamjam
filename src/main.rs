pub mod ast;
pub mod session;
pub mod solve;
pub mod solver;

pub use session::GameSession;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nom::Finish;
use ast::parser;

use z3;

use ast::Module;

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
