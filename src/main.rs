pub mod ast;
pub mod solve;

use std::env;
use std::fs;
use std::process;
use nom_locate::LocatedSpan;
use ast::parser::parse_module;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <input_file>", args[0]);
        process::exit(1);
    }

    let filename = &args[1];

    let contents = fs::read_to_string(filename)
        .unwrap_or_else(|err| {
            eprintln!("Error reading file '{}': {}", filename, err);
            process::exit(1);
        });

    let input = LocatedSpan::new(contents.as_str());

    match parse_module(input) {
        Ok((remaining, module)) => {
            println!("Successfully parsed module:");
            println!();
            print!("{}", module);
            println!();

            if !remaining.fragment().trim().is_empty() {
                println!("Remaining input: {:?}", remaining.fragment());
            }
        }
        Err(err) => {
            eprintln!("Error parsing module: {:?}", err);
            process::exit(1);
        }
    }
}
