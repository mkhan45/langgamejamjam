use std::collections::HashMap;
use z3::ast::{Ast, Bool, Dynamic, Int, Real};
use z3::{FuncDecl, Model, SatResult, Solver, Sort};

use crate::ast::{Rel, Rule, Stage, Term, TermContents};

pub struct GameState {
    solver: Solver,
    relations: HashMap<String, usize>,
    variables: HashMap<String, Dynamic>,
}

impl GameState {
    pub fn new() -> Self {
        let solver = Solver::new();
        Self {
            solver,
            relations: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    pub fn load_stage(&mut self, stage: &Stage) {
        for rule in &stage.rules {
            self.assert_rule(rule);
        }
    }

    pub fn assert_fact(&mut self, term: &Term) {
        let z3_term = self.compile_term_as_bool(term);
        self.solver.assert(&z3_term);
    }

    pub fn assert_rule(&mut self, rule: &Rule) {
        let premise = self.compile_term_as_bool(&rule.premise);
        let conclusion = self.compile_term_as_bool(&rule.conclusion);

        let vars: Vec<Dynamic> = self.variables.values().cloned().collect();

        if vars.is_empty() {
            let implication = premise.implies(&conclusion);
            self.solver.assert(&implication);
        } else {
            let var_refs: Vec<&dyn Ast> = vars.iter().map(|v| v as &dyn Ast).collect();
            let implication = premise.implies(&conclusion);
            let forall = z3::ast::forall_const(&var_refs, &[], &implication);
            self.solver.assert(&forall);
        }

        self.variables.clear();
    }

    pub fn compile_term(&mut self, term: &Term) -> Dynamic {
        match &term.contents {
            TermContents::Int { val } => Int::from_i64((*val).into()).into(),

            TermContents::Float { val } => {
                let (num, den) = float_to_rational(*val);
                Real::from_rational(num, den).into()
            }

            TermContents::Var { name } => {
                if let Some(var) = self.variables.get(name) {
                    var.clone()
                } else {
                    let var: Dynamic = Int::new_const(name.as_str()).into();
                    self.variables.insert(name.clone(), var.clone());
                    var
                }
            }

            TermContents::Atom { text } => Int::new_const(text.as_str()).into(),

            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                if let Some(builtin) = self.compile_builtin(rel_name, args) {
                    return builtin;
                }

                let compiled_args: Vec<Dynamic> =
                    args.iter().map(|arg| self.compile_term(arg)).collect();

                let func = self.get_or_create_relation(rel_name, compiled_args.len());
                let arg_refs: Vec<&dyn Ast> = compiled_args.iter().map(|a| a as &dyn Ast).collect();
                func.apply(&arg_refs)
            }
        }
    }

    pub fn compile_term_as_bool(&mut self, term: &Term) -> Bool {
        let dynamic = self.compile_term(term);
        if let Some(b) = dynamic.as_bool() {
            b
        } else {
            Bool::from_bool(true)
        }
    }

    fn compile_builtin(&mut self, name: &str, args: &[Term]) -> Option<Dynamic> {
        match name {
            "add" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(Int::add(&[&l, &r]).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(Real::add(&[&l, &r]).into())
                } else {
                    None
                }
            }

            "sub" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(Int::sub(&[&l, &r]).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(Real::sub(&[&l, &r]).into())
                } else {
                    None
                }
            }

            "mul" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(Int::mul(&[&l, &r]).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(Real::mul(&[&l, &r]).into())
                } else {
                    None
                }
            }

            "div" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(l.div(&r).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(l.div(&r).into())
                } else {
                    None
                }
            }

            "eq" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                Some(left.eq(&right).into())
            }

            "neq" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                Some(left.eq(&right).not().into())
            }

            "lt" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(l.lt(&r).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(l.lt(&r).into())
                } else {
                    None
                }
            }

            "le" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(l.le(&r).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(l.le(&r).into())
                } else {
                    None
                }
            }

            "gt" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(l.gt(&r).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(l.gt(&r).into())
                } else {
                    None
                }
            }

            "ge" if args.len() == 2 => {
                let left = self.compile_term(&args[0]);
                let right = self.compile_term(&args[1]);
                if let (Some(l), Some(r)) = (left.as_int(), right.as_int()) {
                    Some(l.ge(&r).into())
                } else if let (Some(l), Some(r)) = (left.as_real(), right.as_real()) {
                    Some(l.ge(&r).into())
                } else {
                    None
                }
            }

            "and" if args.len() == 2 => {
                let left = self.compile_term_as_bool(&args[0]);
                let right = self.compile_term_as_bool(&args[1]);
                Some(Bool::and(&[&left, &right]).into())
            }

            "or" if args.len() == 2 => {
                let left = self.compile_term_as_bool(&args[0]);
                let right = self.compile_term_as_bool(&args[1]);
                Some(Bool::or(&[&left, &right]).into())
            }

            "not" if args.len() == 1 => {
                let inner = self.compile_term_as_bool(&args[0]);
                Some(inner.not().into())
            }

            "implies" if args.len() == 2 => {
                let left = self.compile_term_as_bool(&args[0]);
                let right = self.compile_term_as_bool(&args[1]);
                Some(left.implies(&right).into())
            }

            _ => None,
        }
    }

    fn get_or_create_relation(&mut self, name: &str, arity: usize) -> FuncDecl {
        self.relations.insert(name.to_string(), arity);

        let int_sort = Sort::int();
        let bool_sort = Sort::bool();

        let domain: Vec<Sort> = (0..arity).map(|_| int_sort.clone()).collect();
        let domain_refs: Vec<&Sort> = domain.iter().collect();
        FuncDecl::new(name, &domain_refs, &bool_sort)
    }

    pub fn check(&self) -> SatResult {
        self.solver.check()
    }

    pub fn get_model(&self) -> Option<Model> {
        self.solver.get_model()
    }

    pub fn query(&mut self, term: &Term) -> QueryResult {
        self.solver.push();
        let constraint = self.compile_term_as_bool(term);
        self.solver.assert(&constraint);

        let result = match self.solver.check() {
            SatResult::Sat => {
                let model = self.solver.get_model();
                QueryResult::Sat { model }
            }
            SatResult::Unsat => QueryResult::Unsat,
            SatResult::Unknown => QueryResult::Unknown,
        };

        self.solver.pop(1);
        self.variables.clear();
        result
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

pub enum QueryResult {
    Sat { model: Option<Model> },
    Unsat,
    Unknown,
}

fn float_to_rational(f: f32) -> (i64, i64) {
    const PRECISION: i64 = 1_000_000;
    let num = (f * PRECISION as f32) as i64;
    let gcd = gcd(num.abs(), PRECISION);
    (num / gcd, PRECISION / gcd)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parser::parse_term;

    fn parse(input: &str) -> Term {
        let span = nom_locate::LocatedSpan::new(input);
        parse_term(span).unwrap().1
    }

    #[test]
    fn test_simple_fact() {
        let mut state = GameState::new();

        let fact = parse("position(player, 0, 0)");
        state.assert_fact(&fact);

        assert_eq!(state.check(), SatResult::Sat);
    }

    #[test]
    fn test_arithmetic() {
        let mut state = GameState::new();

        let constraint = parse("eq(add(1, 2), 3)");
        state.assert_fact(&constraint);

        assert_eq!(state.check(), SatResult::Sat);
    }

    #[test]
    fn test_query_with_variable() {
        let mut state = GameState::new();

        let fact = parse("position(player, 5, 10)");
        state.assert_fact(&fact);

        let query = parse("position(player, X, Y)");
        match state.query(&query) {
            QueryResult::Sat { model: Some(m) } => {
                println!("Model: {}", m);
            }
            _ => panic!("Expected sat with model"),
        }
    }
}
