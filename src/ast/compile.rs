use std::collections::HashMap;

use crate::ast::{Module, Rel, Rule, Stage, Term, TermContents};
use crate::ast::parser::{self, Span};
use crate::solver::ir::{
    Clause, DrawDirective as IrDrawDirective, Program, Prop, PropId, RelId, RelInfo, RelKind,
    Stage as IrStage, SymbolId, Term as IRTerm, TermId, Var,
};
use nom::Finish;
use nom::Parser;
use nom::multi::many0;
use nom::character::complete::multispace0;

const STDLIB: &str = include_str!("../stdlib.l");

fn parse_stdlib_rules() -> Vec<Rule> {
    let input: Span = STDLIB.into();
    let result = many0(|s| {
        let (s, _) = multispace0(s)?;
        parser::parse_rule(s)
    }).parse(input);
    match result.finish() {
        Ok((_, rules)) => rules,
        Err(_) => panic!("Failed to parse stdlib"),
    }
}

const SMT_INT_RELATIONS: &[(&str, usize)] = &[
    ("int_eq", 2),
    ("int_neq", 2),
    ("int_lt", 2),
    ("int_le", 2),
    ("int_gt", 2),
    ("int_ge", 2),
    ("int_add", 3),
    ("int_sub", 3),
    ("int_mul", 3),
    ("int_div", 3),
];

const SMT_REAL_RELATIONS: &[(&str, usize)] = &[
    ("real_eq", 2),
    ("real_neq", 2),
    ("real_lt", 2),
    ("real_le", 2),
    ("real_gt", 2),
    ("real_ge", 2),
    ("real_add", 3),
    ("real_sub", 3),
    ("real_mul", 3),
    ("real_div", 3),
];

pub struct Compiler<'a> {
    program: &'a mut Program,
    rel_map: HashMap<String, RelId>,
    var_map: HashMap<String, TermId>,
    next_var_map: HashMap<String, TermId>,
}

impl<'a> Compiler<'a> {
    pub fn new(program: &'a mut Program) -> Self {
        let mut rel_map = HashMap::new();
        for (id, rel_info) in program.rels.iter() {
            rel_map.insert(rel_info.name.clone(), id);
        }

        let mut compiler = Self {
            program,
            rel_map,
            var_map: HashMap::new(),
            next_var_map: HashMap::new(),
        };
        compiler.register_builtin_relations();
        compiler
    }

    pub fn with_var_map(program: &'a mut Program, var_map: HashMap<String, TermId>) -> Self {
        let mut rel_map = HashMap::new();
        for (id, rel_info) in program.rels.iter() {
            rel_map.insert(rel_info.name.clone(), id);
        }

        let mut compiler = Self {
            program,
            rel_map,
            var_map,
            next_var_map: HashMap::new(),
        };
        compiler.register_builtin_relations();
        compiler
    }

    pub fn into_var_map(self) -> HashMap<String, TermId> {
        self.var_map
    }

    fn register_builtin_relations(&mut self) {
        for &(name, arity) in SMT_INT_RELATIONS {
            self.get_or_create_rel(name, arity, RelKind::SMTInt);
        }
        for &(name, arity) in SMT_REAL_RELATIONS {
            self.get_or_create_rel(name, arity, RelKind::SMTReal);
        }
    }

    fn get_or_create_rel(&mut self, name: &str, arity: usize, kind: RelKind) -> RelId {
        if let Some(&id) = self.rel_map.get(name) {
            return id;
        }
        let info = RelInfo {
            name: name.to_string(),
            arity,
            kind,
        };
        let id = self.program.rels.alloc(info);
        self.rel_map.insert(name.to_string(), id);
        id
    }

    fn get_or_create_var(&mut self, name: &str) -> TermId {
        if let Some(&id) = self.var_map.get(name) {
            return id;
        }
        let var = Var {
            name: name.to_string(),
        };
        let var_id = self.program.vars.alloc(var);
        let term_id = self.program.terms.alloc(IRTerm::Var(var_id));
        self.var_map.insert(name.to_string(), term_id);
        term_id
    }

    fn intern_symbol(&mut self, s: &str) -> SymbolId {
        self.program.symbols.intern(s.to_string())
    }

    fn alloc_term(&mut self, term: IRTerm) -> TermId {
        self.program.terms.alloc(term)
    }

    fn alloc_prop(&mut self, prop: Prop) -> PropId {
        self.program.props.alloc(prop)
    }

    fn clear_scope(&mut self) {
        self.var_map.clear();
        self.next_var_map.clear();
    }

    fn get_or_create_next_var(&mut self, name: &str) -> TermId {
        if let Some(&id) = self.next_var_map.get(name) {
            return id;
        }
        let next_name = format!("_next_{}", name);
        let var = Var {
            name: next_name.clone(),
        };
        let var_id = self.program.vars.alloc(var);
        let term_id = self.program.terms.alloc(IRTerm::Var(var_id));
        self.next_var_map.insert(name.to_string(), term_id);
        term_id
    }

    fn lower_simple_term(&mut self, term: &Term) -> TermId {
        match &term.contents {
            TermContents::Var { name } => self.get_or_create_var(name),
            TermContents::Atom { text } => {
                let sym_id = self.intern_symbol(text);
                self.alloc_term(IRTerm::Atom(sym_id))
            }
            TermContents::Int { val } => self.alloc_term(IRTerm::Int(*val)),
            TermContents::Float { val } => self.alloc_term(IRTerm::Float(*val)),
            TermContents::App { .. } => {
                panic!("lower_simple_term called on App - use lower_term_to_prop instead")
            }
        }
    }

    fn lower_term_arg(&mut self, term: &Term) -> TermId {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                if rel_name == "next" && args.len() == 1
                    && let TermContents::Var { name } = &args[0].contents
                {
                    return self.get_or_create_next_var(name);
                }

                let lowered_args: Vec<TermId> = args
                    .iter()
                    .map(|a| self.lower_term_arg(a))
                    .collect();

                let sym = self.intern_symbol(rel_name);
                self.alloc_term(IRTerm::App { sym, args: lowered_args })
            }
            _ => self.lower_simple_term(term),
        }
    }

    fn is_smt_relation(&self, name: &str) -> bool {
        SMT_INT_RELATIONS.iter().any(|(n, _)| *n == name)
            || SMT_REAL_RELATIONS.iter().any(|(n, _)| *n == name)
    }

    fn smt_kind(&self, name: &str) -> RelKind {
        if SMT_INT_RELATIONS.iter().any(|(n, _)| *n == name) {
            RelKind::SMTInt
        } else {
            RelKind::SMTReal
        }
    }

    fn extract_conclusions(term: &Term) -> Vec<&Term> {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };
                if rel_name == "each" {
                    args.iter()
                        .flat_map(|arg| Self::extract_conclusions(arg))
                        .collect()
                } else {
                    vec![term]
                }
            }
            _ => vec![term],
        }
    }

    fn lower_term_to_prop(&mut self, term: &Term) -> PropId {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                match rel_name {
                    "and" => {
                        let lhs_prop = self.lower_term_to_prop(&args[0]);
                        let rhs_prop = self.lower_term_to_prop(&args[1]);
                        let and_prop = Prop::And(lhs_prop, rhs_prop);
                        self.alloc_prop(and_prop)
                    }
                    "or" => {
                        let lhs_prop = self.lower_term_to_prop(&args[0]);
                        let rhs_prop = self.lower_term_to_prop(&args[1]);
                        let or_prop = Prop::Or(lhs_prop, rhs_prop);
                        self.alloc_prop(or_prop)
                    }
                    "cond" => {
                        let guard_prop = self.lower_term_to_prop(&args[0]);
                        let lhs_prop = self.lower_term_to_prop(&args[1]);
                        let rhs_prop = self.lower_term_to_prop(&args[2]);
                        let cond_prop = Prop::Cond(guard_prop, lhs_prop, rhs_prop);
                        self.alloc_prop(cond_prop)
                    }
                    "not" => {
                        let prop = self.lower_term_to_prop(&args[0]);
                        let not_prop = Prop::Not(prop);
                        self.alloc_prop(not_prop)
                    }
                    "eq" if args.len() == 2 => {
                        let t1 = self.lower_term_arg(&args[0]);
                        let t2 = self.lower_term_arg(&args[1]);
                        let eq_prop = Prop::Eq(t1, t2);
                        self.alloc_prop(eq_prop)
                    }
                    _ => {
                        let lowered_args: Vec<TermId> = args
                            .iter()
                            .map(|a| self.lower_term_arg(a))
                            .collect();

                        let arity = lowered_args.len();
                        let kind = if self.is_smt_relation(rel_name) {
                            self.smt_kind(rel_name)
                        } else {
                            RelKind::User
                        };

                        let rel_id = self.get_or_create_rel(rel_name, arity, kind);

                        let app_prop = Prop::App {
                            rel: rel_id,
                            args: lowered_args,
                        };
                        self.alloc_prop(app_prop)
                    }
                }
            }
            _ => {
                self.alloc_prop(Prop::True)
            }
        }
    }

    fn lower_fact(&mut self, term: &Term) -> PropId {
        self.lower_term_to_prop(term)
    }

    pub fn compile_fact(&mut self, term: &Term) -> PropId {
        self.lower_term_to_prop(term)
    }

    fn lower_conclusion(&mut self, conclusion: &Term) -> (RelId, Vec<TermId>) {
        match &conclusion.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                let lowered_args: Vec<TermId> = args
                    .iter()
                    .map(|a| self.lower_term_arg(a))
                    .collect();

                let arity = lowered_args.len();
                let rel_id = self.get_or_create_rel(rel_name, arity, RelKind::User);

                (rel_id, lowered_args)
            }
            _ => {
                panic!("Non-rel-app in conclusion")
            }
        }
    }

    fn lower_rule(&mut self, rule: &Rule, fact_var_map: &HashMap<String, TermId>) -> Vec<Clause> {
        self.clear_scope();

        // Restore fact variables so rules can reference state variables from facts
        for (name, term_id) in fact_var_map {
            self.var_map.insert(name.clone(), *term_id);
        }

        let premise_body = self.lower_term_to_prop(&rule.premise);
        let conclusions = Self::extract_conclusions(&rule.conclusion);

        conclusions
            .into_iter()
            .enumerate()
            .map(|(i, conclusion)| {
                let (head_rel, head_args) = self.lower_conclusion(conclusion);
                let name = if i > 0 {
                    format!("{}${}", rule.name, i)
                } else {
                    rule.name.clone()
                };
                Clause {
                    name,
                    head_rel,
                    head_args,
                    body: premise_body,
                }
            })
            .collect()
    }

    fn lower_draw_directive(
        &mut self,
        directive: &crate::ast::DrawDirective,
        fact_var_map: &HashMap<String, TermId>,
    ) -> IrDrawDirective {
        self.var_map = fact_var_map.clone();

        let condition = match &directive.condition {
            Some(term) => self.lower_term_to_prop(term),
            None => self.alloc_prop(Prop::True),
        };

        let draws = directive
            .draws
            .iter()
            .map(|t| self.lower_term_arg(t))
            .collect();

        IrDrawDirective { condition, draws }
    }

    fn lower_stage(&mut self, stage: &Stage, fact_var_map: &HashMap<String, TermId>) -> IrStage {
        let rules = stage.rules.iter().flat_map(|r| self.lower_rule(r, fact_var_map)).collect();

        self.var_map = fact_var_map.clone();
        self.next_var_map.clear();
        let state_constraints = stage.state_constraints
            .iter()
            .map(|t| self.lower_term_to_prop(t))
            .collect();

        let next_var_map = self.next_var_map.clone();

        let draw_directives = stage
            .draw_directives
            .iter()
            .map(|d| self.lower_draw_directive(d, fact_var_map))
            .collect();

        IrStage {
            name: stage.name.clone(),
            rules,
            state_constraints,
            next_var_map,
            draw_directives,
        }
    }

    pub fn compile_query(&mut self, term: &Term) -> (PropId, Vec<(String, TermId)>) {
        let prop_id = self.lower_term_to_prop(term);
        let query_vars: Vec<(String, TermId)> = self.var_map.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        (prop_id, query_vars)
    }

    pub fn compile_module(&mut self, module: &Module) {
        self.program.state_vars = module.state_vars.clone();

        for fact_term in &module.facts {
            let fact_prop = self.lower_fact(fact_term);
            self.program.facts.push(fact_prop);
        }

        let fact_var_map = self.var_map.clone();
        
        for state_var_name in &module.state_vars {
            if let Some(&term_id) = fact_var_map.get(state_var_name) {
                self.program.state_var_term_ids.insert(state_var_name.clone(), term_id);
            }
        }

        for rule in &module.global_stage.rules {
            let clauses = self.lower_rule(rule, &fact_var_map);
            self.program.global_rules.extend(clauses);
        }

        for rule in parse_stdlib_rules() {
            let clauses = self.lower_rule(&rule, &fact_var_map);
            self.program.global_rules.extend(clauses);
        }

        for stage in &module.stages {
            let ir_stage = self.lower_stage(stage, &fact_var_map);
            self.program.stages.push(ir_stage);
        }

        self.var_map = fact_var_map;
    }
}

pub fn compile(module: &Module) -> Program {
    let mut program = Program::default();
    Compiler::new(&mut program).compile_module(module);
    program
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parser;
    use nom::Finish;

    fn parse_and_compile(input: &str) -> Program {
        let result = parser::parse_module(input.into()).finish();
        let (_, module) = result.expect("parse failed");
        compile(&module)
    }

    #[test]
    fn test_simple_fact() {
        let input = r#"Begin Facts:
    position(player, 0, 0)
End Facts

Begin Global:
End Global
"#;
        let program = parse_and_compile(input);
        assert_eq!(program.facts.len(), 1);
        let fact_prop = program.props.get(program.facts[0]);
        match fact_prop {
            Prop::App { rel, args } => {
                assert_eq!(program.rels.get(*rel).name, "position");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected Prop::App"),
        }
    }

    #[test]
    fn test_simple_rule() {
        let input = r#"Begin Facts:
End Facts

Begin Global:
Rule MoveRight:
    position(player, X, Y)
    ----------------------
    position(player, X, Y)
End Global
"#;
        let program = parse_and_compile(input);
        let clause = program.global_rules.iter().find(|c| c.name == "MoveRight").unwrap();
        assert_eq!(program.rels.get(clause.head_rel).name, "position");
    }

    #[test]
    fn test_smt_relation_in_rule() {
        let input = r#"Begin Facts:
End Facts

Begin Global:
Rule Increment:
    int_add(X, 1, Y)
    ----------------
    count(Y)
End Global
"#;
        let program = parse_and_compile(input);

        let clause = program.global_rules.iter().find(|c| c.name == "Increment").unwrap();
        let body_prop = program.props.get(clause.body);
        match body_prop {
            Prop::App { rel, args } => {
                let rel_info = program.rels.get(*rel);
                assert_eq!(rel_info.name, "int_add");
                assert_eq!(rel_info.kind, RelKind::SMTInt);
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected App prop"),
        }
    }

    #[test]
    fn test_stage_compilation() {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global

Begin Stage Movement:
Rule Left:
    position(X)
    -----------
    moved(X)
End Stage Movement
"#;
        let program = parse_and_compile(input);
        assert_eq!(program.stages.len(), 1);
        assert_eq!(program.stages[0].name, "Movement");
        assert_eq!(program.stages[0].rules.len(), 1);
    }

    #[test]
    fn test_each_expands_to_multiple_clauses() {
        let input = r#"Begin Facts:
End Facts

Begin Global:
Rule Draw:
    alive()
    -------
    each(rect(1, 2), rect(3, 4))
End Global
"#;
        let program = parse_and_compile(input);
        let clauses: Vec<_> = program.global_rules.iter()
            .filter(|c| c.name == "Draw" || c.name.starts_with("Draw$"))
            .collect();
        assert_eq!(clauses.len(), 2);
        assert_eq!(program.rels.get(clauses[0].head_rel).name, "rect");
        assert_eq!(program.rels.get(clauses[1].head_rel).name, "rect");
    }

    #[test]
    fn test_each_nested_flattens() {
        let input = r#"Begin Facts:
End Facts

Begin Global:
Rule Multi:
    premise()
    ---------
    each(each(a(), b()), c())
End Global
"#;
        let program = parse_and_compile(input);
        let clauses: Vec<_> = program.global_rules.iter()
            .filter(|c| c.name == "Multi" || c.name.starts_with("Multi$"))
            .collect();
        assert_eq!(clauses.len(), 3);
    }
}
