use std::collections::HashMap;

use crate::ast::{Module, Rel, Rule, Stage, Term, TermContents};
use crate::ir::{
    Clause, Fact, Program, Prop, PropId, RelId, RelInfo, RelKind, SymbolId, Term as IRTerm,
    TermId, Var, VarId,
};

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

pub struct Compiler {
    program: Program,
    rel_map: HashMap<String, RelId>,
    var_map: HashMap<String, VarId>,
    fresh_var_counter: u32,
}

impl Compiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            program: Program::default(),
            rel_map: HashMap::new(),
            var_map: HashMap::new(),
            fresh_var_counter: 0,
        };
        compiler.register_builtin_relations();
        compiler
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

    fn get_or_create_var(&mut self, name: &str) -> VarId {
        if let Some(&id) = self.var_map.get(name) {
            return id;
        }
        let var = Var {
            name: name.to_string(),
        };
        let id = self.program.vars.alloc(var);
        self.var_map.insert(name.to_string(), id);
        id
    }

    fn fresh_var(&mut self) -> VarId {
        let name = format!("_G{}", self.fresh_var_counter);
        self.fresh_var_counter += 1;
        let var = Var { name: name.clone() };
        let id = self.program.vars.alloc(var);
        self.var_map.insert(name, id);
        id
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

    fn clear_rule_scope(&mut self) {
        self.var_map.clear();
        self.fresh_var_counter = 0;
    }

    fn lower_simple_term(&mut self, term: &Term) -> TermId {
        match &term.contents {
            TermContents::Var { name } => {
                let var_id = self.get_or_create_var(name);
                self.alloc_term(IRTerm::Var(var_id))
            }
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

    fn lower_term_arg(&mut self, term: &Term, constraints: &mut Vec<PropId>) -> TermId {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                if self.is_smt_relation(rel_name) {
                    let fresh_var_id = self.fresh_var();
                    let fresh_term_id = self.alloc_term(IRTerm::Var(fresh_var_id));

                    let mut lowered_args: Vec<TermId> = args
                        .iter()
                        .map(|a| self.lower_term_arg(a, constraints))
                        .collect();
                    lowered_args.push(fresh_term_id);

                    let arity = lowered_args.len();
                    let rel_id = self.get_or_create_rel(rel_name, arity, self.smt_kind(rel_name));
                    let prop = Prop::App {
                        rel: rel_id,
                        args: lowered_args,
                    };
                    constraints.push(self.alloc_prop(prop));

                    fresh_term_id
                } else {
                    let fresh_var_id = self.fresh_var();
                    let fresh_term_id = self.alloc_term(IRTerm::Var(fresh_var_id));

                    let lowered_args: Vec<TermId> = args
                        .iter()
                        .map(|a| self.lower_term_arg(a, constraints))
                        .collect();

                    let arity = lowered_args.len();
                    let rel_id = self.get_or_create_rel(rel_name, arity, RelKind::User);

                    let eq_prop = Prop::Eq(fresh_term_id, fresh_term_id);
                    constraints.push(self.alloc_prop(eq_prop));

                    let prop = Prop::App {
                        rel: rel_id,
                        args: lowered_args,
                    };
                    constraints.push(self.alloc_prop(prop));

                    fresh_term_id
                }
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

    fn lower_term_to_prop(&mut self, term: &Term) -> PropId {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                let mut constraints: Vec<PropId> = Vec::new();
                let lowered_args: Vec<TermId> = args
                    .iter()
                    .map(|a| self.lower_term_arg(a, &mut constraints))
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
                let app_prop_id = self.alloc_prop(app_prop);

                self.conjoin_all(constraints, app_prop_id)
            }
            _ => {
                self.alloc_prop(Prop::True)
            }
        }
    }

    fn conjoin_all(&mut self, props: Vec<PropId>, base: PropId) -> PropId {
        props.into_iter().fold(base, |acc, p| {
            let and_prop = Prop::And(p, acc);
            self.alloc_prop(and_prop)
        })
    }

    fn lower_fact(&mut self, term: &Term) -> Option<Fact> {
        match &term.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                self.clear_rule_scope();

                let mut constraints: Vec<PropId> = Vec::new();
                let lowered_args: Vec<TermId> = args
                    .iter()
                    .map(|a| self.lower_term_arg(a, &mut constraints))
                    .collect();

                if !constraints.is_empty() {
                    return None;
                }

                let arity = lowered_args.len();
                let rel_id = self.get_or_create_rel(rel_name, arity, RelKind::User);

                Some(Fact {
                    rel: rel_id,
                    args: lowered_args,
                })
            }
            _ => None,
        }
    }

    fn lower_rule(&mut self, rule: &Rule) -> Clause {
        self.clear_rule_scope();

        let body = self.lower_term_to_prop(&rule.premise);

        let (head_rel, head_args) = match &rule.conclusion.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } | Rel::UserRel { name } => name.as_str(),
                };

                let mut constraints: Vec<PropId> = Vec::new();
                let lowered_args: Vec<TermId> = args
                    .iter()
                    .map(|a| self.lower_term_arg(a, &mut constraints))
                    .collect();

                let arity = lowered_args.len();
                let rel_id = self.get_or_create_rel(rel_name, arity, RelKind::User);

                (rel_id, lowered_args)
            }
            _ => {
                let dummy_rel = self.get_or_create_rel("_true", 0, RelKind::User);
                (dummy_rel, Vec::new())
            }
        };

        Clause {
            name: rule.name.clone(),
            head_rel,
            head_args,
            body,
        }
    }

    fn lower_stage(&mut self, stage: &Stage) -> crate::ir::Stage {
        let rules = stage.rules.iter().map(|r| self.lower_rule(r)).collect();
        crate::ir::Stage {
            name: stage.name.clone(),
            rules,
        }
    }

    pub fn compile(mut self, module: &Module) -> Program {
        for fact_term in &module.facts {
            if let Some(fact) = self.lower_fact(fact_term) {
                self.program.facts.push(fact);
            }
        }

        for rule in &module.global_stage.rules {
            let clause = self.lower_rule(rule);
            self.program.global_rules.push(clause);
        }

        for stage in &module.stages {
            let ir_stage = self.lower_stage(stage);
            self.program.stages.push(ir_stage);
        }

        self.program
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compile(module: &Module) -> Program {
    Compiler::new().compile(module)
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
        let fact = &program.facts[0];
        assert_eq!(program.rels.get(fact.rel).name, "position");
        assert_eq!(fact.args.len(), 3);
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
        assert_eq!(program.global_rules.len(), 1);
        let clause = &program.global_rules[0];
        assert_eq!(clause.name, "MoveRight");
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
        assert_eq!(program.global_rules.len(), 1);

        let clause = &program.global_rules[0];
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
}
