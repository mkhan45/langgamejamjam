pub mod compile;
pub mod parser;

#[cfg(test)]
mod parser_tests;

use std::fmt;

#[derive(Debug, Clone)]
pub struct Module {
    pub state_vars: Vec<String>,
    pub facts: Vec<Term>,
    pub global_stage: Stage,
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone)]
pub struct DrawDirective {
    pub condition: Option<Term>,
    pub draws: Vec<Term>,
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub name: String,
    pub rules: Vec<Rule>,
    pub state_constraints: Vec<Term>,
    pub draw_directives: Vec<DrawDirective>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub premise: Term,
    pub conclusion: Term,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub contents: TermContents,
}

#[derive(Debug, Clone)]
pub enum TermContents {
    App { rel: Rel, args: Vec<Term> },
    Atom { text: String },
    Var { name: String },
    Int { val: i32 },
    Float { val: f32 },
}

#[derive(Debug, Clone)]
pub enum Rel {
    SMTRel { name: String },
    UserRel { name: String },
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.contents {
            TermContents::App { rel, args } => {
                let rel_name = match rel {
                    Rel::SMTRel { name } => name,
                    Rel::UserRel { name } => name,
                };
                write!(f, "{}", rel_name)?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            TermContents::Atom { text } => write!(f, "{}", text),
            TermContents::Var { name } => write!(f, "{}", name),
            TermContents::Int { val } => write!(f, "{}", val),
            TermContents::Float { val } => write!(f, "{}", val),
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rule {}:", self.name)?;

        let premise_str = format!("{}", self.premise);
        let conclusion_str = format!("{}", self.conclusion);
        let max_len = premise_str.len().max(conclusion_str.len());
        let dashes = "-".repeat(max_len);

        writeln!(f, "    {}", premise_str)?;
        writeln!(f, "    {}", dashes)?;
        writeln!(f, "    {}", conclusion_str)
    }
}

impl fmt::Display for DrawDirective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(cond) = &self.condition {
            writeln!(f, "    With")?;
            writeln!(f, "        {}", cond)?;
        }
        writeln!(f, "    Draw")?;
        for draw in &self.draws {
            writeln!(f, "        {}", draw)?;
        }
        Ok(())
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Begin Stage {}:", self.name)?;
        for rule in &self.rules {
            write!(f, "{}", rule)?;
        }
        if !self.state_constraints.is_empty() {
            writeln!(f, "Begin State Constraints:")?;
            for constraint in &self.state_constraints {
                writeln!(f, "    {}", constraint)?;
            }
            writeln!(f, "End State Constraints")?;
        }
        for directive in &self.draw_directives {
            write!(f, "{}", directive)?;
        }
        writeln!(f, "End Stage {}", self.name)
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Begin Facts:")?;
        for state_var in &self.state_vars {
            writeln!(f, "    StateVar {}", state_var)?;
        }
        for fact in &self.facts {
            writeln!(f, "    {}", fact)?;
        }
        writeln!(f, "End Facts")?;
        writeln!(f)?;

        writeln!(f, "Begin Global:")?;
        for rule in &self.global_stage.rules {
            write!(f, "{}", rule)?;
        }
        writeln!(f, "End Global")?;

        for stage in &self.stages {
            writeln!(f)?;
            write!(f, "{}", stage)?;
        }
        Ok(())
    }
}
