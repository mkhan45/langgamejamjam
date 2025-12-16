use nom::Finish;
use z3::SatResult;

use crate::ast::parser::parse_module;
use crate::ast::Term;
use crate::solver::GameState;

pub struct GameSession {
    module_source: String,
    state: GameState,
    stage_names: Vec<String>,
    current_facts: Vec<Term>,
}

impl GameSession {
    pub fn new(source: &str) -> Result<GameSession, ()> {
        let mut session = GameSession {
            module_source: source.to_string(),
            state: GameState::new(),
            stage_names: Vec::new(),
            current_facts: Vec::new(),
        };
        session.reload()?;
        Ok(session)
    }

    pub fn reload(&mut self) -> Result<(), ()> {
        let span = nom_locate::LocatedSpan::new(self.module_source.as_str());
        let (_, module) = parse_module(span)
            .finish()
            .map_err(|e| ())?;

        self.state = GameState::new();
        self.stage_names.clear();
        self.current_facts.clear();

        for fact in module.facts {
            self.state.assert_fact(&fact);
            self.current_facts.push(fact);
        }

        self.state.load_stage(&module.global_stage);

        for stage in &module.stages {
            self.stage_names.push(stage.name.to_string());
        }

        Ok(())
    }

    pub fn stage_count(&self) -> usize {
        self.stage_names.len()
    }

    pub fn get_stage_name(&self, index: usize) -> Option<String> {
        self.stage_names.get(index).cloned()
    }

    pub fn check(&self) -> String {
        match self.state.check() {
            SatResult::Sat => "sat".to_string(),
            SatResult::Unsat => "unsat".to_string(),
            SatResult::Unknown => "unknown".to_string(),
        }
    }

    pub fn fact_count(&self) -> usize {
        self.current_facts.len()
    }

    pub fn get_fact(&self, index: usize) -> Option<String> {
        self.current_facts.get(index).map(|t| t.to_string())
    }
}
