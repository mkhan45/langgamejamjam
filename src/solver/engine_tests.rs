use super::*;
use crate::ast::compile::compile;
use crate::ast::parser;
use nom::Finish;

use crate::solver::ir;

const ALL_STRATEGIES: [SearchStrategy; 2] = [SearchStrategy::BFS, SearchStrategy::DFS];

fn parse_and_compile(input: &str) -> Program {
    let result = parser::parse_module(input.into()).finish();
    let (_, module) = result.expect("parse failed");
    compile(&module)
}

fn for_each_strategy(test_fn: impl Fn(SearchStrategy)) {
    for strategy in ALL_STRATEGIES {
        test_fn(strategy);
    }
}

#[test]
fn test_fact_query() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
    position(player, 0, 0)
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let position_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "position")
            .map(|(id, _)| id)
            .unwrap();

        let player_sym = program.symbols.intern("player".to_string());
        let player_term = program.terms.alloc(Term::Atom(player_sym));
        let x_var = program.vars.alloc(ir::Var {
            name: "X".to_string(),
        });
        let y_var = program.vars.alloc(ir::Var {
            name: "Y".to_string(),
        });
        let x_term = program.terms.alloc(Term::Var(x_var));
        let y_term = program.terms.alloc(Term::Var(y_var));

        let query_prop = program.props.alloc(Prop::App {
            rel: position_rel,
            args: vec![player_term, x_term, y_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let solution = &solution_set.solutions()[0];

        let x_val = solution.subst.walk(x_term, &solver.program.terms);
        let y_val = solution.subst.walk(y_term, &solver.program.terms);

        match solver.program.terms.get(x_val) {
            Term::Int(0) => {}
            other => panic!("Expected Int(0), got {:?} (strategy: {:?})", other, strategy),
        }
        match solver.program.terms.get(y_val) {
            Term::Int(0) => {}
            other => panic!("Expected Int(0), got {:?} (strategy: {:?})", other, strategy),
        }
    });
}

#[test]
fn test_rule_backchain() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
    base(1)
End Facts

Begin Global:
Rule Derive:
    base(X)
    -------
    derived(X)
End Global
"#;
        let mut program = parse_and_compile(input);

        let derived_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "derived")
            .map(|(id, _)| id)
            .unwrap();

        let var = program.vars.alloc(ir::Var {
            name: "Q".to_string(),
        });
        let var_term = program.terms.alloc(Term::Var(var));

        let query_prop = program.props.alloc(Prop::App {
            rel: derived_rel,
            args: vec![var_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let solution = &solution_set.solutions()[0];

        let result = solution.subst.walk(var_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Int(1) => {}
            other => panic!("Expected Int(1), got {:?} (strategy: {:?})", other, strategy),
        }
    });
}

#[test]
fn test_multiple_facts() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
    item(sword)
    item(shield)
    item(potion)
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let item_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "item")
            .map(|(id, _)| id)
            .unwrap();

        let var = program.vars.alloc(ir::Var {
            name: "I".to_string(),
        });
        let var_term = program.terms.alloc(Term::Var(var));

        let query_prop = program.props.alloc(Prop::App {
            rel: item_rel,
            args: vec![var_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 3, "strategy: {:?}", strategy);
    });
}

#[test]
fn test_simple_graph() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
    edge(1, 2)
    edge(2, 3)
End Facts

Begin Global:
    Rule ConnT:
    and(edge(A, B), edge(B, C))
    ----------------------------
    connected(A, C)

    Rule ConnE:
    edge(A, B)
    ----------
    connected(A, B)
End Global
"#;
        let mut program = parse_and_compile(input);

        let connected_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "connected")
            .map(|(id, _)| id)
            .unwrap();

        let query_prop = program.props.alloc(Prop::App {
            rel: connected_rel,
            args: vec![
                program.terms.alloc(Term::Int(1)),
                program.terms.alloc(Term::Int(3)),
            ],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
    });
}

#[test]
fn test_smt_constraint() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
    value(5)
    true()
End Facts

Begin Global:
Rule AddOne:
    and(value(X), int_add(X, 1, Y))
    -------------------------------
    next(Y)
End Global
"#;
        let mut program = parse_and_compile(input);

        let next_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "next")
            .map(|(id, _)| id)
            .unwrap();

        let var = program.vars.alloc(ir::Var {
            name: "R".to_string(),
        });
        let var_term = program.terms.alloc(Term::Var(var));

        let query_prop = program.props.alloc(Prop::App {
            rel: next_rel,
            args: vec![var_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        assert!(
            solution_set.solutions()[0].constraints.is_empty(),
            "constraints should be solved, strategy: {:?}",
            strategy
        );

        let result = solution_set.solutions()[0].subst.walk(var_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Int(6) => {}
            other => panic!(
                "Expected next(6) for int_add(5, 1, Y), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_z3_int_add_forward() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let int_add_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "int_add")
            .map(|(id, _)| id)
            .unwrap();

        let var_b = program.vars.alloc(ir::Var {
            name: "B".to_string(),
        });
        let var_b_term = program.terms.alloc(Term::Var(var_b));
        let one_term = program.terms.alloc(Term::Int(1));

        // int_add(1, 1, B) should give B = 2
        let query_prop = program.props.alloc(Prop::App {
            rel: int_add_rel,
            args: vec![one_term, one_term, var_b_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let result = solution_set.solutions()[0].subst.walk(var_b_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Int(2) => {}
            other => panic!(
                "Expected B=2 for int_add(1, 1, B), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_z3_int_add_backward() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let int_add_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "int_add")
            .map(|(id, _)| id)
            .unwrap();

        let var_b = program.vars.alloc(ir::Var {
            name: "B".to_string(),
        });
        let var_b_term = program.terms.alloc(Term::Var(var_b));
        let two_term = program.terms.alloc(Term::Int(2));
        let five_term = program.terms.alloc(Term::Int(5));

        // int_add(2, B, 5) should give B = 3
        let query_prop = program.props.alloc(Prop::App {
            rel: int_add_rel,
            args: vec![two_term, var_b_term, five_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let result = solution_set.solutions()[0].subst.walk(var_b_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Int(3) => {}
            other => panic!(
                "Expected B=3 for int_add(2, B, 5), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_eq_conjunction_fails_on_conflict() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let var = program.vars.alloc(ir::Var {
            name: "X".to_string(),
        });
        let var_term = program.terms.alloc(Term::Var(var));
        let one_term = program.terms.alloc(Term::Int(1));
        let two_term = program.terms.alloc(Term::Int(2));

        let eq1 = program.props.alloc(Prop::Eq(var_term, one_term));
        let eq2 = program.props.alloc(Prop::Eq(var_term, two_term));
        let query_prop = program.props.alloc(Prop::And(eq1, eq2));

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(
            solution_set.solutions().len(),
            0,
            "and(eq(X, 1), eq(X, 2)) should fail (strategy: {:?})",
            strategy
        );
    });
}

#[test]
fn test_eq_conjunction_succeeds_when_compatible() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let var_x = program.vars.alloc(ir::Var {
            name: "X".to_string(),
        });
        let var_y = program.vars.alloc(ir::Var {
            name: "Y".to_string(),
        });
        let var_x_term = program.terms.alloc(Term::Var(var_x));
        let var_y_term = program.terms.alloc(Term::Var(var_y));
        let one_term = program.terms.alloc(Term::Int(1));

        let eq1 = program.props.alloc(Prop::Eq(var_x_term, one_term));
        let eq2 = program.props.alloc(Prop::Eq(var_x_term, var_y_term));
        let query_prop = program.props.alloc(Prop::And(eq1, eq2));

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(
            solution_set.solutions().len(),
            1,
            "and(eq(X, 1), eq(X, Y)) should succeed with Y=1 (strategy: {:?})",
            strategy
        );

        let y_val = solution_set.solutions()[0]
            .subst
            .walk(var_y_term, &solver.program.terms);
        match solver.program.terms.get(y_val) {
            Term::Int(1) => {}
            other => panic!("Expected Y=1, got {:?} (strategy: {:?})", other, strategy),
        }
    });
}

#[test]
fn test_z3_real_add_forward() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let real_add_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "real_add")
            .map(|(id, _)| id)
            .unwrap();

        let var_c = program.vars.alloc(ir::Var {
            name: "C".to_string(),
        });
        let var_c_term = program.terms.alloc(Term::Var(var_c));
        let one_term = program.terms.alloc(Term::Float(1.5));
        let two_term = program.terms.alloc(Term::Float(2.5));

        // real_add(1.5, 2.5, C) should give C = 4.0
        let query_prop = program.props.alloc(Prop::App {
            rel: real_add_rel,
            args: vec![one_term, two_term, var_c_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let result = solution_set.solutions()[0].subst.walk(var_c_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Float(f) if (*f - 4.0).abs() < 0.0001 => {}
            other => panic!(
                "Expected C=4.0 for real_add(1.5, 2.5, C), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_z3_real_add_backward() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let real_add_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "real_add")
            .map(|(id, _)| id)
            .unwrap();

        let var_b = program.vars.alloc(ir::Var {
            name: "B".to_string(),
        });
        let var_b_term = program.terms.alloc(Term::Var(var_b));
        let two_term = program.terms.alloc(Term::Float(2.0));
        let five_term = program.terms.alloc(Term::Float(5.0));

        // real_add(2.0, B, 5.0) should give B = 3.0
        let query_prop = program.props.alloc(Prop::App {
            rel: real_add_rel,
            args: vec![two_term, var_b_term, five_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let result = solution_set.solutions()[0].subst.walk(var_b_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Float(f) if (*f - 3.0).abs() < 0.0001 => {}
            other => panic!(
                "Expected B=3.0 for real_add(2.0, B, 5.0), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_z3_real_div() {
    for_each_strategy(|strategy| {
        let input = r#"Begin Facts:
End Facts

Begin Global:
End Global
"#;
        let mut program = parse_and_compile(input);

        let real_div_rel = program
            .rels
            .iter()
            .find(|(_, r)| r.name == "real_div")
            .map(|(id, _)| id)
            .unwrap();

        let var_c = program.vars.alloc(ir::Var {
            name: "C".to_string(),
        });
        let var_c_term = program.terms.alloc(Term::Var(var_c));
        let ten_term = program.terms.alloc(Term::Float(10.0));
        let four_term = program.terms.alloc(Term::Float(4.0));

        // real_div(10.0, 4.0, C) should give C = 2.5
        let query_prop = program.props.alloc(Prop::App {
            rel: real_div_rel,
            args: vec![ten_term, four_term, var_c_term],
        });

        let mut solver = Solver::new(&mut program);
        let solution_set = solver.collect_solutions(query_prop, strategy, usize::MAX, 100_000);

        assert_eq!(solution_set.solutions().len(), 1, "strategy: {:?}", strategy);
        let result = solution_set.solutions()[0].subst.walk(var_c_term, &solver.program.terms);
        match solver.program.terms.get(result) {
            Term::Float(f) if (*f - 2.5).abs() < 0.0001 => {}
            other => panic!(
                "Expected C=2.5 for real_div(10.0, 4.0, C), got {:?} (strategy: {:?})",
                other, strategy
            ),
        }
    });
}

#[test]
fn test_eager_constraint_pruning() {
    let input = std::fs::read_to_string("sample/inventory.l")
        .expect("Failed to read sample/inventory.l");
    let mut program = parse_and_compile(&input);

    let cartcost_rel = program
        .rels
        .iter()
        .find(|(_, r)| r.name == "cartCost")
        .map(|(id, _)| id)
        .unwrap();

    let var_c = program.vars.alloc(ir::Var {
        name: "C".to_string(),
    });
    let var_c_term = program.terms.alloc(Term::Var(var_c));
    let var_a = program.vars.alloc(ir::Var {
        name: "A".to_string(),
    });
    let var_a_term = program.terms.alloc(Term::Var(var_a));
    let zero_term = program.terms.alloc(Term::Int(0));

    // cartCost(C, A, 0) - with MaxSize=0, only empty cart should match
    let query_prop = program.props.alloc(Prop::App {
        rel: cartcost_rel,
        args: vec![var_c_term, var_a_term, zero_term],
    });

    let mut solver = Solver::new(&mut program);
    let solution_set = solver.collect_solutions(query_prop, SearchStrategy::default(), 5, 1000);

    assert_eq!(solution_set.solutions().len(), 1, "Should find exactly one solution (empty cart)");
    
    let cart_result = solution_set.solutions()[0].subst.walk(var_c_term, &solver.program.terms);
    match solver.program.terms.get(cart_result) {
        Term::Atom(s) => {
            let name = solver.program.symbols.get(*s);
            assert_eq!(name, "nil", "Cart should be nil");
        }
        other => panic!("Expected nil cart, got {:?}", other),
    }

    let cost_result = solution_set.solutions()[0].subst.walk(var_a_term, &solver.program.terms);
    match solver.program.terms.get(cost_result) {
        Term::Int(0) => {}
        other => panic!("Expected cost 0, got {:?}", other),
    }
}

#[test]
fn test_cartcost_with_items() {
    let input = std::fs::read_to_string("sample/inventory.l")
        .expect("Failed to read sample/inventory.l");
    let mut program = parse_and_compile(&input);

    let cartcost_rel = program
        .rels
        .iter()
        .find(|(_, r)| r.name == "cartCost")
        .map(|(id, _)| id)
        .unwrap();

    let var_c = program.vars.alloc(ir::Var {
        name: "C".to_string(),
    });
    let var_c_term = program.terms.alloc(Term::Var(var_c));
    let var_t = program.vars.alloc(ir::Var {
        name: "T".to_string(),
    });
    let var_t_term = program.terms.alloc(Term::Var(var_t));
    let two_term = program.terms.alloc(Term::Int(2));

    // cartCost(C, T, 2) - find carts with up to 2 items
    let query_prop = program.props.alloc(Prop::App {
        rel: cartcost_rel,
        args: vec![var_c_term, var_t_term, two_term],
    });

    let mut solver = Solver::new(&mut program);
    let solution_set = solver.collect_solutions(query_prop, SearchStrategy::default(), 10, 100_000);

    assert!(solution_set.solutions().len() >= 5, "Should find multiple cart combinations, got {}", solution_set.solutions().len());
}

#[test]
fn test_cartcost_specific_total() {
    let input = std::fs::read_to_string("sample/inventory.l")
        .expect("Failed to read sample/inventory.l");
    let mut program = parse_and_compile(&input);

    let cartcost_rel = program
        .rels
        .iter()
        .find(|(_, r)| r.name == "cartCost")
        .map(|(id, _)| id)
        .unwrap();

    let var_c = program.vars.alloc(ir::Var {
        name: "C".to_string(),
    });
    let var_c_term = program.terms.alloc(Term::Var(var_c));
    let ten_term = program.terms.alloc(Term::Int(10));
    let one_term = program.terms.alloc(Term::Int(1));

    // cartCost(C, 10, 1) - find single-item carts costing exactly 10
    let query_prop = program.props.alloc(Prop::App {
        rel: cartcost_rel,
        args: vec![var_c_term, ten_term, one_term],
    });

    let mut solver = Solver::new(&mut program);
    let solution_set = solver.collect_solutions(query_prop, SearchStrategy::default(), 5, 1000);

    assert_eq!(solution_set.solutions().len(), 1, "Should find exactly one cart costing 10 with max 1 item");
}

#[test]
fn test_cartcost_unbound_maxsize() {
    // Regression test: cartCost(A, 25, B) should work even when B (MaxSize) is unbound
    let input = std::fs::read_to_string("sample/inventory.l")
        .expect("Failed to read sample/inventory.l");
    let mut program = parse_and_compile(&input);

    let cartcost_rel = program
        .rels
        .iter()
        .find(|(_, r)| r.name == "cartCost")
        .map(|(id, _)| id)
        .unwrap();

    let var_a = program.vars.alloc(ir::Var {
        name: "A".to_string(),
    });
    let var_a_term = program.terms.alloc(Term::Var(var_a));
    let var_b = program.vars.alloc(ir::Var {
        name: "B".to_string(),
    });
    let var_b_term = program.terms.alloc(Term::Var(var_b));
    let twenty_five_term = program.terms.alloc(Term::Int(25));

    // cartCost(A, 25, B) - find carts costing 25 with any max size
    let query_prop = program.props.alloc(Prop::App {
        rel: cartcost_rel,
        args: vec![var_a_term, twenty_five_term, var_b_term],
    });

    let mut solver = Solver::new(&mut program);
    let solution_set = solver.collect_solutions(query_prop, SearchStrategy::default(), 5, 5000);

    assert!(!solution_set.solutions().is_empty(), "Should find carts costing 25 with unbound MaxSize");
}
