#[cfg(test)]
mod tests {
    use crate::frontend::Frontend;

    #[test]
    fn test_state_var_declaration() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar Health
    eq(Health, 10)
End Facts

Begin Global:
End Global
"#).unwrap();

        assert_eq!(frontend.program.state_vars.len(), 1);
        assert_eq!(frontend.program.state_vars[0], "Health");
        
        let health = frontend.get_state_var("Health");
        assert!(health.is_some(), "Health should be defined");
        assert_eq!(health.unwrap(), "10");
    }

    #[test]
    fn test_multiple_state_vars() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar Health
    StateVar Score
    StateVar Lives
    eq(Health, 100)
    eq(Score, 0)
    eq(Lives, 3)
End Facts

Begin Global:
End Global
"#).unwrap();

        assert_eq!(frontend.program.state_vars.len(), 3);
        
        let vars = frontend.state_vars();
        assert_eq!(vars.len(), 3);
        
        assert_eq!(frontend.get_state_var("Health").unwrap(), "100");
        assert_eq!(frontend.get_state_var("Score").unwrap(), "0");
        assert_eq!(frontend.get_state_var("Lives").unwrap(), "3");
    }

    #[test]
    fn test_state_constraint_basic() {
        let input = std::fs::read_to_string("sample/state_basic.l")
            .expect("Failed to read sample/state_basic.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        assert_eq!(frontend.get_state_var("Health").unwrap(), "10");

        frontend.run_stage(0).expect("Stage should succeed");

        assert_eq!(frontend.get_state_var("Health").unwrap(), "9");
    }

    #[test]
    fn test_state_constraint_multiple_updates() {
        let input = std::fs::read_to_string("sample/state_multiple.l")
            .expect("Failed to read sample/state_multiple.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        assert_eq!(frontend.get_state_var("Health").unwrap(), "100");
        assert_eq!(frontend.get_state_var("Score").unwrap(), "0");

        frontend.run_stage(0).expect("Stage should succeed");

        assert_eq!(frontend.get_state_var("Health").unwrap(), "95");
        assert_eq!(frontend.get_state_var("Score").unwrap(), "10");
    }

    #[test]
    fn test_state_constraint_repeated_execution() {
        let input = std::fs::read_to_string("sample/state_basic.l")
            .expect("Failed to read sample/state_basic.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        assert_eq!(frontend.get_state_var("Health").unwrap(), "10");

        for expected in (7..=9).rev() {
            frontend.run_stage(0).expect("Stage should succeed");
            assert_eq!(
                frontend.get_state_var("Health").unwrap(), 
                expected.to_string(),
                "After decrement, Health should be {}", expected
            );
        }
    }

    #[test]
    fn test_state_constraint_ambiguous_error() {
        let input = std::fs::read_to_string("sample/state_ambiguous.l")
            .expect("Failed to read sample/state_ambiguous.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        let result = frontend.run_stage(0);
        assert!(result.is_err(), "Ambiguous state update should fail");
        
        let err = result.unwrap_err();
        assert!(err.contains("Ambiguous") || err.contains("multiple solutions"), 
            "Error should mention ambiguity: {}", err);
    }

    #[test]
    fn test_state_constraint_no_solution_error() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar X
    eq(X, 1)
End Facts

Begin Global:
End Global

Begin Stage Impossible:
Begin State Constraints:
    int_add(next(X), 0, 999)
    int_add(next(X), 0, 1)
End State Constraints
End Stage Impossible
"#).unwrap();

        let result = frontend.run_stage(0);
        assert!(result.is_err(), "Contradictory constraints should fail");
        
        let err = result.unwrap_err();
        assert!(err.contains("no solutions"), "Error should mention no solutions: {}", err);
    }

    #[test]
    fn test_run_stage_by_name() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar Counter
    eq(Counter, 0)
End Facts

Begin Global:
End Global

Begin Stage Increment:
Begin State Constraints:
    int_add(Counter, 1, next(Counter))
End State Constraints
End Stage Increment
"#).unwrap();

        assert_eq!(frontend.get_state_var("Counter").unwrap(), "0");

        frontend.run_stage_by_name("Increment").expect("Stage should succeed");

        assert_eq!(frontend.get_state_var("Counter").unwrap(), "1");
    }

    #[test]
    fn test_stage_without_constraints() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar X
    eq(X, 5)
End Facts

Begin Global:
End Global

Begin Stage NoConstraints:
Rule Dummy:
    eq(A, A)
    --------
    dummy(A)
End Stage NoConstraints
"#).unwrap();

        assert_eq!(frontend.get_state_var("X").unwrap(), "5");

        frontend.run_stage(0).expect("Empty constraint stage should succeed");

        assert_eq!(frontend.get_state_var("X").unwrap(), "5");
    }

    #[test]
    fn test_state_var_not_mentioned_preserved() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar A
    StateVar B
    eq(A, 10)
    eq(B, 20)
End Facts

Begin Global:
End Global

Begin Stage UpdateA:
Begin State Constraints:
    int_add(A, 1, next(A))
End State Constraints
End Stage UpdateA
"#).unwrap();

        assert_eq!(frontend.get_state_var("A").unwrap(), "10");
        assert_eq!(frontend.get_state_var("B").unwrap(), "20");

        frontend.run_stage(0).expect("Stage should succeed");

        assert_eq!(frontend.get_state_var("A").unwrap(), "11");
        assert_eq!(frontend.get_state_var("B").unwrap(), "20");
    }

    #[test]
    fn test_parser_state_constraints_section() {
        use crate::ast::parser;
        use nom::Finish;

        let input = r#"Begin Facts:
    StateVar X
    eq(X, 1)
End Facts

Begin Global:
End Global

Begin Stage TestStage:
Rule Dummy:
    eq(A, A)
    --------
    dummy(A)
Begin State Constraints:
    eq(X, 2)
End State Constraints
End Stage TestStage
"#;

        let result = parser::parse_module(input.into()).finish();
        assert!(result.is_ok(), "Parsing should succeed: {:?}", result);
        
        let (_, module) = result.unwrap();
        assert_eq!(module.state_vars.len(), 1);
        assert_eq!(module.stages.len(), 1);
        assert_eq!(module.stages[0].state_constraints.len(), 1);
    }

    #[test]
    fn test_runner_should_jump_rule() {
        let input = std::fs::read_to_string("sample/runner.l")
            .expect("Failed to read sample/runner.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        // Verify state variables are initialized correctly
        assert_eq!(frontend.get_state_var("RunnerY").unwrap(), "0");
        assert_eq!(frontend.get_state_var("RunnerVY").unwrap(), "0");
        assert_eq!(frontend.get_state_var("ObstacleX").unwrap(), "100");

        // shouldJump should fail because ObstacleX is 30, which is not in range [8, 18]
        match frontend.query_start_global("shouldJump()") {
            Ok(None) => {}, // Expected: no solution
            Ok(Some(sol)) => {
                eprintln!("DEBUG: shouldJump matched with solution: {}", sol);
                panic!("shouldJump() should not match when ObstacleX=30")
            },
            Err(e) => panic!("Query failed with error: {}", e),
        }
        
        // Test the compound condition directly - should also fail
        match frontend.query_start_global("and(real_eq(RunnerY, 0.0), and(real_ge(ObstacleX, 8.0), real_le(ObstacleX, 18.0)))") {
            Ok(None) => {}, // Expected: no solution
            Ok(Some(sol)) => {
                eprintln!("DEBUG: Compound query matched with solution: {}", sol);
                panic!("Compound condition should not match when ObstacleX=30")
            },
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_runner_should_jump_in_range() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar RunnerY
    StateVar RunnerVY
    StateVar ObstacleX
    eq(RunnerY, 0.0)
    eq(RunnerVY, 0.0)
    eq(ObstacleX, 10.0)
End Facts

Begin Global:
Rule ShouldJump:
    and(real_eq(RunnerY, 0.0), and(real_ge(ObstacleX, 8.0), real_le(ObstacleX, 18.0)))
    ---------------------------------------------------------------------------------
    shouldJump()
End Global
"#).unwrap();

        // Verify state variables
        assert_eq!(frontend.get_state_var("RunnerY").unwrap(), "0");
        assert_eq!(frontend.get_state_var("ObstacleX").unwrap(), "10");

        // shouldJump should succeed because ObstacleX is 10, which is in range [8, 18]
        match frontend.query_start_global("shouldJump()") {
            Ok(Some(_)) => {}, // Expected: solution found
            Ok(None) => panic!("shouldJump() should match when RunnerY=0 and ObstacleX=10"),
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_simple_real_constraint() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar X
    eq(X, 50.0)
End Facts

Begin Global:
End Global
"#).unwrap();

        // Direct constraint test: should fail because X is 50, not less than 18
        match frontend.query_start_global("real_le(X, 18.0)") {
            Ok(None) => {}, // Expected: no solution
            Ok(Some(sol)) => panic!("real_le(X, 18.0) should fail when X=50, but got: {}", sol),
            Err(e) => panic!("Query failed with error: {}", e),
        }

        // This should succeed: X is 50, which is >= 8
        match frontend.query_start_global("real_ge(X, 8.0)") {
            Ok(Some(_)) => {}, // Expected: solution found
            Ok(None) => panic!("real_ge(X, 8.0) should match when X=50"),
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_and_constraint() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar X
    eq(X, 50.0)
End Facts

Begin Global:
End Global
"#).unwrap();

        // This should fail: X is 50, so (X >= 8) AND (X <= 18) is false (second part fails)
        match frontend.query_start_global("and(real_ge(X, 8.0), real_le(X, 18.0))") {
            Ok(None) => {}, // Expected: no solution
            Ok(Some(sol)) => {
                eprintln!("DEBUG: and(real_ge(X, 8.0), real_le(X, 18.0)) matched when X=50: {}", sol);
                panic!("and constraint should fail when X=50")
            },
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_rule_with_state_var_constraint() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar X
    eq(X, 50.0)
End Facts

Begin Global:
Rule TestRule:
    and(real_ge(X, 8.0), real_le(X, 18.0))
    -----------------------------------------------
    test_pred()
End Global
"#).unwrap();

        // Query the rule: should fail because X is 50, which fails the second constraint
        match frontend.query_start_global("test_pred()") {
            Ok(None) => {}, // Expected: no solution
            Ok(Some(sol)) => {
                eprintln!("DEBUG: test_pred() matched when X=50: {}", sol);
                panic!("test_pred() should fail when X=50")
            },
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_query_after_stage_execution() {
        let input = std::fs::read_to_string("sample/state_basic.l")
            .expect("Failed to read sample/state_basic.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        // Query before stage - baseline
        match frontend.query_start_global("eq(1, 1)") {
            Ok(Some(_)) => {}, // Expected
            Ok(None) => panic!("eq(1, 1) should work before stage"),
            Err(e) => panic!("Query failed before stage: {}", e),
        }

        // Run the stage
        frontend.run_stage(0).expect("Stage should succeed");
        assert_eq!(frontend.get_state_var("Health").unwrap(), "9");

        // Query after stage should still work - simple query
        match frontend.query_start_global("eq(1, 1)") {
            Ok(Some(_)) => {}, // Expected: solution found
            Ok(None) => {
                panic!("eq(1, 1) should always succeed after running a stage")
            },
            Err(e) => panic!("Query failed with error: {}", e),
        }

        // Test with state variables
        match frontend.query_start_global("eq(Health, 9)") {
            Ok(Some(_)) => {}, // Expected: solution found (Health was just set to 9)
            Ok(None) => panic!("eq(Health, 9) should succeed after running stage"),
            Err(e) => panic!("Query failed with error: {}", e),
        }
    }

    #[test]
    fn test_query_after_runner_stage() {
        let input = std::fs::read_to_string("sample/runner.l")
            .expect("Failed to read sample/runner.l");
        
        let mut frontend = Frontend::new();
        frontend.load(&input).unwrap();

        // Query before stage
        match frontend.query_start_global("eq(1, 1)") {
            Ok(Some(_)) => {}, // Expected
            Ok(None) => panic!("eq(1, 1) should work before stage"),
            Err(e) => panic!("Query failed before stage: {}", e),
        }

        // Run Control stage (uses shouldJump rule)
        frontend.run_stage_by_name("Control").expect("Control stage should succeed");

        // Query after stage
        match frontend.query_start_global("eq(1, 1)") {
            Ok(Some(_)) => {}, // Expected
            Ok(None) => panic!("eq(1, 1) should work after Control stage"),
            Err(e) => panic!("Query failed after Control stage: {}", e),
        }
    }

    #[test]
    fn test_or_with_not_simple() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    eq(1, 1)
End Facts

Begin Global:
Rule AlwaysTrue:
    eq(1, 1)
    --------
    truePred()
End Global
"#).unwrap();

        // First verify truePred works
        eprintln!("TEST 1: Direct truePred query");
        match frontend.query_start_global("truePred()") {
            Ok(Some(sol)) => {
                eprintln!("  Result: success - {}", sol);
            },
            Ok(None) => {
                eprintln!("  Result: no solution");
                panic!("truePred() should succeed");
            },
            Err(e) => {
                eprintln!("  Result: error - {}", e);
                panic!("Query failed: {}", e);
            },
        }

        // Test: or(truePred(), not(truePred()))  
        // First branch should succeed, so or should succeed
        eprintln!("TEST 2: or(truePred(), not(truePred()))");
        match frontend.query_start_global("or(truePred(), not(truePred()))") {
            Ok(Some(sol)) => {
                eprintln!("  Result: success - {}", sol);
            },
            Ok(None) => {
                eprintln!("  Result: no solution");
                panic!("or(truePred(), not(truePred())) should always succeed");
            },
            Err(e) => {
                eprintln!("  Result: error - {}", e);
                panic!("Query failed: {}", e);
            },
        }
    }

    #[test]
    fn test_not_with_smt_constraints() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    StateVar ObstacleX
    eq(ObstacleX, 30.0)
End Facts

Begin Global:
Rule ShouldJump:
    real_ge(ObstacleX, 8.0)
    -------------------------
    shouldJump()
End Global
"#).unwrap();

        // Direct test: shouldJump should succeed (ObstacleX >= 8.0)
        eprintln!("TEST 1: Direct shouldJump query");
        match frontend.query_start_global("shouldJump()") {
            Ok(Some(sol)) => {
                eprintln!("  Result: success - {}", sol);
            },
            Ok(None) => {
                eprintln!("  Result: no solution");
                panic!("shouldJump() should succeed when ObstacleX=30");
            },
            Err(e) => {
                eprintln!("  Result: error - {}", e);
                panic!("Query failed: {}", e);
            },
        }

        // Test negation: not(shouldJump) should fail
        eprintln!("TEST 2: Negation not(shouldJump)");
        match frontend.query_start_global("not(shouldJump())") {
            Ok(None) => {
                eprintln!("  Result: no solution (expected - negation fails)");
            },
            Ok(Some(sol)) => {
                eprintln!("  Result: solution found - {}", sol);
                panic!("not(shouldJump()) should fail, but got: {}", sol);
            },
            Err(e) => {
                eprintln!("  Result: error - {}", e);
                panic!("Query failed: {}", e);
            },
        }

        // Test or with both branches - at least one must succeed
        eprintln!("TEST 3: Or with both branches");
        match frontend.query_start_global("or(shouldJump(), not(shouldJump()))") {
            Ok(Some(sol)) => {
                eprintln!("  Result: success - {}", sol);
            },
            Ok(None) => {
                eprintln!("  Result: no solution");
                panic!("or(shouldJump(), not(shouldJump())) should always succeed");
            },
            Err(e) => {
                eprintln!("  Result: error - {}", e);
                panic!("Query failed: {}", e);
            },
        }
    }
}
