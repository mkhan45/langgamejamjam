#[cfg(test)]
mod tests {
    use crate::frontend::Frontend;

    fn query_succeeds(frontend: &mut Frontend, query: &str) -> bool {
        frontend.query_batch(query, 10).unwrap_or_default().iter().any(|r| r != "no")
    }

    fn query_fails(frontend: &mut Frontend, query: &str) -> bool {
        !query_succeeds(frontend, query)
    }

    #[test]
    fn test_position_query() {
        let mut frontend = Frontend::new();
        frontend.load(
            "Begin Facts:\n    position(player, 0, 0)\nEnd Facts\n\nBegin Global:\nEnd Global\n",
        )
        .unwrap();

        assert!(query_succeeds(&mut frontend, "position(player, X, Y)"));
    }

    #[test]
    fn test_eq_fact_constrains_query() {
        let mut frontend = Frontend::new();
        frontend.load("Begin Facts:\n    eq(X, 1)\nEnd Facts\n\nBegin Global:\nEnd Global\n").unwrap();

        assert!(query_fails(&mut frontend, "eq(X, 2)"));
    }

    #[test]
    fn test_eq_fact_allows_compatible_query() {
        let mut frontend = Frontend::new();
        frontend.load("Begin Facts:\n    eq(X, 1)\nEnd Facts\n\nBegin Global:\nEnd Global\n").unwrap();

        assert!(query_succeeds(&mut frontend, "eq(X, Y)"));
    }

    #[test]
    fn test_eq_with_cons_term() {
        let mut frontend = Frontend::new();
        frontend.load("Begin Facts:\n    eq(X, cons(A, B))\nEnd Facts\n\nBegin Global:\nEnd Global\n")
            .unwrap();

        assert!(query_succeeds(&mut frontend, "eq(X, Y)"));
    }

    #[test]
    fn test_eq_fact_with_rule_present() {
        let mut frontend = Frontend::new();
        frontend.load(
            "Begin Facts:\n    eq(L, pair(1, 2))\nEnd Facts\n\nBegin Global:\n    Rule Test:\n    eq(A, B)\n    --------\n    someThing(B, A)\nEnd Global\n"
        ).unwrap();

        let result = frontend.query_batch("eq(A, L)", 10).unwrap();
        assert!(!result.is_empty());
        assert!(result[0].contains("pair"));
    }

    #[test]
    fn test_incremental_query() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    true()
End Facts

Begin Global:
    Rule Base:
    true()
    ------
    num(1)

    Rule Step:
    num(X)
    ------
    num(s(X))
End Global
"#).unwrap();

        frontend.max_steps = 1000;

        let first = frontend.query_start_global("num(X)").unwrap();
        assert!(first.is_some(), "Should find first solution");
        eprintln!("First: {:?}", first);

        let second = frontend.query_next();
        assert!(second.is_some(), "Should find second solution");
        eprintln!("Second: {:?}", second);

        let third = frontend.query_next();
        assert!(third.is_some(), "Should find third solution");
        eprintln!("Third: {:?}", third);

        frontend.query_stop();
        assert!(!frontend.has_more_solutions(), "No more after stop");
    }

    #[test]
    fn test_draw_directive_unconditional() {
        let mut frontend = Frontend::new();
        frontend
            .load(
                r#"Begin Facts:
End Facts

Begin Global:
End Global

Begin Stage Draw:
    Draw
        rect(1.0, 2.0, 3.0, 4.0)
        rect(5.0, 6.0, 7.0, 8.0)
End Stage Draw
"#,
            )
            .unwrap();

        let draws = frontend.collect_draws(0).unwrap();
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].name, "rect");
        assert_eq!(draws[0].args, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(draws[1].name, "rect");
        assert_eq!(draws[1].args, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_draw_directive_with_condition_true() {
        let mut frontend = Frontend::new();
        frontend
            .load(
                r#"Begin Facts:
    StateVar Dead
    Dead = no
End Facts

Begin Global:
End Global

Begin Stage Draw:
    With
        Dead = no
    Draw
        rect(1.0, 2.0, 3.0, 4.0)
End Stage Draw
"#,
            )
            .unwrap();

        let draws = frontend.collect_draws(0).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].name, "rect");
        assert_eq!(draws[0].args, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_draw_directive_with_condition_false() {
        let mut frontend = Frontend::new();
        frontend
            .load(
                r#"Begin Facts:
    StateVar Dead
    Dead = yes
End Facts

Begin Global:
End Global

Begin Stage Draw:
    With
        Dead = no
    Draw
        rect(1.0, 2.0, 3.0, 4.0)
End Stage Draw
"#,
            )
            .unwrap();

        let draws = frontend.collect_draws(0).unwrap();
        assert_eq!(draws.len(), 0);
    }

    #[test]
    fn test_draw_directive_with_state_var_interpolation() {
        let mut frontend = Frontend::new();
        frontend
            .load(
                r#"Begin Facts:
    StateVar PlayerY
    PlayerY = 5.0
End Facts

Begin Global:
End Global

Begin Stage Draw:
    Draw
        rect(1.0, PlayerY, 1.0, 1.0)
End Stage Draw
"#,
            )
            .unwrap();

        let draws = frontend.collect_draws(0).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].args, vec![1.0, 5.0, 1.0, 1.0]);
    }

    #[test]
    fn test_draw_directive_multiple_blocks() {
        let mut frontend = Frontend::new();
        frontend
            .load(
                r#"Begin Facts:
    StateVar Dead
    Dead = no
End Facts

Begin Global:
End Global

Begin Stage Draw:
    With
        Dead = no
    Draw
        rect(1.0, 0.0, 1.0, 1.0)

    With
        Dead = yes
    Draw
        rect(40.0, 40.0, 1.0, 1.0)
End Stage Draw
"#,
            )
            .unwrap();

        let draws = frontend.collect_draws(0).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].args, vec![1.0, 0.0, 1.0, 1.0]);
    }
}
