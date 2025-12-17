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

        let first = frontend.query_start("num(X)").unwrap();
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
}
