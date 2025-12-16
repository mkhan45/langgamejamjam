#[cfg(test)]
mod tests {
    use crate::Frontend;

    #[test]
    fn test_samelength_concrete() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    true
End Facts

Begin Global:
    Rule SameLenBase:
    true
    -----------------
    sameLength(nil, nil)

    Rule SameLenRec:
    sameLength(Xs, Ys)
    ------------------
    sameLength(cons(Xh, Xs), cons(Yh, Ys))
End Global
"#).unwrap();

        let result = frontend.query("sameLength(nil, nil)").unwrap();
        eprintln!("nil, nil: {:?}", result);
        assert!(!result.is_empty(), "sameLength(nil, nil) should succeed");
        
        let result2 = frontend.query("sameLength(cons(a, nil), cons(b, nil))").unwrap();
        eprintln!("cons(a,nil), cons(b,nil): {:?}", result2);
        assert!(!result2.is_empty(), "sameLength(cons(a,nil), cons(b,nil)) should succeed");
        
        let result3 = frontend.query("sameLength(cons(a, nil), nil)").unwrap();
        eprintln!("cons(a,nil), nil: {:?}", result3);
        assert!(result3.is_empty(), "sameLength(cons(a,nil), nil) should fail");
    }

    #[test]
    fn test_samelength_open_query() {
        let mut frontend = Frontend::new();
        frontend.load(r#"Begin Facts:
    true
End Facts

Begin Global:
    Rule SameLenBase:
    true
    -----------------
    sameLength(nil, nil)

    Rule SameLenRec:
    sameLength(Xs, Ys)
    ------------------
    sameLength(cons(Xh, Xs), cons(Yh, Ys))
End Global
"#).unwrap();

        let result = frontend.query("sameLength(A, B)").unwrap();
        eprintln!("Open query results: {:?}", result);
        assert_eq!(result.len(), 10, "Default limit should be 10");
        assert!(result[0].contains("nil") && result[0].contains("A =") && result[0].contains("B ="));
        
        let result_5 = frontend.query_with_limit("sameLength(A, B)", 5).unwrap();
        assert_eq!(result_5.len(), 5, "Custom limit should be respected");
    }
}
