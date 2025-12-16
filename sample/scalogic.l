Begin Facts:
    true()
    eq(L, cons(1, cons(2, cons(3, cons(4, nil)))))
End Facts

Begin Global:
    Rule SameLenBase:
    true()
    -----------------
    sameLength(nil, nil)

    Rule SameLenRec:
    sameLength(Xs, Ys)
    ------------------
    sameLength(cons(Xh, Xs), cons(Yh, Ys))

    Rule ContainsHead:
    true()
    ------------------
    contains(cons(X, Xs), X)

    Rule ContainsTail:
    contains(Xs, X)
    ---------------
    contains(cons(Y, Xs), X)

    Rule ContainsAllBase:
    true()
    ---------------------
    containsAll(nil, L2)

    Rule ContainsAllRec:
    and(contains(L2, X), containsAll(Xs, L2))
    -----------------------------------------
    containsAll(cons(X, Xs), L2)

    Rule AppendBase:
    true()
    ----------------
    append(nil, X, cons(X, nil))

    Rule AppendRec:
    append(Xs, X, Ys)
    -----------------
    append(cons(Xh, Xs), X, cons(Xh, Ys))

    Rule ReverseBase:
    true()
    -----------------
    reverseOf(nil, nil)

    Rule ReverseRec:
    and(reverseOf(Xs, Yp), append(Yp, Xh, Y))
    -----------------------------------------
    reverseOf(cons(Xh, Xs), Y)

    Rule Palindrome:
    reverseOf(X, X)
    ---------------
    palindrome(X)
End Global
