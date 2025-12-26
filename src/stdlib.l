Begin Global:
    # List membership
    Rule ContainsHead:
    true()
    ------------------
    contains(cons(X, Xs), X)

    Rule ContainsTail:
    contains(Xs, X)
    ---------------
    contains(cons(Y, Xs), X)

    # List append
    Rule AppendNil:
    true()
    ----------------
    append(nil, L, L)

    Rule AppendCons:
    append(Xs, Ys, Zs)
    ------------------
    append(cons(X, Xs), Ys, cons(X, Zs))

    # List length
    Rule LengthNil:
    true()
    --------------
    length(nil, 0)

    Rule LengthCons:
    and(length(Xs, N), int_add(N, 1, N1))
    -------------------------------------
    length(cons(X, Xs), N1)

    # List head/tail
    Rule Head:
    true()
    ----------------
    head(cons(X, Xs), X)

    Rule Tail:
    true()
    -----------------
    tail(cons(X, Xs), Xs)

    # Reverse
    Rule ReverseBase:
    true()
    -----------------
    reverseOf(nil, nil)

    Rule ReverseRec:
    and(reverseOf(Xs, Yp), append(Yp, cons(Xh, nil), Y))
    ----------------------------------------------------
    reverseOf(cons(Xh, Xs), Y)

    # Nth element (0-indexed)
    Rule NthZero:
    true()
    ----------------
    nth(cons(X, Xs), 0, X)

    Rule NthSucc:
    and(int_sub(N, 1, N1), nth(Xs, N1, X))
    ---------------------------------------
    nth(cons(Y, Xs), N, X)

    # Min/Max for integers
    Rule MinLeft:
    int_le(X, Y)
    ------------
    min(X, Y, X)

    Rule MinRight:
    int_lt(Y, X)
    ------------
    min(X, Y, Y)

    Rule MaxLeft:
    int_ge(X, Y)
    ------------
    max(X, Y, X)

    Rule MaxRight:
    int_gt(Y, X)
    ------------
    max(X, Y, Y)

    # Range check
    Rule Between:
    and(int_ge(X, Low), int_le(X, High))
    ------------------------------------
    between(X, Low, High)

    # Parity
    Rule EvenZero:
    int_eq(N, 0)
    ------------
    even(0)

    Rule OddOne:
    int_eq(N, 1)
    -----------
    odd(1)

    Rule EvenRec:
    and(int_sub(N, 2, N2), even(N2))
    --------------------------------
    even(N)

    Rule OddRec:
    and(int_sub(N, 2, N2), odd(N2))
    -------------------------------
    odd(N)
End Global
