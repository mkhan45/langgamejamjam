Begin Facts:
    item(apple, 10)
    item(banana, 5)
    item(orange, 8)
    item(mango, 12)
    coupon(apple, 3)
    coupon(mango, 5)
    true()
End Facts

Begin Global:
    Rule CanBuyEmpty:
    int_ge(Budget, 0)
    -----------------
    canBuy(nil, Budget)

    Rule CanBuyAdd:
    and(item(I, P), and(int_sub(Budget, P, Remaining), canBuy(Rest, Remaining)))
    ----------------------------------------------------------------------------
    canBuy(cons(I, Rest), Budget)

    Rule CartCostEmpty:
    true()
    ------
    cartCost(nil, 0, MaxSize)

    Rule CartCostAdd:
    and(int_ge(MaxSize, 1), and(int_sub(MaxSize, 1, NewMax), and(item(I, P), and(cartCost(Rest, RestCost, NewMax), int_add(P, RestCost, Total)))))
    ----------------------------------------------------------------------------------------------------------------------------------------------
    cartCost(cons(I, Rest), Total, MaxSize)

    Rule InCart:
    true()
    ------
    inCart(I, cons(I, Rest))

    Rule InCartTail:
    inCart(I, Rest)
    ---------------
    inCart(I, cons(X, Rest))

    Rule HasDiscount:
    and(inCart(I, Cart), coupon(I, Discount))
    -----------------------------------------
    hasDiscount(Cart, I, Discount)

    Rule Cheaper:
    and(item(A, Pa), and(item(B, Pb), int_lt(Pa, Pb)))
    --------------------------------------------------
    cheaper(A, B)

    Rule CartSizeEmpty:
    true()
    ------
    cartSize(nil, 0)

    Rule CartSizeAdd:
    and(cartSize(Rest, N), int_add(N, 1, N1))
    -----------------------------------------
    cartSize(cons(I, Rest), N1)

    Rule FullCart:
    and(canBuy(Cart, Budget), and(cartSize(Cart, N), int_ge(N, 3)))
    ---------------------------------------------------------------
    fullCart(Cart, Budget)
End Global
