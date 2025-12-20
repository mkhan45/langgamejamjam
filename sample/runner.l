Begin Facts:
    StateVar RunnerY
    StateVar RunnerVY
    StateVar ObstacleX
    StateVar Dead
    eq(RunnerY, 0.0)
    eq(RunnerVY, 0.0)
    eq(ObstacleX, 100.0)
    eq(Dead, no)
    true()
End Facts

Begin Global:
    Rule ShouldJump:
    and(real_eq(RunnerY, 0.0), key_pressed(space))
    ----------------------------------------------
    shouldJump()

    Rule RespawnObstacle:
    real_lt(ObstacleX, -3.0)
    ------------------------
    shouldRespawnObstacle()

    Rule GameOver:
    and(real_lt(RunnerY, 1.0), real_lt(ObstacleX, 2.0))
    ----------------------------------------------------
    collided()
End Global

Begin Stage Control:
    Begin State Constraints:
        or(
            and(shouldJump(), eq(next(RunnerVY), 3.5)),
            and(not(shouldJump()), eq(next(RunnerVY), RunnerVY))
        )
        or(
            and(shouldRespawnObstacle(), eq(next(ObstacleX), 100.0)),
            and(not(shouldRespawnObstacle()), real_sub(ObstacleX, 1.5, next(ObstacleX)))
        )
        or(
            and(collided(), eq(next(Dead), yes)),
            and(not(collided()), eq(next(Dead), Dead))
        )
    End State Constraints
End Stage Control

Begin Stage Physics:
    Begin State Constraints:
        real_sub(RunnerVY, 0.5, NewVY)
        real_add(RunnerY, RunnerVY, NewY)
        or(
            and(real_gt(NewY, 0.0), and(eq(next(RunnerY), NewY), eq(next(RunnerVY), NewVY))),
            and(real_le(NewY, 0.0), and(eq(next(RunnerY), 0.0), eq(next(RunnerVY), 0.0)))
        )
    End State Constraints
End Stage Physics

Begin Stage Draw:
    Rule DrawPlayer:
    not(eq(Dead, yes))
    --------------
    rect(1.0, RunnerY, 1.0, 1.0)

    Rule DrawObstacle:
    not(eq(Dead, yes))
    --------------
    rect(ObstacleX, 0.0, 1.0, 1.0)

    Rule DrawLEye:
    eq(Dead, yes)
    -------------
    rect(40.0, 40.0, 1.0, 1.0)

    Rule DrawREye:
    eq(Dead, yes)
    -------------
    rect(60.0, 40.0, 1.0, 1.0)
    
    Rule DrawFrown:
    eq(Dead, yes)
    -------------
    rect(46.0, 25.0, 9.0, 1.0)

    Rule DrawFrownL:
    eq(Dead, yes)
    -------------
    rect(45.0, 23.0, 1.0, 2.0)

    Rule DrawFrownR:
    eq(Dead, yes)
    -------------
    rect(55.0, 23.0, 1.0, 2.0)
End Stage Draw
