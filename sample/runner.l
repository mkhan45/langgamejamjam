Begin Facts:
    StateVar RunnerY
    StateVar RunnerVY
    StateVar ObstacleX
    eq(RunnerY, 0.0)
    eq(RunnerVY, 0.0)
    eq(ObstacleX, 30.0)
End Facts

Begin Global:
Rule ShouldJump:
    and(real_eq(RunnerY, 0.0), and(real_ge(ObstacleX, 8.0), real_le(ObstacleX, 18.0)))
    ---------------------------------------------------------------------------------
    shouldJump()

Rule RespawnObstacle:
    real_lt(ObstacleX, -3.0)
    ------------------------
    shouldRespawnObstacle()
End Global

Begin Stage Control:
    Begin State Constraints:
        or(
            and(shouldJump(), eq(next(RunnerVY), 3.5)),
            and(not(shouldJump()), eq(next(RunnerVY), RunnerVY))
        )
        or(
            and(shouldRespawnObstacle(), eq(next(ObstacleX), 30.0)),
            and(not(shouldRespawnObstacle()), real_sub(ObstacleX, 1.5, next(ObstacleX)))
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
