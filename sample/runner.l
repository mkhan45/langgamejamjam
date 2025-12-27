Begin Facts:
    StateVar RunnerY
    StateVar RunnerVY
    StateVar ObstacleX
    StateVar ObstacleVX
    StateVar Dead
    RunnerY = 0.0
    RunnerVY = 0.0
    ObstacleX = 100.0
    ObstacleVX = 1.5
    Dead = no
    true()
End Facts

Begin Global:
    Rule ShouldJump:
    real_eq(RunnerY, 0.0) & key_pressed(space)
    ----------------------------------------------
    shouldJump()

    Rule RespawnObstacle:
    ObstacleX .< -3.0
    ------------------------
    shouldRespawnObstacle()

    Rule GameOver:
    RunnerY .< 1.0 & ObstacleX .< 2.0
    ---------------------------------
    collided()
End Global

Begin Stage Control:
    Begin State Constraints:
        cond(shouldJump(), next(RunnerVY) = 3.5, next(RunnerVY) = RunnerVY)

        cond(shouldRespawnObstacle(), next(ObstacleX) = 100.0 & real_add(ObstacleVX, 0.25, next(ObstacleVX)), real_sub(ObstacleX, ObstacleVX, next(ObstacleX)) & next(ObstacleVX) = ObstacleVX)

        cond(collided(), next(Dead) = yes, next(Dead) = Dead)
    End State Constraints
End Stage Control

Begin Stage Physics:
    Begin State Constraints:
        real_sub(RunnerVY, 0.5, NewVY)
        real_add(RunnerY, RunnerVY, NewY)
        cond(NewY .> 0.0, next(RunnerY) = NewY & next(RunnerVY) = NewVY, next(RunnerY) = 0.0 & next(RunnerVY) = 0.0)
    End State Constraints
End Stage Physics

Begin Stage Draw:
    With
        Dead = no
    Draw
        rect(1.0, RunnerY, 1.0, 1.0)
        rect(ObstacleX, 0.0, 1.0, 1.0)

    With
        Dead = yes
    Draw
        rect(40.0, 40.0, 1.0, 1.0)
        rect(60.0, 40.0, 1.0, 1.0)
        rect(46.0, 25.0, 9.0, 1.0)
        rect(45.0, 23.0, 1.0, 2.0)
        rect(55.0, 23.0, 1.0, 2.0)
End Stage Draw
