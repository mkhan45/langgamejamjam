```
Begin Initial State
    StateVars RunnerY, RunnerVY, ObstacleX, ObstacleVX

    -- these are constraints, not necessarily assignments
    RunnerY = 0.0
    RunnerVY = 0.0
    ObstacleX = 100.0
    ObstacleVX = 1.5
    Dead = no
End Initial State

Begin Global Rules
...
End Global Rules

Begin Stage Control
    Rules:
        ...

    State Query:
        QueryState RunneryVY, ObstacleX, Dead

        when shouldJump(): next(RunnerVY) = 3.5,
        otherwise: next(RunnerVY) = RunnerVY

        -- maybe add a preserve(X) sugar for next(X) = X
        when shouldRespawnObstacle():
            next(ObstacleX) = 100.0 ∧ real_add(ObstacleVX, 0.25, next(ObstacleVX))
        otherwise:
            real_sub(ObstacleX, ObstacleVX, next(ObstacleX)) ∧ next(ObstacleVX) = ObstacleVX

        when collided(): next(Dead) = yes
        otherwise: preserve(Dead)
End Stage Control

Begin Stage Physics:
    State Query for RunnerY, RunnerVY:
        real_sub(RunnerVY, 0.5, NewVY)
        real_add(RunnerY, RunnerVY, NewY)

        when NewY .> 0.0: next(RunnerY) = NewY ∧ next(RunnerVY) = NewVY
        otherwise: next(RunnerY) = 0.0 ∧ next(RunnerVY) = 0.0
End Stage Physics

Begin Stage Draw:
    ...
End Stage Draw
```
