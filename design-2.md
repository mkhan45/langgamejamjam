```
Begin Initial State
    StateVars RunnerY, RunnerVY, ObstacleX, ObstacleVX

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
    Begin Rules
        ...
    End Rules

    Begin State Query
        QueryVars RunneryVY, ObstacleX, Dead
        ...
    End State Query
End Stage Control
```
