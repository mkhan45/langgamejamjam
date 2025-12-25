# gg3z

A relational/logic/SMT language for developing games, developed for the [2025 Lang Jam Game Jam](https://itch.io/jam/langjamgamejam).
Try out the editor and a simple game: <https://mkhan45.github.io/gg3z/>

`gg3z` was rated the most creative language in the jam, but I'm a bit embarrassed that I barely wrote a game for it.
Check out the submission here: <https://itch.io/jam/langjamgamejam/rate/4136346>

# Relational Programming

e.g. Prolog, Minikanren, SQL

tldr; instead of writing a function `y = f(x)`, we could write a relation `f(x, y)`.
Then, as a programmer, we can "run" our relations backwards, e.g. `f(X, y)` would
provide us with a value for `X` such that `f(X) = y`. A good mental model is that
we are building a series of constraints on `X`, so we could write `less_than(X, 5); greater_than(0, X)`
and the solver could provide `X = 3` or `X = 1`.

<details>
<summary>Aside</summary>
I like relational programming; it's more abstract than functional programming but suffers
because programmers usually care to specify what the computer should do rather than what
we want. This type of thinking is so natural to programmers that even giving an example
of the power of relational languages can be difficult, because common go-to language
examples like list sorting or FizzBuzz are implicitly still processes.

Relational programming is a bad fit for gamedev for performance reasons, but also because
of their realtime state / IO, and game ticks/timesteps. So there's not one obvious way
to build a game engine on a relational language, but I took the easiest path of imposing
constraints on the next time step. It's kind of an imperative shell, functional core thing.
In the future, I would like to imagine an engine that is more free from its timestep such that even numerical
integration methods are abstracted.
</details>

# Concept

Relational programming's reversible computation seems impossible to use for gamedev
since any reverse computation might do a big search, and if relations are only used "forwards" then
it's just functional programming. But I thought it might have value for an interactive game editor
where realtime performance isn't a concern. A pain-point in gamedev is creating a scene or game state
to easily test some interaction or mechanic, so it would be cool if the editor allowed developers to query
a for a specific game state and then run the game in realtime from there. It also motivates a workflow
where developers write the premise of the relation and then are provided a visualization of the game state
that helps motivate the relation's conclusion.

If relations only need to be reversed in the editor, we can also compile them to functional or even imperative
code for the actual realtime engine. We could also imagine annotations and a "mode"-checker that lets us know
if a definition might use a slow search.

I didn't actually finish either of these features in time for the end of the jam, but the core pieces should be there.
There are some issues with constraint scoping and stages and FFI and such that blocked state queries working properly on the
frontend, but I would like to implement it soon. For the demo game, running the full solver every frame was fast enough,
so I didn't bother compiling anything yet.

# Language Design

TODO: details

Declare some state variables, and then write rules and constraints on their value after the next timestep using
the `next` relation.

# Name

[`ggez`](https://github.com/ggez/ggez) was one of the first game engines I used, and it was hugely
beneficial for my first real projects. `gg3z` uses the Z3 solver so I considered `ggz3`, but `gg3z`
sounds/looks cooler and hints at the reversible computation model.
