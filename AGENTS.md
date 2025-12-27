# Langame Reference

A miniKanren-style relational logic language with SMT constraint support, designed for game logic.

## Commands
```bash
cargo test           # Run tests
cargo build          # Build
```

---

## Language Syntax

```
Begin Facts:
    <term>*              # Facts: propositions always true
End Facts

Begin Global:
    <rule>*              # Rules available in all stages
End Global

Begin Stage <Name>:
    <rule>*
    Begin State Constraints:   # Optional: define state transitions
        <term>*
    End State Constraints
    <draw_directive>*          # Optional: draw directives
End Stage <Name>
```

### Comments
- **Line comments**: Start with `#` and continue to end of line
- **Placement**: Can be used anywhere whitespace is allowed

### Terms
- **Variables**: Uppercase identifiers (`X`, `Health`, `Result`)
- **Atoms**: Lowercase identifiers (`player`, `foo`)
- **Literals**: Integers (`42`) and floats (`3.14`)
- **Applications**: `relation(arg1, arg2, ...)`

### Rules
```
Rule <Name>:
    <premise>       # Condition
    --------
    <conclusion>    # What follows if premise holds
```

### Draw Directives
Forward-evaluated drawing commands. Unlike rules (which use back-chaining), draw directives evaluate the condition and emit draw commands for each solution.

```
With
    <condition>     # Query to evaluate
Draw
    <term>*         # Commands to emit for each solution

Draw
    <term>*         # Unconditional (no With clause)
```

**Example:**
```
Begin Stage Draw:
    With
        Dead = no
    Draw
        rect(1.0, PlayerY, 1.0, 1.0)
        rect(ObstacleX, 0.0, 1.0, 1.0)

    With
        Dead = yes
    Draw
        rect(40.0, 40.0, 1.0, 1.0)
        rect(60.0, 40.0, 1.0, 1.0)
End Stage Draw
```

**Semantics:**
1. For each `With/Draw` block, query the condition
2. For each solution, substitute bindings into draw terms
3. Collect all resulting draw commands
4. Unconditional `Draw` blocks (no `With`) always emit their draws

State variables in draw terms are interpolated with current values.

### Logical Connectives
- `and(P, Q)` — conjunction
- `or(P, Q)` — disjunction
- `not(P)` — negation (via failure to prove)
- `eq(X, Y)` — structural unification

### Operator Syntax (Sugar)
Binary and unary operators as syntactic sugar. Precedence from lowest to highest:

| Operator | Desugars to | Notes |
|----------|-------------|-------|
| `a \| b`, `a ∨ b` | `or(a, b)` | Disjunction |
| `a & b`, `a ∧ b` | `and(a, b)` | Conjunction |
| `<`, `<=`, `>`, `>=` | `int_lt`, `int_le`, `int_gt`, `int_ge` | Integer comparison |
| `==` | `int_eq` | Integer equality |
| `.<`, `.<=`, `.>`, `.>=`, `.==` | `real_lt`, `real_le`, `real_gt`, `real_ge`, `real_eq` | Real comparison |
| `a = b` | `eq(a, b)` | Structural unification (binds tighter than `&`, `\|`) |
| `!a`, `¬a` | `not(a)` | Negation (unary prefix) |

All binary operators are left-associative. Parentheses work for grouping: `(a | b) & c`.

### SMT Relations (solved via Z3)
```
int_eq, int_neq, int_lt, int_le, int_gt, int_ge
int_add, int_sub, int_mul, int_div
real_eq, real_neq, real_lt, real_le, real_gt, real_ge
real_add, real_sub, real_mul, real_div
```

---

## Language Semantics

### Declarative Reading
A rule `P -------- C` means: "to prove C, prove P." Facts are axioms—always provable.

A query asks: "find substitutions θ such that θ(query) is provable from facts + rules."

### Proof Search
Langame uses **SLD resolution** with **back-chaining**:
1. Start with query as the goal
2. For each goal, find a fact or rule head that unifies with it
3. Replace goal with the rule's premise (instantiated)
4. Repeat until no goals remain

Multiple unifying facts/rules create **choice points** → branching search → multiple solutions.

### Closed-World Assumption
If a proposition cannot be proven, it is false. This enables `not(P)`: succeeds iff P has no proof.

---

## Solver

### State
```rust
struct State {
    subst: Subst,                    // θ: Var → Term
    constraints: ConstraintStore,    // Deferred SMT constraints
    goals: Vector<PropId>,           // Remaining propositions to prove
}
```

A **solution** is a State with empty goals and satisfiable constraints.

### Search Algorithm
```
                    ┌─────────────────────────────────────────┐
                    │          SearchQueue (BFS/DFS)          │
                    │  [State₁, State₂, State₃, ...]          │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              Pop State                   │
                    └─────────────────────────────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
               goals empty?                      goals non-empty
                      │                                 │
                      ▼                                 ▼
           ┌──────────────────┐              ┌──────────────────┐
           │  Solve SMT       │              │  Pop first goal  │
           │  constraints     │              │  Process it      │
           └──────────────────┘              └──────────────────┘
                      │                                 │
                      ▼                                 ▼
               SAT? → Solution              Push resulting state(s)
              UNSAT? → Discard               back to queue
```

### Goal Processing Rules

| Goal Type | Processing |
|-----------|------------|
| `True` | Remove goal, continue with same state |
| `False` | Discard state (proof failed) |
| `And(P, Q)` | Replace with two goals: P, Q |
| `Or(P, Q)` | Branch: create two states, one with P, one with Q |
| `Eq(t₁, t₂)` | Unify t₁ and t₂; fail if non-unifiable |
| `Not(P)` | Succeeds iff P has no satisfiable proof (see below) |
| `App` (user) | Back-chain against facts and rules |
| `App` (SMT) | Add constraint to store, defer solving |

### Back-Chaining (User Relations)

When goal is `rel(args...)`:
1. **Try each fact**: If fact is `rel(fact_args...)`, unify `args` with `fact_args`. On success, goal is solved.
2. **Try each rule**: If rule conclusion is `rel(rule_args...)`:
   - Rename all variables in rule to fresh names (prevents capture)
   - Unify `args` with `rule_args`
   - On success, push rule's premise as new goal

Each successful unification creates a new branch in the search.

### Unification

`unify(t₁, t₂, subst) → Option<subst'>`

```
unify(X, t)         = Some(subst[X → t])           if X not in t
unify(t, X)         = Some(subst[X → t])           if X not in t
unify(atom, atom)   = Some(subst)                  same atom
unify(n, n)         = Some(subst)                  same int/float
unify(f(a...), f(b...)) = unify_args(a..., b...)   same functor
unify(_, _)         = None                         otherwise
```

**Occurs check**: Not currently implemented (infinite terms possible).

**Walk**: Before unifying, `walk(term, subst)` follows variable bindings to their ultimate value.

### Negation as Failure

`not(P)` processing:
1. Create a sub-search for P
2. Exhaust all branches
3. If ANY branch yields a satisfiable solution → `not(P)` fails
4. If NO branch yields a satisfiable solution → `not(P)` succeeds

**Critical**: A branch with unsatisfiable SMT constraints is NOT a solution—it must be discarded before concluding negation.

### SMT Constraint Solving

SMT relations (`int_add`, `real_lt`, etc.) deferred during SLD resolution:
1. When encountered, add constraint to `ConstraintStore`
2. Continue processing other goals
3. At solution time (empty goals), invoke Z3 on all accumulated constraints
4. If SAT: extract model, bind variables → solution
5. If UNSAT: discard state

**Two constraint operations:**
- `propagate_ground(subst, program, z3_solver)` — Eager propagation during search: solve only fully-ground constraints, defer variables. Used for in-search pruning.
- `solve_all(subst, program, z3_solver)` — Final validation after proof search: solve all constraints (ground and non-ground). Critical for correctness in negation.

**Constraint translation** (examples):
```
int_add(A, B, C)  →  A + B = C
int_lt(A, B)      →  A < B
real_div(A, B, C) →  A / B = C
```

---

## Compilation

### Term Lowering
```
AST Variable "X"    →  Term::Var(fresh_id) or lookup in var_map
AST Atom "foo"      →  Term::Atom(intern("foo"))
AST Int 42          →  Term::Int(42)
AST App f(a,b,c)    →  Term::App { sym: f, args: [lower(a), lower(b), lower(c)] }
```

### Proposition Lowering
```
and(P, Q)           →  Prop::And(lower(P), lower(Q))
or(P, Q)            →  Prop::Or(lower(P), lower(Q))
not(P)              →  Prop::Not(lower(P))
eq(X, Y)            →  Prop::Eq(lower(X), lower(Y))
user_rel(args...)   →  Prop::App { rel: user_rel, args: [...] }
smt_rel(args...)    →  Prop::App { rel: smt_rel, args: [...] }  (RelKind::SMT*)
```

### Rule Compilation
```
Rule Name:
    premise
    --------
    conclusion

→ IrRule {
    name: "Name",
    premise: lower_to_prop(premise),
    conclusion: lower_to_prop(conclusion),
  }
```

During back-chaining, rules are **instantiated** with fresh variables to prevent capture.

---

## State Variables

For mutable game state that persists across stage executions.

### Declaration
```
Begin Facts:
    StateVar Health
    eq(Health, 100)
End Facts
```

### The `next()` Intrinsic
`next(VarName)` creates a fresh variable for the next-state value:
```
int_sub(Health, 10, next(Health))   # next Health = current - 10
```
Without `next()`, `int_sub(Health, 10, Health)` would require `Health - 10 = Health` (unsatisfiable).

### State Constraints
Defined in stages; must have **exactly one solution** (deterministic updates):
```
Begin Stage Update:
Begin State Constraints:
    int_sub(Health, 5, next(Health))
End State Constraints
End Stage Update
```

### Architecture Principle
**Facts are the single source of truth.** `var_map` is a runtime index.

1. `StateVar X` + `eq(X, v)` → fact `eq(term_x, v)` added, `var_map["X"] = term_x`
2. Query `eq(X, Y)` → compiler resolves X via `var_map` → unifies against fact
3. `run_stage()` → solve constraints → update `var_map` + replace facts atomically

---

## IR Types

```rust
// Terms
enum Term {
    Var(VarId), Atom(SymbolId), Int(i32), Float(f32),
    App { sym: SymbolId, args: Vec<TermId> }
}

// Propositions
enum Prop {
    True, False,
    Eq(TermId, TermId),
    And(PropId, PropId), Or(PropId, PropId), Not(PropId),
    App { rel: RelId, args: Vec<TermId> }
}

// Relation kinds
enum RelKind { User, SMTInt, SMTReal }
```

---

## Frontend API

```rust
struct Frontend {
    program: Program,
    var_map: HashMap<String, TermId>,  // State variable index
    strategy: SearchStrategy,          // BFS (default) or DFS
    max_steps: usize,
}

struct SolutionSet {
    solutions: Vec<State>,
    reason: TerminationReason,
}

enum TerminationReason {
    LimitReached,      // Hit solution limit
    SearchExhausted,   // No more proof search branches
    MaxStepsReached,   // Hit step limit during search
}
```

**Methods:**
- `load(source)` → `Result<(), String>` — parse and compile
- `query_batch(query_str, limit)` → `Result<Vec<String>, String>` — batch query with solution limit
- `query_start(query_str)` / `query_next()` — incremental query
- `has_more_solutions()` — check if more results available
- `query_stop()` — abandon current incremental query
- `run_stage(index)` / `run_stage_by_name(name)` → `Result<(), String>` — execute state transitions
- `get_state_var(name)` → `Option<String>` — get current state value
- `state_vars()` → `Vec<(String, String)>` — get all state variables

**Solution Collection:**
`collect_solutions(goal, strategy, limit, max_steps) → SolutionSet` collects up to `limit` solutions. The `reason` field distinguishes between:
- **LimitReached**: Requested `limit` solutions found; may have more.
- **SearchExhausted**: No more proof branches; found all solutions.
- **MaxStepsReached**: Step limit hit; result inconclusive.

Critical for state constraint determinism checking: when requesting 2 solutions, `LimitReached` proves non-determinism (≥2 solutions exist), while `MaxStepsReached` is inconclusive.

**FFI**: All exposed via C FFI for JS/WASM. `main()` is empty.

---

## Other Guidelines

- Avoid using `mod.rs`. Instead, for a module `example`, use an `example.rs` file and an `example/` directory.
- Prefer higher-level conceptual reasoning to concrete code reasoning. Motivate code changes with a mental model of the problem when applicable.
- Remember that we use immutable data structures in the solver
- Compiling is sometimes extremely expensive.
    - For tests, never use --release.
    - Agents probably shouldn't use ./build-web.sh, but if so it should always be `./build-web.sh release`
    - Never do a clean build or run cargo clean.
