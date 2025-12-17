# Langame AST Documentation

This document describes the Abstract Syntax Tree (AST) structure for the Langame relational language parser.

## Overview

The Langame AST is a hierarchical structure consisting of four main levels:

```
Module
  ├── Facts (Terms)
  ├── Global Stage
  │     └── Rule(s)
  │           ├── Premise (Term)
  │           └── Conclusion (Term)
  └── Stage(s)
        └── Rule(s)
              ├── Premise (Term)
              └── Conclusion (Term)
```

## AST Node Types

### Module

Represents a complete Langame file.

**Structure:**
```rust
pub struct Module<'a> {
    pub span: Span<'a>,
    pub facts: Vec<Term<'a>>,
    pub global_stage: Stage<'a>,
    pub stages: Vec<Stage<'a>>,
}
```

**Fields:**
- `span`: Source location information
- `facts`: Initial facts (terms) that are always true
- `global_stage`: Global rules that apply across all stages (named "Global")
- `stages`: Zero or more named stages

**Grammar:**
```
Module =
    Begin Facts:
    <term>*
    End Facts

    Begin Global:
    <rule>*
    End Global

    <stage>*
```

**Example:**
```
Begin Facts:
    initial(0)
    count(5)
End Facts

Begin Global:
Rule Increment:
    count(X)
    --------
    count(add(X, 1))
End Global

Begin Stage Arithmetic:
  ...
End Stage Arithmetic

Begin Stage Logic:
  ...
End Stage Logic
```

### Stage

Represents a named collection of rules.

**Structure:**
```rust
pub struct Stage<'a> {
    pub span: Span<'a>,
    pub name: &'a str,
    pub rules: Vec<Rule<'a>>,
}
```

**Fields:**
- `span`: Source location information
- `name`: Stage identifier
- `rules`: Zero or more rules within this stage

**Grammar:**
```
Begin Stage <StageName>:
<rule>*
End Stage <StageName>
```

**Example:**
```
Begin Stage Arithmetic:
Rule Add:
    add(X, Y)
    ---------
    sum(X, Y)
End Stage Arithmetic
```

### Rule

Represents an inference rule with a premise and conclusion.

**Structure:**
```rust
pub struct Rule<'a> {
    pub span: Span<'a>,
    pub name: &'a str,
    pub premise: Term<'a>,
    pub conclusion: Term<'a>,
}
```

**Fields:**
- `span`: Source location information
- `name`: Rule identifier
- `premise`: The condition or input term
- `conclusion`: The result or output term

**Grammar:**
```
Rule <name>:
    term     # premise
    --------
    term     # conclusion
```

**Example:**
```
Rule AddCommutative:
    add(X, Y)
    ---------
    add(Y, X)
```

### Term

Represents an expression in the language.

**Structure:**
```rust
pub struct Term<'a> {
    pub span: Span<'a>,
    pub contents: TermContents<'a>,
}

pub enum TermContents<'a> {
    App { rel: Rel<'a>, args: Vec<Term<'a>> },
    Atom { text: &'a str },
    Var { name: &'a str },
    Int { val: i32 },
    Float { val: f32 },
}
```

**Variants:**

1. **Application (App)**: Function/relation application
   - `rel`: The relation being applied (SMT or user-defined)
   - `args`: Arguments to the relation
   - Example: `add(X, Y)`, `mul(3, 4)`

2. **Atom**: Lowercase identifier
   - `text`: The atom text
   - Example: `foo`, `bar`, `int`

3. **Variable (Var)**: Uppercase identifier
   - `name`: The variable name
   - Example: `X`, `Y`, `Result`

4. **Integer (Int)**: Numeric integer
   - `val`: The integer value
   - Example: `42`, `-5`

5. **Float**: Numeric floating-point
   - `val`: The float value
   - Example: `3.14`, `-2.5`

**Grammar:**
```
term = <atom>                    # lowercase identifier
     | <int>                     # integer literal
     | <float>                   # float literal
     | <var>                     # uppercase identifier
     | <relation>(<term>,*)      # application
```

**Examples:**
```
X                    # Variable
42                   # Integer
3.14                 # Float
foo                  # Atom
add(X, Y)            # Application with 2 args
mul(add(1, 2), 3)    # Nested application
typeof(42, int)      # Mixed types
```

### Relation

Represents the relation/function in an application.

**Structure:**
```rust
pub enum Rel<'a> {
    SMTRel { name: &'a str },
    UserRel { name: &'a str },
}
```

**Variants:**
1. **SMTRel**: Built-in SMT solver relation
2. **UserRel**: User-defined relation

Currently, the parser treats all relations as `UserRel`.

## Parser Functions

The parser provides three main entry points:

1. **`parse_module`**: Parses a complete module (entire file)
2. **`parse_stage`**: Parses a single stage
3. **`parse_rule`**: Parses a single rule
4. **`parse_term`**: Parses a single term

All functions return `IResult<Span, T>` where `T` is the corresponding AST node type.

## Example Complete File

```
Begin Facts:
    initial(0)
    max_count(100)
    enabled(true)
End Facts

Begin Global:
Rule IncrementCounter:
    count(X)
    --------
    count(add(X, 1))
End Global

Begin Stage TypeSystem:
Rule IntegerType:
    typeof(42, int)
    ---------------
    valid(42)

Rule AdditionType:
    typeof(add(X, Y), int)
    ----------------------
    valid(add(X, Y))
End Stage TypeSystem

Begin Stage Arithmetic:
Rule AddCommutative:
    add(X, Y)
    ---------
    add(Y, X)

Rule Distributive:
    mul(add(X, Y), Z)
    -----------------
    add(mul(X, Z), mul(Y, Z))
End Stage Arithmetic
```

This would parse into:
- A `Module` with:
  - 3 facts: `initial(0)`, `max_count(100)`, `enabled(true)`
  - A global stage ("Global") with 1 rule
  - 2 named stages: "TypeSystem" and "Arithmetic"
- Stage 1 ("TypeSystem") with 2 rules
- Stage 2 ("Arithmetic") with 2 rules
- Each rule containing premise and conclusion terms with various structures (applications, variables, atoms, integers)

## Notes

- Identifiers starting with uppercase letters are parsed as **variables**
- Identifiers starting with lowercase letters are parsed as **atoms**
- **Facts must be on separate lines** - each fact in the Facts section must be followed by a line ending to prevent keywords like "End" and "Facts" from being parsed as terms
- The parser uses `nom` for parsing and `nom_locate` for span tracking
- All AST nodes carry span information for error reporting and source mapping

---

# IR Design (src/ir.rs)

The IR is a lower-level representation compiled from the AST, optimized for the solver.

## Core Types

### Term (IR)
```rust
pub enum Term {
    Var(VarId),
    Atom(SymbolId),
    Int(i32),
    Float(f32),
    App { sym: SymbolId, args: Vec<TermId> },
}
```

### Prop (Propositions)
```rust
pub enum Prop {
    True,
    False,
    Eq(TermId, TermId),
    And(PropId, PropId),
    Or(PropId, PropId),
    Not(PropId),
    App { rel: RelId, args: Vec<TermId> },
}
```

### RelKind
```rust
pub enum RelKind {
    User,      // Back-chainable user-defined relations
    SMTInt,    // Integer arithmetic constraints
    SMTReal,   // Real arithmetic constraints
}
```

## Arenas and Interning

- `Arena<T>`: Stores items, returns `Id<T>` handles
- `Interner<T>`: Deduplicates values (used for symbols)
- `Program`: Contains all arenas (terms, props, vars, symbols, rels) plus facts, global_rules, and stages

## Facts as Propositions

Facts are stored as `Vec<PropId>` and compiled using `lower_term_to_prop`. This unified representation means:

- `position(player, 0, 0)` → `Prop::App { rel: position, args: [...] }`
- `eq(X, 1)` → `Prop::Eq(X_term, 1_term)`

When solving a query, all facts are conjoined as goals with the query. This allows:
- `Prop::App` facts to be matched during back-chaining
- `Prop::Eq` facts to constrain variables via unification

Variables defined in facts persist into query scope via the `var_map` stored in Frontend.

### Fact-Query Interaction Examples

**Constraining queries with eq facts:**
```
Begin Facts:
    eq(X, 1)
End Facts
```
- Query `eq(X, 2)` → **fails** (X is already bound to 1)
- Query `eq(X, Y)` → **succeeds** with Y=1
- Query `eq(X, cons(A, B))` → **fails** (1 doesn't unify with cons(A, B))

**Relational facts with back-chaining:**
```
Begin Facts:
    position(player, 0, 0)
End Facts
```
- Query `position(player, X, Y)` → **succeeds** with X=0, Y=0
- Query `position(enemy, X, Y)` → **fails** (no matching fact)

**Key insight:** There's no semantic difference between a relational fact and an asserted proposition. All facts are just propositions conjoined with queries during solving.

---

# Solver Architecture (src/solver/engine.rs)

miniKanren-style relational solver with constraint support.

## Core Components

### State
```rust
pub struct State {
    pub subst: Subst,           // Variable substitution map
    pub constraints: ConstraintStore,  // Pending arithmetic constraints
    pub goals: Vector<PropId>,  // Goal queue (FIFO)
}
```

### Subst (Substitution)
- `walk(t, terms)`: Follows variable chains to resolve a term
- `extend(v, t)`: Returns new Subst with v→t binding
- `unify(t1, t2, terms)`: Structural unification, returns Option<Subst>
- `unify_args(args1, args2, terms)`: Unifies argument lists pairwise

### Unification
- Variables unify with anything (extending substitution)
- Atoms/Ints/Floats unify only if equal
- `Term::App` unifies if same symbol and args unify pairwise

### Solver
```rust
pub struct Solver<'a> {
    pub program: &'a mut Program,
}
```

- `query(goal: PropId)` → `SolutionIter`: Returns iterator over solutions (BFS by default)
- `query_with_strategy(goal, strategy)` → `SolutionIter`: Use specific search strategy
- `SolutionIter.with_limit(n)`: Limit number of solutions
- `SolutionIter.with_max_steps(n)`: Limit search steps

### SearchStrategy

```rust
pub enum SearchStrategy {
    BFS,  // Breadth-first (default) - explores all states at depth N before N+1
    DFS,  // Depth-first - explores deepest states first
}
```

Use `SearchStrategy::DFS` when you want to find solutions quickly without exploring all branches at each depth. BFS is fairer but may use more memory on deep search trees.

### Goal Processing

1. **Prop::True**: Pop goal, continue
2. **Prop::And(p1, p2)**: Push both p1 and p2 as goals
3. **Prop::Or(p1, p2)**: Branch search (two states)
4. **Prop::Eq(t1, t2)**: Unify terms
5. **Prop::App (User rel)**: Back-chain through facts and rules
6. **Prop::App (SMT rel)**: Add to constraint store

### Back-chaining

For User relations:
1. Try unifying with each fact
2. Try each rule whose head matches
3. `rename_term()` / `rename_prop()` create fresh variable copies when instantiating clauses

### ArithConstraint
```rust
pub enum ArithConstraint {
    IntEq, IntLt, IntLe, IntGt, IntGe, IntNeq,
    IntAdd, IntSub, IntMul, IntDiv,
    RealEq, RealLt, RealLe, RealGt, RealGe, RealNeq,
    RealAdd, RealSub, RealMul, RealDiv,
}
```

### Z3 Constraint Solving

The solver uses Z3 (via the `z3` crate with bundled feature) to solve arithmetic constraints.

**How it works:**
1. When the solver reaches a state with no remaining goals but has pending constraints, it calls `ConstraintStore::solve()`
2. `solve()` translates `ArithConstraint`s to Z3 assertions
3. Z3 checks satisfiability and, if SAT, extracts a model
4. Variable bindings from the model are added back to the substitution

**Term to Z3 conversion:**
- `Term::Int(i)` → Z3 integer constant
- `Term::Float(f)` → Z3 rational (float × 1,000,000 / 1,000,000 for precision)
- `Term::Var(v)` → Z3 integer/real variable (fresh if not in map)

**Constraint translation:**
| ArithConstraint | Z3 Assertion |
|-----------------|--------------|
| `IntAdd(a, b, c)` | `a + b = c` |
| `IntSub(a, b, c)` | `a - b = c` |
| `IntMul(a, b, c)` | `a * b = c` |
| `IntDiv(a, b, c)` | `a / b = c` |
| `IntEq(a, b)` | `a = b` |
| `IntNeq(a, b)` | `a ≠ b` |
| `IntLt/Le/Gt/Ge` | `a < b`, `a ≤ b`, etc. |
| Real variants | Same, using Z3 Real sort |

**Example flow:**
```
Query: next(Y) with rule next(int_add(X, 1)) :- value(X) and fact value(5)
1. Back-chain: unify Y with int_add(X, 1), add value(X) as goal
2. Unify X=5 from fact
3. int_add(5, 1, Y) added to constraint store
4. No more goals → solve constraints
5. Z3 solves: 5 + 1 = Y → Y = 6
6. Solution returned with Y=6
```

---

# Compilation (src/ast/compile.rs)

## Compiler
- Takes `&mut Program` reference
- `compile_module(module)`: Compiles full AST module to IR
- `compile_query(term)`: Compiles a query term, returns `(PropId, Vec<(String, TermId)>)` with query variables

## Lowering Rules

- **AST App → IR Term::App**: For non-SMT function applications (user-defined functions)
- **SMT relations**: Desugared with fresh variables (e.g., `int_add(X, 1)` in arg position becomes fresh var + constraint)
- **`and(P, Q)`**: Becomes `Prop::And`
- **`or(P, Q)`**: Becomes `Prop::Or`
- **`eq(X, Y)`**: Becomes `Prop::Eq` for structural unification
- **Facts**: Lowered via `lower_term_to_prop()` to propositions (App facts become `Prop::App`, eq facts become `Prop::Eq`)

## SMT Relations (builtin)
```
int_eq, int_neq, int_lt, int_le, int_gt, int_ge
int_add, int_sub, int_mul, int_div
real_eq, real_neq, real_lt, real_le, real_gt, real_ge
real_add, real_sub, real_mul, real_div
```

---

# Frontend (src/main.rs)

## Frontend struct
```rust
pub struct Frontend {
    pub program: Program,
    pub var_map: HashMap<String, TermId>,
}
```

- `load(source)`: Parses and compiles source string, stores var_map from compilation
- `query(query_str)`: Runs query with default limit (10 solutions)
- `query_with_limit(query_str, n)`: Custom solution limit

## FFI Functions
All public functionality exposed via C FFI for JS/WASM integration:
- `create_frontend`, `free_frontend`
- `frontend_load`, `frontend_query`
- `frontend_fact_count`, `frontend_rule_count`, `frontend_stage_count`

**Key constraint**: `main()` is empty - everything uses FFI for JS integration.

---

# Running Tests

```bash
cargo test
```

# Building

```bash
cargo build
cargo build --release
```
