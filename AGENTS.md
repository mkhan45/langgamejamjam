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
