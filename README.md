# Roast — lightweight Rust REPL language

Roast is a small statically-typed toy language with a persistent REPL implemented in Rust.
It includes a parser (pest), a typed AST & analyser, and a simple runtime with closures.
This repository is intended for learning, experimentation, and as a foundation for language work.

Key points
- REPL-style interactive shell with line-editing (via `crossterm`).
- Parsing implemented with `pest` (`src/grammar/roast.pest`).
- A typed AST and simple type analyser in `src/typed_ast.rs`.
- Interpreter/runtime in `src/main.rs` using `Environment` + `Value` model.

Quick start

- Build:
```roast/README.md#L1-4
cargo build --release
```

- Run the REPL:
```roast/README.md#L5-8
cargo run --release
# or for debug:
cargo run
```

REPL usage

When you run `cargo run` you will start the Roast REPL. The REPL supports:
- Editing keys: arrows, Home, End, Delete, Backspace.
- Multi-line input (blocks are accepted; REPL detects completeness).
- Commands that start with `:` (colon).

Supported REPL commands
- `:help` — show brief help
- `:exit` / `:quit` — exit
- `:reset` — clear type symbol table and runtime environment
- `:env` — print current runtime `Environment`
- `:history` — show in-session history of inputs

Example REPL session
```roast/README.md#L9-22
> let x: i64 = 10;
> x
10
> fn add(a: i64, b: i64) -> i64 { a + b }
> add(2, 3)
5
> :env
# prints environment/state
```

Language summary

- Statements: `let`, assignment, function declaration, `return`, expression statements.
- Types: `i32`, `i64`, `f32`, `f64`, `bool`, `String` (and `str` in grammar).
- Expressions: binary (`+ - * / %`, comparisons), unary (`-`, `!`), function calls, literals, identifiers.
- Blocks use `{ ... }` and can contain statements and a final return expression (REPL grammar supports a `return_expression` in a block).

Grammar (high level)
The grammar is implemented in `src/grammar/roast.pest`. Example of the top-level rules:
```roast/src/grammar/roast.pest#L1-40
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }

program = { SOI ~ statement* ~ EOI }

statement = {
    let_statement
    | assignment_statement
    | function_declaration
    | return_statement
    | expression_statement
}
```
See the full file for literals, precedence and function rules.

Project layout (important files)
- `Cargo.toml` — crate metadata and dependencies (`pest`, `pest_derive`, `crossterm`).
- `src/main.rs` — REPL, runtime `Value` / `Environment`, evaluator (`eval_expression`, `exec_statement`, `eval_block`).
- `src/parser.rs` — uses `pest` to build `ast::Program`.
- `src/ast.rs` — the untyped AST (statements, expressions, operators).
- `src/typed_ast.rs` — typed AST, symbol table, type analysis (`Repl` helper lives here).
- `src/grammar/roast.pest` — grammar definition.

Implementation notes (things you should know as a contributor)
- Parsing: `parser.rs` uses `pest` and builds the untyped AST (`ast::Program`).
- Type analysis: `typed_ast.rs` performs a pass to infer/check types and produces `TypedProgram`. The REPL keeps a `Repl` state with persistent symbols.
- Runtime: `main.rs` contains a simple environment model `Environment` (vector of HashMaps for scopes). Functions capture their defining `Environment` to provide closures.
- The evaluator supports:
  - Integer/float arithmetic for same-typed operands (I32 vs I64, F64). Some type combinations are not mixed/coerced implicitly.
  - String concatenation for `+` when both operands are `String`.
  - Function calls create a new scope on top of the captured closure environment and bind parameters to argument values.

Limitations / current simplifications
- The runtime `Value::get_type` simplifies function types to `Unit` in places — functional typing isn't fully expressed at runtime.
- No standard library of functions (you can define your own functions inside the REPL).
- Limited type coercions and diagnostics compared to mature languages.
- Error reporting: parse/type/runtime errors are reported to the REPL, but error messages can be improved for clarity and source locations.
- No persistent storage or project-level compilation pipeline — the focus is REPL/experimentation.

Examples (language snippets)

Let and assignment:
```roast/README.md#L23-30
let mut count: i64 = 0;
count = count + 1;
```

Function:
```roast/README.md#L31-38
fn factorial(n: i64) -> i64 {
    if n == 0 {
        return 1
    } else {
        return n * factorial(n - 1)
    }
}
factorial(5)
```

Note: The grammar currently does not include `if`/`else` in the grammar snippet shown above; if/else or control-flow constructs would need grammar and evaluator additions. (Use functions and early `return` for control-flow today.)
