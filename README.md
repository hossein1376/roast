# Roast — lightweight Rust REPL language

Roast is a small statically-typed toy language with a persistent REPL implemented in Rust.
It includes a parser (pest), a typed AST & analyser, and a simple runtime with closures.
This repository is intended for learning, experimentation, and as a foundation for language work.

The language rules, including features that are not implemented yet, are in [docs/spec.md](docs/spec.md).

Key points
- REPL-style interactive shell with line-editing (via `crossterm`).
- Parsing implemented with `pest` (`src/grammar/roast.pest`).
- A typed AST and simple type analyser in `src/typed_ast.rs`.
- Interpreter/runtime in `src/main.rs` using `Environment` + `Value` model.

Quick start

- Build:

```sh
cargo build --release
```

- Run the REPL:

```sh
cargo run --release
# or for debug:
cargo run
```

- Run a file (no REPL). Output is whatever `print` writes. Errors go to stderr and the process exits non-zero:

```sh
cargo run --release -- program.roast
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

```text
> let x: i64 = 10;
> x
10
> func add(a: i64, b: i64) -> i64 { a + b }
> add(2, 3)
5
> :env
# prints environment/state
```

Language summary

- Statements: `let`, assignment, function declaration, `return`, `if` / `else`, `while`, expression statements.
- Types: `i32`, `i64`, `f32`, `f64`, `bool`, `String` (and `str` in grammar).
- Expressions: binary (`+ - * / %`, comparisons), unary (`-`, `!`), function calls, literals, identifiers.
- Blocks use `{ ... }`. A final expression is the block value when that block is a function body. Only `return` leaves a function early.
- `print(value)` is a builtin. It writes one line and returns unit. The name `print` is reserved.
- `if` and `while` conditions must be `bool`. `else` and `else if` are optional. `break` and `continue` apply to the innermost `while`.

Grammar (high level)
The grammar is implemented in `src/grammar/roast.pest`. Example of the top-level rules:
```text
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }

program = { SOI ~ statement* ~ EOI }

statement = {
    let_statement
    | assignment_statement
    | function_declaration
    | return_statement
    | if_statement
    | while_statement
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
- `print` is the only builtin.
- No implicit numeric coercion. `i32` and `i64` do not mix. An unsuffixed number takes a nearby annotation or the other operand's numeric type when it fits, and otherwise defaults to `i64` or `f64`.
- Parse errors include pest locations. Type and runtime errors are messages without source spans.
- Closures snapshot the environment at definition. Direct recursion works. Mutual recursion does not.
- No REPL session save/load. Pass a source file to run it once.

Examples (language snippets)

Let and assignment:

```text
let mut count: i64 = 0;
count = count + 1;
```

Function:

```text
func factorial(n: i64) -> i64 {
    if n == 0 {
        return 1
    } else {
        return n * factorial(n - 1)
    }
}
factorial(5)
```

`factorial(5)` evaluates to `120`. A function calls itself by name. A function does not see other functions defined later.

Loop:

```text
let mut i: i64 = 0
let mut s: i64 = 0
while i < 5 {
    s = s + i
    i = i + 1
}
print(s)
```

`else if`, `break`, and `continue`:

```text
let mut i: i64 = 0
while i < 10 {
    i = i + 1
    if i == 2 {
        continue
    } else if i == 4 {
        break
    }
    print(i)
}
```
