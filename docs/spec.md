# Roast language specification

Roast is a small statically typed language with an interpreter and a persistent REPL. This document describes the language implemented in this repository, then the features that are planned and not implemented.

A **current** rule is what the compiler and interpreter do today. A **future** rule is a design target. A program that depends on a future rule is rejected or behaves as the current section says.

## 1. Source form

A program is a sequence of statements. The grammar is `src/grammar/roast.pest`.

Whitespace is a space, tab, newline, or carriage return. It separates tokens and is otherwise ignored. Newlines do not end statements.

A semicolon after a statement is optional.

There are no comments. A future revision will accept `//` line comments and `/* */` block comments, which nest no deeper than one level.

### 1.1 Identifiers

An identifier starts with an ASCII letter and continues with ASCII letters, digits, or `_`.

`true` and `false` are boolean literals, not identifiers. `print` is a reserved function name.

Other words (`let`, `mut`, `func`, `return`, `if`, `else`, `while`, `break`, `continue`) are recognized in statement position. They are not fully reserved inside expressions. A future revision will reserve them everywhere.

### 1.2 Literals

| Form | Example | Type before context |
| --- | --- | --- |
| Integer | `0`, `42` | `i64`, or `i32` when context requires it and the value fits |
| Float | `1.0`, `2.5` | `f64`, or `f32` when context requires it |
| Boolean | `true`, `false` | `bool` |
| String | `"hi"` | `String` |

Integer literals have no sign and no underscores. A leading `-` is unary negation. The digits must fit in `i64`. In an `i32` context the digits must fit in `i32` before negation is applied, so `2147483648` is rejected even when written as `-2147483648`. Negating a value that is already the `i32` or `i64` minimum overflows at run time.

Float literals have digits on both sides of the dot. There is no exponent form.

A string is a double quote, then any characters except a raw `"`, then a closing quote. Backslash is an ordinary character. The quotes are not part of the value. A future revision will interpret `\\`, `\"`, `\n`, `\t`, and `\r`.

## 2. Types

The current types are:

| Type | Meaning |
| --- | --- |
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `bool` | `true` or `false` |
| `String` | text |
| function | parameters and a return type, written only by a `func` declaration |
| `()` | unit, the type of a function with no declared return and of `print` |

In a type annotation, `String`, `string`, and `str` are the same type. There is no function-type syntax, so a parameter or a `let` annotation cannot name a function type. The type of a function name is still a function type, and `let g = f` copies that type.

No two numeric types are the same. `i32` and `i64` do not mix. `f32` and `f64` do not mix. Integers and floats do not mix. An unsuffixed literal is the exception: it takes an annotation, a function parameter, or the other operand's numeric type when the value fits. Otherwise an integer literal is `i64` and a float literal is `f64`.

A future revision may add widening only: `i32` to `i64`, `f32` to `f64`, and an integer to a float when the other operand is a float. Narrowing will stay illegal. That rule is not implemented. Today a non-literal `i32` added to an `i64` is a type error.

## 3. Names and scopes

A name is declared by `let` or `func`, or as a function parameter. Declaring a name that already exists in the same scope is an error. An inner scope may hide an outer name.

Lookup walks from the innermost scope outward.

| Region | Scope |
| --- | --- |
| Top level and REPL | one global scope |
| Function body | one scope holding the parameters and the body |
| `if` branch, `else` branch, `while` body | a fresh scope for that block |

A `let` in a branch that does not run is not visible after the branch. Two branches may each declare the same name. Assignment (`=`) updates a name in an enclosing scope. It does not declare a name.

Names are immutable unless declared with `let mut`. Assignment to an immutable name is a type error. Assignment must have the variable's type.

A function captures the environment that exists when the function is defined. The capture is a snapshot of values. Later assignment to an outer `mut` variable is not visible inside the function. A function does not see functions defined later. A function does see itself, so direct recursion works. Mutual recursion does not.

A future revision will put functions defined in the same scope in view of each other, and will make a captured `mut` variable observe later assignments. That replaces snapshot capture. Until then, the snapshot rules above are the language.

## 4. Statements

### 4.1 `let`

```text
let name = expression
let mut name = expression
let name: Type = expression
let mut name: Type = expression
```

The name is bound in the current scope. With an annotation, the expression must have that type. Without one, the expression's type is the variable's type.

### 4.2 Assignment

```text
name = expression
```

`name` must already be a `mut` variable. The expression must have that variable's type.

### 4.3 Expression statement

An expression used as a statement is evaluated. At the top level of the REPL, its value is printed, except `()`. Inside a block, the value is kept only when that expression is the final statement of a function body. A following statement discards it. It does not return from the function.

### 4.4 `if`

```text
if expression block
if expression block else block
if expression block else if expression block
if expression block else if expression block else block
```

The condition must be `bool`. `else if` is an `else` whose block contains the next `if`.

`if` is a statement. It does not yield a value. A bare expression in a branch is not a function result. Only `return` leaves the function from a branch.

A future revision will also allow `if` as an expression when both the `then` block and the `else` block produce the same type. The statement form will stay.

### 4.5 `while`

```text
while expression block
```

The condition must be `bool`. It is tested before every iteration. The body is a new scope each iteration.

`break` ends the innermost `while`. `continue` skips the rest of that body and tests the condition again. `break` and `continue` are errors outside a `while`. They do not cross a function call. They do not carry a value.

A future revision will add:

```text
for name in expression .. expression block
```

The bounds are the same integer type. The range is half-open. `name` is immutable and local to the body. `break` and `continue` apply to that `for` as they do to `while`.

### 4.6 `return`

```text
return
return expression
```

`return` leaves the enclosing function. A bare `return` has type `()`. The expression's type must match the function's return type.

At the top level there is no function return type to check. `return` there yields its value to the REPL.

### 4.7 Blocks

```text
{ statement ... }
{ statement ... expression }
```

A block is a function body, an `if` branch, or a `while` body. It is not a free expression.

When the block is a function body, the result is:

1. the value of a `return`, if one runs, or
2. the final expression, if the block ends in an expression, or
3. `()` if control falls off the end.

A function whose declared return type is not `()` must return on every path. A path returns when it reaches `return`, when both branches of an `if` / `else` return, or when the function body ends in an expression of the return type. A `while` does not count as a guaranteed return. A unit function may fall off the end.

## 5. Functions

```text
func name() block
func name() -> Type block
func name(param: Type, ...) -> Type block
```

Omitting `-> Type` means the function returns `()`. Parameters are immutable. Duplicate parameter names are an error. The function name is in scope for the whole body, including before later statements in that body, so the function may call itself.

A call is `name(argument, ...)`. The name must be a function or `print`. The argument count must match. Each argument must have the parameter's type. Arguments are evaluated left to right.

The call creates a scope, binds the callee under its name, then binds the parameters. Parameter names hide the callee name if they collide. The body then runs. `break` or `continue` that escapes a function is a run-time error. The type checker rejects that program first.

Function values are not comparable. There is no literal for an anonymous function. A future revision may add anonymous functions. It is not designed here beyond the closure rules in section 3.

## 6. Expressions

Precedence, tightest first:

1. unary `-` and `!`, and calls, literals, names, parentheses
2. `*`, `/`, `%`, left associative
3. `+`, `-`, left associative
4. `==`, `!=`, `<`, `<=`, `>`, `>=`, left associative

`a < b < c` is `(a < b) < c`. The comparison yields `bool`, so the next comparison is a type error. Comparisons do not chain.

Parentheses group expressions. There is no assignment expression.

### 6.1 Operators

Both operands of a binary operator must already have the same type, after literal adaptation.

| Operator | Types | Result |
| --- | --- | --- |
| `+` | `i32`, `i64`, `f32`, `f64` | same numeric type |
| `+` | `String` | `String`, concatenation |
| `-`, `*`, `/`, `%` | `i32`, `i64`, `f32`, `f64` | same numeric type |
| `==`, `!=` | numbers, `bool`, `String` | `bool` |
| `<`, `<=`, `>`, `>=` | numbers | `bool` |
| unary `-` | numbers | same numeric type |
| unary `!` | `bool` | `bool` |

Any other combination is a type error. In particular, `bool` has no ordering and no arithmetic, and `String` has no ordering and no arithmetic except `+`.

A future revision will add `&&` and `||` below comparisons. Both operands will be `bool`. Evaluation will be short-circuit: the right operand runs only when the left does not decide the result. They are not in the language today.

### 6.2 Arithmetic

Integer `+`, `-`, `*`, `/`, `%`, and unary `-` fail at run time on overflow instead of wrapping. Integer `/` and `%` fail when the right operand is zero. Dividing or remaindering the minimum integer by `-1` is overflow.

Float operations use IEEE-754. Division by zero is not a Roast error.

## 7. Builtins

`print(value)` accepts one argument of any type, writes one line of text, and returns `()`. The text is the value as defined in section 9. A user function named `print` is rejected.

A future standard library, not implemented, is expected to add:

| Call | Contract |
| --- | --- |
| `len(s: String) -> i64` | number of bytes in the string |
| `assert(cond: bool)` | return `()` when `cond` is true, otherwise fail at run time |

## 8. Evaluation and errors

Execution is a tree walk. There is no separate bytecode.

Parse errors report a source location. Type errors and run-time errors are messages without a source span. A future revision will attach a line and column to type and run-time errors, using the same locations the parser already has.

The type checker runs before execution of an input. A program that fails to type-check is not run.

Integer range and operator rules are type errors when the checker can see them. Overflow and division by zero are run-time errors.

Calling a name that is not a function, or using a name that was never given a value, is a run-time error. The type checker rejects a missing name. A value can still be missing if execution stops after the name was typed and before it was bound. The REPL avoids that for a failed input. See section 9.

## 9. Printing values

| Value | Text |
| --- | --- |
| `i32`, `i64` | decimal, with a leading `-` when negative |
| `f32`, `f64` | decimal |
| `bool` | `true` or `false` |
| `String` | the characters, with no added quotes |
| `()` | `()` |
| function | `func` |

The REPL prints the value of a top-level expression. It does not print `()`. `print` writes the text above and does not cause a second `()` line.

`:env` prints the interpreter's environment for debugging. That dump is not the language's value syntax.

## 10. Tools

### 10.1 Running a file

```text
roast path
```

`path` is one source file. Roast reads it, type-checks the whole file, then runs it. Success exits `0`. A usage error, a missing file, a parse error, a type error, or a run-time error prints to standard error and exits `1`. Standard output is whatever `print` writes. The REPL does not start. More than one argument is a usage error.

### 10.2 REPL

With no arguments, Roast starts an interactive session. The session keeps names and values until `:reset` or exit.

Input may span lines. A line is complete when quotes are closed and `{` braces are balanced. Semicolons are not required.

Editing keys are the arrows, Home, End, Delete, and Backspace.

| Command | Effect |
| --- | --- |
| `:help` | list commands |
| `:exit`, `:quit` | leave |
| `:reset` | drop every name and value |
| `:env` | show the value environment |
| `:history` | show inputs from this session |

`exit` and `quit` without a colon also leave.

One submission is type-checked as a whole. If that fails, the session is unchanged. If it type-checks and a later statement fails at run time, earlier statements from that submission stay, and the failed statement and anything after it are discarded.

A future revision will add:

| Command | Effect |
| --- | --- |
| `:save path` | write this session's successful inputs to `path` |
| `:load path` | run `path` in the current session, as if the text had been submitted |

`:load` will stop on the first error and will not reset names that already existed.

## 11. What this spec does not promise

These are absent on purpose. They are not scheduled by the rules above.

- A package or module system, and `import`.
- A compiler that writes an executable. The file runner interprets one file.
- Generics, arrays, tuples, records, and user-defined types.
- Pattern matching, `switch`, and labeled `break`.
- Methods and operator overloading.

## 12. Current grammar

The normative syntax is `src/grammar/roast.pest`. The summary below matches that file. If they disagree, the grammar file wins.

```text
program    = statement*
statement  = let | assignment | func | return
           | if | while | break | continue
           | expression optional-semicolon
let        = "let" "mut"? identifier (":" type)? "=" expression ";"?
assignment = identifier "=" expression ";"?
func       = "func" identifier "(" parameters? ")" ("->" type)? block
return     = "return" expression? ";"?
if         = "if" expression block ("else" (if | block))?
while      = "while" expression block
break      = "break" ";"?
continue   = "continue" ";"?
block      = "{" statement* expression? "}"
type       = "i32" | "i64" | "f32" | "f64" | "bool" | "String" | "str"
```

Expression precedence is comparison, then addition, then multiplication, then unary, then calls, names, literals, and parentheses.
