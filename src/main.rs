use std::{collections::HashMap, error::Error};

use ast::*;
use parser::*;
use typed_ast::*;

mod ast;
mod parser;
mod typed_ast;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    if let Some(path) = args.next() {
        if args.next().is_some() {
            eprintln!("usage: roast [file]");
            std::process::exit(1);
        }
        return run_file(&path);
    }
    repl()
}

fn run_file(path: &str) -> Result<(), Box<dyn Error>> {
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("{path}: {err}");
            std::process::exit(1);
        }
    };
    if let Err(err) = execute_source(&source) {
        eprintln!("{err}");
        std::process::exit(1);
    }
    Ok(())
}

/// Type-check and run one REPL input.
///
/// A type error leaves the previous REPL state unchanged. A runtime error
/// keeps earlier statements from that input and drops the failed one and
/// anything after it.
fn execute_repl_program(
    repl: &mut Repl,
    env: &mut Environment,
    program: Program,
) -> Result<Vec<Value>, Box<dyn Error>> {
    let source_statements = program.statements.clone();
    let saved = repl.symbols.clone();
    let typed = repl.feed_program(program)?;
    let mut values = Vec::new();
    for (index, stmt) in typed.statements.iter().enumerate() {
        match exec_statement(stmt, env) {
            Ok(Flow::Value(value) | Flow::Return(value)) => values.push(value),
            Ok(Flow::Next) => {}
            Err(err) => {
                repl.symbols = saved.clone();
                if let Err(type_err) = repl.replay(&source_statements[..index]) {
                    repl.symbols = saved;
                    return Err(type_err.into());
                }
                return Err(err.into());
            }
        }
    }
    Ok(values)
}

/// Parse, type-check, and run a program. The result is the last expression
/// value, or `Unit` when the program does not end in one.
pub fn execute_source(source: &str) -> Result<Value, Box<dyn Error>> {
    let program = parse_program(source)?;
    let mut repl_state = Repl::new();
    let typed = repl_state.feed_program(program)?;
    let mut env = Environment::new();
    let mut last = Value::Unit;
    for stmt in &typed.statements {
        last = match exec_statement(stmt, &mut env)? {
            Flow::Value(value) | Flow::Return(value) => value,
            Flow::Next => Value::Unit,
        };
    }
    Ok(last)
}

fn repl() -> Result<(), Box<dyn Error>> {
    use std::io::{self, Write};

    // REPL keeps a persistent type symbol table and runtime environment.
    let mut repl = Repl::new();
    let mut env = Environment::new();

    println!("Roast REPL. Type ':help' for commands. Type ':exit' or 'exit' to leave.");

    // We'll use crossterm to capture key events and handle editing (arrow keys, home/end/delete)
    use crossterm::{
        cursor::MoveToColumn,
        event::{Event, KeyCode, KeyModifiers, poll, read},
        execute,
        terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
    };
    use std::time::Duration;

    enable_raw_mode()?;
    // If evaluation panics, raw mode is still cleared.
    struct RawModeGuard;
    impl Drop for RawModeGuard {
        fn drop(&mut self) {
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
    let _raw_mode = RawModeGuard;

    // Helper to print multi-line output safely by temporarily disabling raw mode.
    // This clears the current line and moves the cursor to column 0 before printing
    // so diagnostics (multi-line output with carets) don't accumulate indentation.
    fn print_safe(s: &str) -> Result<(), Box<dyn Error>> {
        // Disable raw mode so printed diagnostics behave normally.
        disable_raw_mode()?;

        // Move to start of current line and clear it to avoid growing indentation
        execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
        println!("{}", s);
        io::stdout().flush()?;

        // Re-enable raw mode for continued interactive input.
        enable_raw_mode()?;
        Ok(())
    }

    // Helper: determine whether the current buffer is a complete program/statement.
    // We'll consider the input complete if:
    // - braces are balanced ({}), and
    // - not inside an unclosed double-quote string, and
    // - the trimmed input is non-empty (semicolons are optional).
    fn is_input_complete(s: &str) -> bool {
        // Track braces and quotes. This is a heuristic sufficient for REPL convenience.
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut escape = false;

        for ch in s.chars() {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            match ch {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }
        }

        if in_string {
            return false;
        }
        // If braces are still unbalanced (more opens than closes), it's not complete.
        if brace_depth > 0 {
            return false;
        }

        let trimmed = s.trim_end();
        if trimmed.is_empty() {
            return false;
        }

        // Consider input complete as long as we're not inside a string and braces are balanced.
        true
    }

    // History: simple in-memory vector
    let mut history: Vec<String> = Vec::new();
    let mut history_pos: Option<usize> = None;

    loop {
        // Line buffer and cursor position (in characters)
        let mut buffer = String::new();
        let mut cursor_pos: usize = 0;
        let mut prompt = "> ".to_string();

        // For multi-line inputs we will accumulate into `full_input`.
        let mut full_input = String::new();

        // Initial prompt: clear line and print prompt at column 0
        execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
        print!("{}", prompt);
        io::stdout().flush()?;

        // Input loop: handle key events until Enter confirms a complete statement
        loop {
            // Wait for an event (timeout so program remains responsive)
            if !poll(Duration::from_millis(100))? {
                continue;
            }

            match read()? {
                Event::Key(key_event) => {
                    match key_event.code {
                        // Ctrl-C: exit REPL
                        KeyCode::Char('c')
                            if key_event.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            disable_raw_mode()?;
                            println!();
                            return Ok(());
                        }

                        KeyCode::Left => {
                            if cursor_pos > 0 {
                                cursor_pos -= 1;
                            }
                            // re-draw
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Right => {
                            if cursor_pos < buffer.chars().count() {
                                cursor_pos += 1;
                            }
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Home => {
                            cursor_pos = 0;
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::End => {
                            cursor_pos = buffer.chars().count();
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Backspace => {
                            if cursor_pos > 0 {
                                // Remove character before cursor_pos
                                let mut chars: Vec<char> = buffer.chars().collect();
                                chars.remove(cursor_pos - 1);
                                buffer = chars.iter().collect();
                                cursor_pos -= 1;
                            }
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Delete => {
                            let len = buffer.chars().count();
                            if cursor_pos < len {
                                let mut chars: Vec<char> = buffer.chars().collect();
                                chars.remove(cursor_pos);
                                buffer = chars.iter().collect();
                            }
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Up => {
                            if history.is_empty() {
                                // nothing
                            } else {
                                if let Some(pos) = history_pos {
                                    if pos > 0 {
                                        history_pos = Some(pos - 1);
                                    }
                                } else {
                                    history_pos = Some(history.len() - 1);
                                }
                                if let Some(pos) = history_pos {
                                    buffer = history[pos].clone();
                                    cursor_pos = buffer.chars().count();
                                }
                            }
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Down => {
                            if history.is_empty() {
                                // nothing
                            } else {
                                if let Some(pos) = history_pos {
                                    if pos + 1 < history.len() {
                                        history_pos = Some(pos + 1);
                                        buffer = history[history_pos.unwrap()].clone();
                                    } else {
                                        history_pos = None;
                                        buffer.clear();
                                    }
                                }
                                // if None do nothing
                                cursor_pos = buffer.chars().count();
                            }
                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Char(ch) => {
                            // Insert character at cursor position
                            let mut chars: Vec<char> = buffer.chars().collect();
                            chars.insert(cursor_pos, ch);
                            buffer = chars.iter().collect();
                            cursor_pos += 1;

                            execute!(io::stdout(), Clear(ClearType::CurrentLine), MoveToColumn(0))?;
                            print!("{}{}", prompt, buffer);
                            let col = (prompt.len() + cursor_pos) as u16;
                            execute!(io::stdout(), MoveToColumn(col))?;
                            io::stdout().flush()?;
                        }

                        KeyCode::Enter => {
                            // If current full_input + buffer is a complete program, finish and process.
                            let candidate = if full_input.is_empty() {
                                buffer.clone()
                            } else {
                                format!("{}\\n{}", full_input, buffer)
                            };

                            // If this is a REPL command (starts with ':'), treat it as complete immediately.
                            if candidate.trim_start().starts_with(':') {
                                full_input = candidate;
                                println!();
                                break;
                            }

                            if is_input_complete(&candidate) {
                                // finalize
                                full_input = candidate;
                                println!();
                                break;
                            } else {
                                // continue multi-line input
                                full_input = candidate;
                                // append newline to display and clear buffer for next line
                                print!("\n");
                                prompt = "... ".to_string();
                                buffer.clear();
                                cursor_pos = 0;
                                history_pos = None;
                                io::stdout().flush()?;
                            }
                        }

                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Completed (multi-line) input is in full_input. If user never accumulated multi-line,
        // full_input may be empty and buffer contains the single-line; handle accordingly.
        let input_line = if full_input.is_empty() {
            buffer.clone()
        } else {
            full_input.clone()
        };
        let trimmed = input_line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // Commands start with ':' (also accept bare 'exit'/'quit')
        if trimmed.starts_with(':') {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            match parts[0] {
                ":exit" | ":quit" => break,
                ":reset" => {
                    repl.reset();
                    env = Environment::new();
                    print_safe("State cleared")?;
                    continue;
                }
                ":env" => {
                    print_safe(&format!("{:#?}", env))?;
                    continue;
                }
                ":help" => {
                    print_safe(
                        "Commands: :help, :reset, :env, :history, :exit. Editing: arrows, Home, End, Delete.",
                    )?;
                    continue;
                }
                ":history" => {
                    if history.is_empty() {
                        print_safe("History is empty")?;
                    } else {
                        let mut out = String::new();
                        for (i, entry) in history.iter().enumerate() {
                            out.push_str(&format!("{}: {}\n", i, entry));
                        }
                        print_safe(&out)?;
                    }
                    continue;
                }
                _ => {
                    print_safe(&format!("Unknown command: {}", parts[0]))?;
                    continue;
                }
            }
        }
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }

        // Save to history (don't store duplicate consecutive identical entries)
        if history.last().map_or(true, |last| last != &input_line) {
            history.push(input_line.clone());
        }
        history_pos = None;

        // Previously we auto-appended a semicolon to standalone expressions.
        // Semicolons are now optional in the grammar/REPL — pass the input through as-is.
        let program_str = input_line.clone();

        match parse_program(&program_str) {
            Ok(program) => match execute_repl_program(&mut repl, &mut env, program) {
                Ok(values) => {
                    for val in values {
                        if let Err(err) = print_safe(&format!("{:?}", val)) {
                            eprintln!("Printing error: {}", err);
                        }
                    }
                }
                Err(err) => {
                    if let Err(print_err) = print_safe(&err.to_string()) {
                        eprintln!("Printing error: {}", print_err);
                    }
                }
            },
            Err(e) => {
                if let Err(err) = print_safe(&format!("Parse error: {}", e)) {
                    eprintln!("Printing error: {}", err);
                }
            }
        }
    }

    disable_raw_mode()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum Flow {
    /// Statement finished. Keep going.
    Next,
    /// Expression statement. The REPL prints it. A block does not return it
    /// unless it is the block's trailing value.
    Value(Value),
    /// A `return` statement. Blocks, `if`, and `while` propagate it.
    Return(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Unit,
    Function {
        params: Vec<Parameter>,
        body: TypedBlock,
        closure: Environment, // Captured environment
    },
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::I32(_) => Type::I32,
            Value::I64(_) => Type::I64,
            Value::F32(_) => Type::F32,
            Value::F64(_) => Type::F64,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Unit => Type::Unit,
            Value::Function { .. } => Type::Unit, // Simplified
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    scopes: Vec<HashMap<String, Value>>,
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn define(&mut self, name: String, value: Value) {
        self.scopes.last_mut().unwrap().insert(name, value);
    }

    pub fn get(&self, name: &str) -> Result<Value, RuntimeError> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Ok(value.clone());
            }
        }
        Err(RuntimeError {
            msg: format!("Undefined variable '{}'", name),
        })
    }

    pub fn set(&mut self, name: &str, value: Value) -> Result<(), RuntimeError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return Ok(());
            }
        }
        Err(RuntimeError {
            msg: format!("Undefined variable '{}'", name),
        })
    }
}

#[derive(Debug)]
pub struct RuntimeError {
    pub msg: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Runtime error: {}", self.msg)
    }
}

impl std::error::Error for RuntimeError {}

pub fn eval_expression(
    expr: &TypedExpression,
    env: &mut Environment,
) -> Result<Value, RuntimeError> {
    match expr {
        TypedExpression::IntegerLiteral { value, expr_type } => match expr_type {
            Type::I32 => i32::try_from(*value).map(Value::I32).map_err(|_| RuntimeError {
                msg: format!("integer literal {value} does not fit in i32"),
            }),
            Type::I64 => Ok(Value::I64(*value)),
            _ => Err(RuntimeError {
                msg: format!("Invalid integer type: {:?}", expr_type),
            }),
        },

        TypedExpression::FloatLiteral { value, expr_type } => match expr_type {
            Type::F32 => Ok(Value::F32(*value as f32)),
            Type::F64 => Ok(Value::F64(*value)),
            _ => Err(RuntimeError {
                msg: format!("Invalid float type: {:?}", expr_type),
            }),
        },

        TypedExpression::BooleanLiteral { value, .. } => Ok(Value::Bool(*value)),

        TypedExpression::StringLiteral { value, .. } => Ok(Value::String(value.clone())),

        TypedExpression::Identifier { name, .. } => env.get(name),

        TypedExpression::Binary {
            left,
            operator,
            right,
            ..
        } => {
            let left_val = eval_expression(left, env)?;
            let right_val = eval_expression(right, env)?;
            eval_binary_op(&left_val, operator, &right_val)
        }

        TypedExpression::Unary {
            operator, operand, ..
        } => {
            let operand_val = eval_expression(operand, env)?;
            eval_unary_op(operator, &operand_val)
        }

        TypedExpression::FunctionCall {
            name, arguments, ..
        } => {
            if name == "print" {
                if arguments.len() != 1 {
                    return Err(RuntimeError {
                        msg: format!("print expects 1 argument, got {}", arguments.len()),
                    });
                }
                let value = eval_expression(&arguments[0], env)?;
                println!("{value:?}");
                return Ok(Value::Unit);
            }

            let func_val = env.get(name)?;
            let Value::Function {
                params,
                body,
                closure,
            } = &func_val
            else {
                return Err(RuntimeError {
                    msg: format!("'{}' is not a function", name),
                });
            };

            let mut arg_values = Vec::new();
            for arg in arguments {
                arg_values.push(eval_expression(arg, env)?);
            }

            if arg_values.len() != params.len() {
                return Err(RuntimeError {
                    msg: format!(
                        "Function '{}' expects {} arguments, got {}",
                        name,
                        params.len(),
                        arg_values.len()
                    ),
                });
            }

            // Bind the callee in the call scope so direct recursion resolves.
            // The closure stays the snapshot from definition time.
            let mut call_env = closure.clone();
            call_env.push_scope();
            call_env.define(name.clone(), func_val.clone());

            for (param, value) in params.iter().zip(arg_values) {
                call_env.define(param.name.clone(), value);
            }

            let result = match eval_block(body, &mut call_env)? {
                Flow::Return(value) | Flow::Value(value) => value,
                Flow::Next => Value::Unit,
            };
            Ok(result)
        }
    }
}

fn int_arith_msg(op: &BinaryOperator, rhs_is_zero: bool) -> String {
    match op {
        BinaryOperator::Divide if rhs_is_zero => "Division by zero".to_string(),
        BinaryOperator::Modulo if rhs_is_zero => "Modulo by zero".to_string(),
        _ => "Integer overflow".to_string(),
    }
}

fn checked_i32(left: i32, op: &BinaryOperator, right: i32) -> Result<i32, RuntimeError> {
    let value = match op {
        BinaryOperator::Add => left.checked_add(right),
        BinaryOperator::Subtract => left.checked_sub(right),
        BinaryOperator::Multiply => left.checked_mul(right),
        BinaryOperator::Divide => left.checked_div(right),
        BinaryOperator::Modulo => left.checked_rem(right),
        _ => None,
    };
    value.ok_or_else(|| RuntimeError {
        msg: int_arith_msg(op, right == 0),
    })
}

fn checked_i64(left: i64, op: &BinaryOperator, right: i64) -> Result<i64, RuntimeError> {
    let value = match op {
        BinaryOperator::Add => left.checked_add(right),
        BinaryOperator::Subtract => left.checked_sub(right),
        BinaryOperator::Multiply => left.checked_mul(right),
        BinaryOperator::Divide => left.checked_div(right),
        BinaryOperator::Modulo => left.checked_rem(right),
        _ => None,
    };
    value.ok_or_else(|| RuntimeError {
        msg: int_arith_msg(op, right == 0),
    })
}

fn eval_binary_op(left: &Value, op: &BinaryOperator, right: &Value) -> Result<Value, RuntimeError> {
    match (left, op, right) {
        (Value::I32(l), op, Value::I32(r))
            if matches!(
                op,
                BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo
            ) =>
        {
            Ok(Value::I32(checked_i32(*l, op, *r)?))
        }
        (Value::I64(l), op, Value::I64(r))
            if matches!(
                op,
                BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo
            ) =>
        {
            Ok(Value::I64(checked_i64(*l, op, *r)?))
        }

        // Float arithmetic (F32)
        (Value::F32(l), BinaryOperator::Add, Value::F32(r)) => Ok(Value::F32(l + r)),
        (Value::F32(l), BinaryOperator::Subtract, Value::F32(r)) => Ok(Value::F32(l - r)),
        (Value::F32(l), BinaryOperator::Multiply, Value::F32(r)) => Ok(Value::F32(l * r)),
        (Value::F32(l), BinaryOperator::Divide, Value::F32(r)) => Ok(Value::F32(l / r)),
        (Value::F32(l), BinaryOperator::Modulo, Value::F32(r)) => Ok(Value::F32(l % r)),

        // Float arithmetic (F64)
        (Value::F64(l), BinaryOperator::Add, Value::F64(r)) => Ok(Value::F64(l + r)),
        (Value::F64(l), BinaryOperator::Subtract, Value::F64(r)) => Ok(Value::F64(l - r)),
        (Value::F64(l), BinaryOperator::Multiply, Value::F64(r)) => Ok(Value::F64(l * r)),
        (Value::F64(l), BinaryOperator::Divide, Value::F64(r)) => Ok(Value::F64(l / r)),
        (Value::F64(l), BinaryOperator::Modulo, Value::F64(r)) => Ok(Value::F64(l % r)),

        // Comparisons (F32)
        (Value::F32(l), BinaryOperator::Equal, Value::F32(r)) => Ok(Value::Bool(l == r)),
        (Value::F32(l), BinaryOperator::NotEqual, Value::F32(r)) => Ok(Value::Bool(l != r)),
        (Value::F32(l), BinaryOperator::LessThan, Value::F32(r)) => Ok(Value::Bool(l < r)),
        (Value::F32(l), BinaryOperator::LessThanOrEqual, Value::F32(r)) => Ok(Value::Bool(l <= r)),
        (Value::F32(l), BinaryOperator::GreaterThan, Value::F32(r)) => Ok(Value::Bool(l > r)),
        (Value::F32(l), BinaryOperator::GreaterThanOrEqual, Value::F32(r)) => {
            Ok(Value::Bool(l >= r))
        }

        (Value::F64(l), BinaryOperator::Equal, Value::F64(r)) => Ok(Value::Bool(l == r)),
        (Value::F64(l), BinaryOperator::NotEqual, Value::F64(r)) => Ok(Value::Bool(l != r)),
        (Value::F64(l), BinaryOperator::LessThan, Value::F64(r)) => Ok(Value::Bool(l < r)),
        (Value::F64(l), BinaryOperator::LessThanOrEqual, Value::F64(r)) => Ok(Value::Bool(l <= r)),
        (Value::F64(l), BinaryOperator::GreaterThan, Value::F64(r)) => Ok(Value::Bool(l > r)),
        (Value::F64(l), BinaryOperator::GreaterThanOrEqual, Value::F64(r)) => {
            Ok(Value::Bool(l >= r))
        }

        // String concatenation
        (Value::String(l), BinaryOperator::Add, Value::String(r)) => {
            Ok(Value::String(format!("{}{}", l, r)))
        }

        // Comparisons (I32)
        (Value::I32(l), BinaryOperator::Equal, Value::I32(r)) => Ok(Value::Bool(l == r)),
        (Value::I32(l), BinaryOperator::NotEqual, Value::I32(r)) => Ok(Value::Bool(l != r)),
        (Value::I32(l), BinaryOperator::LessThan, Value::I32(r)) => Ok(Value::Bool(l < r)),
        (Value::I32(l), BinaryOperator::LessThanOrEqual, Value::I32(r)) => Ok(Value::Bool(l <= r)),
        (Value::I32(l), BinaryOperator::GreaterThan, Value::I32(r)) => Ok(Value::Bool(l > r)),
        (Value::I32(l), BinaryOperator::GreaterThanOrEqual, Value::I32(r)) => {
            Ok(Value::Bool(l >= r))
        }

        // Comparisons (I64)
        (Value::I64(l), BinaryOperator::Equal, Value::I64(r)) => Ok(Value::Bool(l == r)),
        (Value::I64(l), BinaryOperator::NotEqual, Value::I64(r)) => Ok(Value::Bool(l != r)),
        (Value::I64(l), BinaryOperator::LessThan, Value::I64(r)) => Ok(Value::Bool(l < r)),
        (Value::I64(l), BinaryOperator::LessThanOrEqual, Value::I64(r)) => Ok(Value::Bool(l <= r)),
        (Value::I64(l), BinaryOperator::GreaterThan, Value::I64(r)) => Ok(Value::Bool(l > r)),
        (Value::I64(l), BinaryOperator::GreaterThanOrEqual, Value::I64(r)) => {
            Ok(Value::Bool(l >= r))
        }

        // Boolean operations
        (Value::Bool(l), BinaryOperator::Equal, Value::Bool(r)) => Ok(Value::Bool(l == r)),
        (Value::Bool(l), BinaryOperator::NotEqual, Value::Bool(r)) => Ok(Value::Bool(l != r)),

        // String comparisons
        (Value::String(l), BinaryOperator::Equal, Value::String(r)) => Ok(Value::Bool(l == r)),
        (Value::String(l), BinaryOperator::NotEqual, Value::String(r)) => Ok(Value::Bool(l != r)),

        _ => Err(RuntimeError {
            msg: format!("Invalid binary operation: {:?} {:?} {:?}", left, op, right),
        }),
    }
}

fn eval_unary_op(op: &UnaryOperator, operand: &Value) -> Result<Value, RuntimeError> {
    match (op, operand) {
        (UnaryOperator::Negate, Value::I32(v)) => v.checked_neg().map(Value::I32).ok_or_else(|| {
            RuntimeError {
                msg: "Integer overflow".to_string(),
            }
        }),
        (UnaryOperator::Negate, Value::I64(v)) => v.checked_neg().map(Value::I64).ok_or_else(|| {
            RuntimeError {
                msg: "Integer overflow".to_string(),
            }
        }),
        (UnaryOperator::Negate, Value::F32(v)) => Ok(Value::F32(-v)),
        (UnaryOperator::Negate, Value::F64(v)) => Ok(Value::F64(-v)),
        (UnaryOperator::Not, Value::Bool(v)) => Ok(Value::Bool(!v)),
        _ => Err(RuntimeError {
            msg: format!("Invalid unary operation: {:?} {:?}", op, operand),
        }),
    }
}

pub fn exec_statement(
    stmt: &TypedStatement,
    env: &mut Environment,
) -> Result<Flow, RuntimeError> {
    match stmt {
        TypedStatement::Let { name, value, .. } => {
            let val = eval_expression(value, env)?;
            env.define(name.clone(), val);
            Ok(Flow::Next)
        }

        TypedStatement::Assignment { name, value } => {
            let val = eval_expression(value, env)?;
            env.set(name, val)?;
            Ok(Flow::Next)
        }

        TypedStatement::FunctionDeclaration {
            name,
            parameters,
            body,
            ..
        } => {
            let func = Value::Function {
                params: parameters.clone(),
                body: body.clone(),
                closure: env.clone(),
            };
            env.define(name.clone(), func);
            Ok(Flow::Next)
        }

        TypedStatement::Return { value } => {
            let val = if let Some(expr) = value {
                eval_expression(expr, env)?
            } else {
                Value::Unit
            };
            Ok(Flow::Return(val))
        }

        TypedStatement::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let branch = match eval_expression(condition, env)? {
                Value::Bool(true) => Some(then_branch),
                Value::Bool(false) => else_branch.as_ref(),
                other => {
                    return Err(RuntimeError {
                        msg: format!("if condition must be bool, got {:?}", other),
                    });
                }
            };
            if let Some(branch) = branch {
                if let Flow::Return(value) = eval_in_scope(branch, env)? {
                    return Ok(Flow::Return(value));
                }
            }
            Ok(Flow::Next)
        }

        TypedStatement::While { condition, body } => loop {
            match eval_expression(condition, env)? {
                Value::Bool(true) => {}
                Value::Bool(false) => return Ok(Flow::Next),
                other => {
                    return Err(RuntimeError {
                        msg: format!("while condition must be bool, got {:?}", other),
                    });
                }
            }
            if let Flow::Return(value) = eval_in_scope(body, env)? {
                return Ok(Flow::Return(value));
            }
        },

        TypedStatement::Expression(expr) => {
            // The REPL prints this. Inside a block it is a value only when
            // it is the final statement.
            let v = eval_expression(expr, env)?;
            Ok(Flow::Value(v))
        }
    }
}

fn eval_in_scope(block: &TypedBlock, env: &mut Environment) -> Result<Flow, RuntimeError> {
    env.push_scope();
    let result = eval_block(block, env);
    env.pop_scope();
    result
}

pub fn eval_block(block: &TypedBlock, env: &mut Environment) -> Result<Flow, RuntimeError> {
    let mut last = Flow::Next;
    for stmt in &block.statements {
        match exec_statement(stmt, env)? {
            Flow::Return(value) => return Ok(Flow::Return(value)),
            flow @ (Flow::Next | Flow::Value(_)) => last = flow,
        }
    }

    if let Some(expr) = &block.return_expression {
        return Ok(Flow::Value(eval_expression(expr, env)?));
    }
    Ok(last)
}

pub fn execute_program(program: &TypedProgram) -> Result<(), RuntimeError> {
    let mut env = Environment::new();

    for stmt in &program.statements {
        exec_statement(stmt, &mut env)?;
    }

    println!("{:#?}", env);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    #[test]
    fn factorial_matches_readme() {
        let value = execute_source(
            r#"
            func factorial(n: i64) -> i64 {
                if n == 0 {
                    return 1
                } else {
                    return n * factorial(n - 1)
                }
            }
            factorial(5)
            "#,
        )
        .unwrap();
        assert_eq!(value, Value::I64(120));
    }

    #[test]
    fn trailing_expression_is_function_value() {
        let value = execute_source(
            r#"
            func add(a: i64, b: i64) -> i64 { a + b }
            add(2, 3)
            "#,
        )
        .unwrap();
        assert_eq!(value, Value::I64(5));
    }

    #[test]
    fn if_assignment_does_not_return() {
        let value = execute_source(
            r#"
            func f() -> i64 {
                let mut x: i64 = 0
                if true {
                    x = 1
                } else {
                    x = 2
                }
                x
            }
            f()
            "#,
        )
        .unwrap();
        assert_eq!(value, Value::I64(1));
    }

    #[test]
    fn while_sums_zero_through_four() {
        let value = execute_source(
            r#"
            let mut i: i64 = 0
            let mut s: i64 = 0
            while i < 5 {
                s = s + i
                i = i + 1
            }
            s
            "#,
        )
        .unwrap();
        assert_eq!(value, Value::I64(10));
    }

    #[test]
    fn print_returns_unit() {
        let value = execute_source("print(1)").unwrap();
        assert_eq!(value, Value::Unit);
    }

    #[test]
    fn f64_comparison_works() {
        let value = execute_source("let x: f64 = 1.5\nx < 2.0").unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn i32_literal_follows_the_other_operand() {
        let value = execute_source("let mut x: i32 = 1\nx = x + 1\nx").unwrap();
        assert_eq!(value, Value::I32(2));
        let value = execute_source("func inc(n: i32) -> i32 { n + 1 }\ninc(1)").unwrap();
        assert_eq!(value, Value::I32(2));
    }

    #[test]
    fn modulo_by_zero_is_an_error() {
        let err = execute_source("1 % 0").unwrap_err();
        assert!(err.to_string().contains("Modulo by zero"), "{err}");
    }

    #[test]
    fn integer_overflow_is_an_error() {
        let err = execute_source("let x: i64 = 9223372036854775807\nx + 1").unwrap_err();
        assert!(err.to_string().contains("Integer overflow"), "{err}");
    }

    #[test]
    fn branch_locals_can_share_a_name() {
        let value = execute_source(
            r#"
            let mut n: i64 = 0
            if true {
                let x: i64 = 1
                n = x
            } else {
                let x: i64 = 2
                n = x
            }
            n
            "#,
        )
        .unwrap();
        assert_eq!(value, Value::I64(1));
    }

    #[test]
    fn runtime_error_keeps_earlier_statements_only() {
        let mut repl_state = Repl::new();
        let mut env = Environment::new();
        let program = parse_program("let x: i64 = 1\nlet y: i64 = 1 / 0\nlet z: i64 = 3").unwrap();
        assert!(execute_repl_program(&mut repl_state, &mut env, program).is_err());
        assert!(repl_state.symbols.lookup("x").is_ok());
        assert!(repl_state.symbols.lookup("y").is_err());
        assert!(repl_state.symbols.lookup("z").is_err());
        assert_eq!(env.get("x").unwrap(), Value::I64(1));
        assert!(env.get("y").is_err());
    }

    #[test]
    fn type_error_rolls_back_the_whole_input() {
        let mut repl_state = Repl::new();
        let mut env = Environment::new();
        let first = parse_program("let x: i64 = 4").unwrap();
        execute_repl_program(&mut repl_state, &mut env, first).unwrap();
        let bad = parse_program("let y: i64 = true\nlet z: i64 = 5").unwrap();
        assert!(execute_repl_program(&mut repl_state, &mut env, bad).is_err());
        assert!(repl_state.symbols.lookup("x").is_ok());
        assert!(repl_state.symbols.lookup("y").is_err());
        assert!(repl_state.symbols.lookup("z").is_err());
        assert_eq!(env.get("x").unwrap(), Value::I64(4));
    }
}
