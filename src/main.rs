use std::{collections::HashMap, error::Error};

use ast::*;
use parser::*;
use typed_ast::*;

mod ast;
mod parser;
mod typed_ast;

fn main() -> Result<(), Box<dyn Error>> {
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
    // - and (ends with ';' or ends with '}').
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
        if brace_depth > 0 {
            return false;
        }

        let trimmed = s.trim_end();
        if trimmed.is_empty() {
            return false;
        }
        let ends_with_semicolon = trimmed.ends_with(';');
        let ends_with_close_brace = trimmed.ends_with('}');

        brace_depth == 0 && (ends_with_semicolon || ends_with_close_brace)
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

        // If the input didn't end with a semicolon but looks like a standalone expression, add one
        let program_str = if trimmed.ends_with(';')
            || trimmed.starts_with("fn")
            || trimmed.starts_with("let")
            || trimmed.starts_with("return")
        {
            input_line.clone()
        } else {
            format!("{};", input_line)
        };

        match parse_program(&program_str) {
            Ok(program) => {
                match repl.feed_program(program) {
                    Ok(typed_prog) => {
                        // Execute typed statements using the persistent environment.
                        for stmt in &typed_prog.statements {
                            match exec_statement(stmt, &mut env) {
                                Ok(Some(val)) => {
                                    if let Err(err) = print_safe(&format!("{:?}", val)) {
                                        eprintln!("Printing error: {}", err);
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    if let Err(err) = print_safe(&format!("Runtime error: {}", e)) {
                                        eprintln!("Printing error: {}", err);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if let Err(err) = print_safe(&format!("Type error: {}", e.msg)) {
                            eprintln!("Printing error: {}", err);
                        }
                    }
                }
            }
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
            Type::I32 => Ok(Value::I32(*value as i32)),
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
            // Get the function
            let func_val = env.get(name)?;

            match func_val {
                Value::Function {
                    params,
                    body,
                    closure,
                } => {
                    // Evaluate arguments
                    let mut arg_values = Vec::new();
                    for arg in arguments {
                        arg_values.push(eval_expression(arg, env)?);
                    }

                    // Check argument count
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

                    // Create new environment with closure
                    let mut call_env = closure;
                    call_env.push_scope();

                    // Bind parameters
                    for (param, value) in params.iter().zip(arg_values) {
                        call_env.define(param.name.clone(), value);
                    }

                    // Execute function body
                    let result = eval_block(&body, &mut call_env)?;

                    call_env.pop_scope();

                    Ok(result)
                }
                _ => Err(RuntimeError {
                    msg: format!("'{}' is not a function", name),
                }),
            }
        }
    }
}

fn eval_binary_op(left: &Value, op: &BinaryOperator, right: &Value) -> Result<Value, RuntimeError> {
    match (left, op, right) {
        // Integer arithmetic (I32)
        (Value::I32(l), BinaryOperator::Add, Value::I32(r)) => Ok(Value::I32(l + r)),
        (Value::I32(l), BinaryOperator::Subtract, Value::I32(r)) => Ok(Value::I32(l - r)),
        (Value::I32(l), BinaryOperator::Multiply, Value::I32(r)) => Ok(Value::I32(l * r)),
        (Value::I32(l), BinaryOperator::Divide, Value::I32(r)) => {
            if *r == 0 {
                return Err(RuntimeError {
                    msg: "Division by zero".to_string(),
                });
            }
            Ok(Value::I32(l / r))
        }
        (Value::I32(l), BinaryOperator::Modulo, Value::I32(r)) => Ok(Value::I32(l % r)),

        // Integer arithmetic (I64)
        (Value::I64(l), BinaryOperator::Add, Value::I64(r)) => Ok(Value::I64(l + r)),
        (Value::I64(l), BinaryOperator::Subtract, Value::I64(r)) => Ok(Value::I64(l - r)),
        (Value::I64(l), BinaryOperator::Multiply, Value::I64(r)) => Ok(Value::I64(l * r)),
        (Value::I64(l), BinaryOperator::Divide, Value::I64(r)) => {
            if *r == 0 {
                return Err(RuntimeError {
                    msg: "Division by zero".to_string(),
                });
            }
            Ok(Value::I64(l / r))
        }
        (Value::I64(l), BinaryOperator::Modulo, Value::I64(r)) => Ok(Value::I64(l % r)),

        // Float arithmetic (F64)
        (Value::F64(l), BinaryOperator::Add, Value::F64(r)) => Ok(Value::F64(l + r)),
        (Value::F64(l), BinaryOperator::Subtract, Value::F64(r)) => Ok(Value::F64(l - r)),
        (Value::F64(l), BinaryOperator::Multiply, Value::F64(r)) => Ok(Value::F64(l * r)),
        (Value::F64(l), BinaryOperator::Divide, Value::F64(r)) => Ok(Value::F64(l / r)),

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
        (UnaryOperator::Negate, Value::I32(v)) => Ok(Value::I32(-v)),
        (UnaryOperator::Negate, Value::I64(v)) => Ok(Value::I64(-v)),
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
) -> Result<Option<Value>, RuntimeError> {
    match stmt {
        TypedStatement::Let { name, value, .. } => {
            let val = eval_expression(value, env)?;
            env.define(name.clone(), val);
            Ok(None)
        }

        TypedStatement::Assignment { name, value } => {
            let val = eval_expression(value, env)?;
            env.set(name, val)?;
            Ok(None)
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
            Ok(None)
        }

        TypedStatement::Return { value } => {
            let val = if let Some(expr) = value {
                eval_expression(expr, env)?
            } else {
                Value::Unit
            };
            Ok(Some(val))
        }

        TypedStatement::Expression(expr) => {
            // For expression statements return the evaluated value so the REPL can print it.
            let v = eval_expression(expr, env)?;
            Ok(Some(v))
        }
    }
}

pub fn eval_block(block: &TypedBlock, env: &mut Environment) -> Result<Value, RuntimeError> {
    // Execute all statements
    for stmt in &block.statements {
        if let Some(return_val) = exec_statement(stmt, env)? {
            return Ok(return_val);
        }
    }

    // Evaluate return expression if present
    if let Some(expr) = &block.return_expression {
        eval_expression(expr, env)
    } else {
        Ok(Value::Unit)
    }
}

pub fn execute_program(program: &TypedProgram) -> Result<(), RuntimeError> {
    let mut env = Environment::new();

    for stmt in &program.statements {
        exec_statement(stmt, &mut env)?;
    }

    println!("{:#?}", env);

    Ok(())
}
