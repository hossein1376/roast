use std::{collections::HashMap, error::Error};

use ast::*;
use parser::*;
use typed_ast::*;

mod ast;
mod parser;
mod typed_ast;

fn main() -> Result<(), Box<dyn Error>> {
    let source = r#"
        let x = 5;
        let y = 10;

        fn add(a: i64, b: i64) -> i64 {
            return a + b;
        }

        let result = add(x, y);
    "#;

    // Parse
    let program = parse_program(source)?;

    // Type check
    let typed_program = analyze_program(program).unwrap();

    // Execute
    execute_program(&typed_program)?;

    println!("Program executed successfully!");

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
            eval_expression(expr, env)?;
            Ok(None)
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
