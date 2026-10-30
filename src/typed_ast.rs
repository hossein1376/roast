use std::collections::HashMap;

use crate::ast::*;

#[derive(Debug, Clone, PartialEq)]
pub enum TypedExpression {
    Binary {
        left: Box<TypedExpression>,
        operator: BinaryOperator,
        right: Box<TypedExpression>,
        expr_type: Type,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<TypedExpression>,
        expr_type: Type,
    },
    FunctionCall {
        name: String,
        arguments: Vec<TypedExpression>,
        expr_type: Type,
    },
    Identifier {
        name: String,
        expr_type: Type,
        mutable: bool,
    },
    IntegerLiteral {
        value: i64,
        expr_type: Type,
    },
    FloatLiteral {
        value: f64,
        expr_type: Type,
    },
    BooleanLiteral {
        value: bool,
        expr_type: Type,
    },
    StringLiteral {
        value: String,
        expr_type: Type,
    },
}

impl TypedExpression {
    pub fn get_type(&self) -> &Type {
        match self {
            TypedExpression::Binary { expr_type, .. } => expr_type,
            TypedExpression::Identifier { expr_type, .. } => expr_type,
            TypedExpression::IntegerLiteral { expr_type, .. } => expr_type,
            TypedExpression::Unary { expr_type, .. } => expr_type,
            TypedExpression::FunctionCall { expr_type, .. } => expr_type,
            TypedExpression::FloatLiteral { expr_type, .. } => expr_type,
            TypedExpression::BooleanLiteral { expr_type, .. } => expr_type,
            TypedExpression::StringLiteral { expr_type, .. } => expr_type,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedStatement {
    Let {
        mutable: bool,
        name: String,
        type_annotation: Type,
        value: TypedExpression,
    },
    Assignment {
        name: String,
        value: TypedExpression,
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<Parameter>,
        return_type: Type,
        body: TypedBlock,
    },
    Return {
        value: Option<TypedExpression>,
    },
    If {
        condition: TypedExpression,
        then_branch: TypedBlock,
        else_branch: Option<TypedBlock>,
    },
    While {
        condition: TypedExpression,
        body: TypedBlock,
    },
    Expression(TypedExpression),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedBlock {
    pub statements: Vec<TypedStatement>,
    pub return_expression: Option<TypedExpression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
    Bool,
    String,
    Function {
        param_types: Vec<Type>,
        return_type: Box<Type>,
    },
    Unit,
    Unknown,
}

#[derive(Clone)]
pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

#[derive(Clone)]
pub struct Symbol {
    pub symbol_type: Type,
    pub mutable: bool,
}

impl SymbolTable {
    /// Start with an empty global scope
    pub fn new() -> Self {
        SymbolTable {
            scopes: vec![HashMap::new()],
        }
    }

    /// Push a new scope (for blocks, functions, etc.)
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        } else {
            panic!("Cannot pop global scope");
        }
    }

    /// Insert a symbol into the current scope
    pub fn insert(&mut self, name: String, symbol: Symbol) -> Result<(), TypeError> {
        let current_scope = self.scopes.last_mut().unwrap();

        if current_scope.contains_key(&name) {
            return Err(TypeError {
                msg: format!("Symbol '{}' already declared in this scope", name),
            });
        }

        current_scope.insert(name, symbol);
        Ok(())
    }

    /// Look up a symbol, searching from innermost to outermost scope
    pub fn lookup(&self, name: &str) -> Result<&Symbol, TypeError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Ok(symbol);
            }
        }

        Err(TypeError {
            msg: format!("Undefined symbol '{}'", name),
        })
    }
}

#[derive(Debug)]
pub struct TypeError {
    pub msg: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Type error: {}", self.msg)
    }
}

impl std::error::Error for TypeError {}

#[derive(Debug)]
pub struct TypedProgram {
    pub statements: Vec<TypedStatement>,
}

pub fn analyse_expression(
    expr: Expression,
    symbols: &SymbolTable,
    hint: Option<&Type>,
) -> Result<TypedExpression, TypeError> {
    match expr {
        Expression::Binary {
            left,
            operator,
            right,
        } => {
            let operand_hint = arithmetic_hint(hint, &operator);
            let mut typed_left = analyse_expression(*left, symbols, operand_hint)?;
            let mut typed_right = analyse_expression(*right, symbols, operand_hint)?;
            adapt_literal(&mut typed_left, typed_right.get_type())?;
            adapt_literal(&mut typed_right, typed_left.get_type())?;

            let left_type = typed_left.get_type();
            let right_type = typed_right.get_type();
            if left_type != right_type {
                return Err(TypeError {
                    msg: format!(
                        "Type mismatch: {:?} {:?} {:?}",
                        left_type, operator, right_type
                    ),
                });
            }

            let expr_type = binary_result_type(left_type, &operator)?;

            Ok(TypedExpression::Binary {
                left: Box::new(typed_left),
                operator,
                right: Box::new(typed_right),
                expr_type,
            })
        }

        Expression::Identifier(name) => {
            let symbol = symbols.lookup(&name)?;
            Ok(TypedExpression::Identifier {
                name,
                mutable: symbol.mutable,
                expr_type: symbol.symbol_type.clone(),
            })
        }

        Expression::IntegerLiteral(i) => {
            let expr_type = match hint {
                Some(Type::I32) => {
                    if i32::try_from(i).is_err() {
                        return Err(TypeError {
                            msg: format!("integer literal {i} does not fit in i32"),
                        });
                    }
                    Type::I32
                }
                _ => Type::I64,
            };
            Ok(TypedExpression::IntegerLiteral {
                value: i,
                expr_type,
            })
        }

        Expression::FloatLiteral(f) => {
            let expr_type = match hint {
                Some(Type::F32) => Type::F32,
                _ => Type::F64,
            };
            Ok(TypedExpression::FloatLiteral { value: f, expr_type })
        }

        Expression::BooleanLiteral(b) => Ok(TypedExpression::BooleanLiteral {
            value: b,
            expr_type: Type::Bool,
        }),

        Expression::StringLiteral(s) => Ok(TypedExpression::StringLiteral {
            value: s,
            expr_type: Type::String,
        }),

        Expression::Unary { operator, operand } => {
            let operand_hint = match (&operator, hint) {
                (UnaryOperator::Negate, Some(hint)) if is_number(hint) => Some(hint),
                _ => None,
            };
            let typed_operand = analyse_expression(*operand, symbols, operand_hint)?;
            let operand_type = typed_operand.get_type();

            // Validate operator
            match (&operator, &operand_type) {
                (UnaryOperator::Negate, Type::I32 | Type::I64 | Type::F32 | Type::F64) => {}
                (UnaryOperator::Not, Type::Bool) => {}
                _ => {
                    return Err(TypeError {
                        msg: format!(
                            "Invalid unary operator {:?} for type {:?}",
                            operator, operand_type
                        ),
                    });
                }
            }

            Ok(TypedExpression::Unary {
                operator,
                operand: Box::new(typed_operand.clone()),
                expr_type: operand_type.clone(),
            })
        }

        Expression::FunctionCall { name, arguments } => {
            if name == "print" {
                if arguments.len() != 1 {
                    return Err(TypeError {
                        msg: format!("print expects 1 argument, got {}", arguments.len()),
                    });
                }
                let typed_arg =
                    analyse_expression(arguments.into_iter().next().unwrap(), symbols, None)?;
                return Ok(TypedExpression::FunctionCall {
                    name,
                    arguments: vec![typed_arg],
                    expr_type: Type::Unit,
                });
            }

            let func_symbol = symbols.lookup(&name)?;

            let (param_types, return_type) = match &func_symbol.symbol_type {
                Type::Function {
                    param_types,
                    return_type,
                } => (param_types, return_type.as_ref()),
                _ => {
                    return Err(TypeError {
                        msg: format!("'{}' is not a function", name),
                    });
                }
            };

            // Check argument count
            if arguments.len() != param_types.len() {
                return Err(TypeError {
                    msg: format!(
                        "Function '{}' expects {} arguments, got {}",
                        name,
                        param_types.len(),
                        arguments.len()
                    ),
                });
            }

            // Analyze and type-check each argument
            let mut typed_args = Vec::new();
            for (i, arg) in arguments.into_iter().enumerate() {
                let expected = param_types[i].clone();
                let typed_arg = analyse_expression(arg, symbols, Some(&expected))?;
                let arg_type = typed_arg.get_type();

                if arg_type != &param_types[i] {
                    return Err(TypeError {
                        msg: format!(
                            "Argument {} of function '{}': expected {:?}, got {:?}",
                            i, name, param_types[i], arg_type
                        ),
                    });
                }

                typed_args.push(typed_arg);
            }

            Ok(TypedExpression::FunctionCall {
                name,
                arguments: typed_args,
                expr_type: return_type.clone(),
            })
        }
    }
}

fn is_number(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::I64 | Type::F32 | Type::F64)
}

fn arithmetic_hint<'a>(hint: Option<&'a Type>, op: &BinaryOperator) -> Option<&'a Type> {
    match (hint, op) {
        (
            Some(hint),
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo,
        ) if is_number(hint) => Some(hint),
        _ => None,
    }
}

/// An unsuffixed numeric literal takes the other operand's type when it fits.
fn adapt_literal(expr: &mut TypedExpression, target: &Type) -> Result<(), TypeError> {
    match expr {
        TypedExpression::IntegerLiteral { value, expr_type } if expr_type != target => {
            match target {
                Type::I32 => {
                    if i32::try_from(*value).is_err() {
                        return Err(TypeError {
                            msg: format!("integer literal {value} does not fit in i32"),
                        });
                    }
                    *expr_type = Type::I32;
                }
                Type::I64 => *expr_type = Type::I64,
                _ => {}
            }
        }
        TypedExpression::FloatLiteral { expr_type, .. }
            if matches!(target, Type::F32 | Type::F64) && expr_type != target =>
        {
            *expr_type = target.clone();
        }
        _ => {}
    }
    Ok(())
}

fn binary_result_type(left: &Type, op: &BinaryOperator) -> Result<Type, TypeError> {
    let result = match (left, op) {
        (
            Type::I32 | Type::I64 | Type::F32 | Type::F64,
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo,
        ) => Some(left.clone()),
        (
            Type::I32 | Type::I64 | Type::F32 | Type::F64,
            BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual,
        ) => Some(Type::Bool),
        (Type::String, BinaryOperator::Add) => Some(Type::String),
        (Type::String | Type::Bool, BinaryOperator::Equal | BinaryOperator::NotEqual) => {
            Some(Type::Bool)
        }
        _ => None,
    };
    result.ok_or_else(|| TypeError {
        msg: format!("Operator {:?} not supported for {:?}", op, left),
    })
}

fn analyse_in_new_scope(
    block: Block,
    symbols: &mut SymbolTable,
    expected_return: Option<&Type>,
    in_loop: bool,
) -> Result<TypedBlock, TypeError> {
    symbols.push_scope();
    let result = analyse_block(block, symbols, expected_return, in_loop);
    symbols.pop_scope();
    result
}

fn guarantees_return(block: &TypedBlock) -> bool {
    block.statements.iter().any(stmt_returns) || trailing_value_type(block).is_some()
}

fn stmt_returns(stmt: &TypedStatement) -> bool {
    match stmt {
        TypedStatement::Return { .. } => true,
        TypedStatement::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => branch_returns(then_branch) && branch_returns(else_branch),
        _ => false,
    }
}

/// A branch value is not a function return. Only `return` counts.
fn branch_returns(block: &TypedBlock) -> bool {
    block.statements.iter().any(stmt_returns)
}

fn analyse_function_body(
    parameters: &[Parameter],
    body: Block,
    symbols: &mut SymbolTable,
    ret_type: &Type,
) -> Result<TypedBlock, TypeError> {
    for param in parameters {
        symbols.insert(
            param.name.clone(),
            Symbol {
                symbol_type: parse_type(&param.type_annotation)?,
                mutable: false,
            },
        )?;
    }
    analyse_block(body, symbols, Some(ret_type), false)
}

/// Type of a block's trailing expression, when the block ends in a value.
///
/// A final `return_expression` wins. Otherwise the last expression statement
/// is the value. `let`, assignment, `if`, and `while` do not produce one.
fn trailing_value_type(block: &TypedBlock) -> Option<&Type> {
    if let Some(expr) = &block.return_expression {
        return Some(expr.get_type());
    }
    match block.statements.last() {
        Some(TypedStatement::Expression(expr)) => Some(expr.get_type()),
        _ => None,
    }
}

fn parse_type(type_str: &str) -> Result<Type, TypeError> {
    match type_str {
        "i32" => Ok(Type::I32),
        "i64" => Ok(Type::I64),
        "f32" => Ok(Type::F32),
        "f64" => Ok(Type::F64),
        "bool" => Ok(Type::Bool),
        // accept common aliases produced by the grammar
        "String" | "string" | "str" => Ok(Type::String),
        "()" => Ok(Type::Unit),
        other => Err(TypeError {
            msg: format!("Unknown type: {}", other),
        }),
    }
}

fn analyse_block(
    block: Block,
    symbols: &mut SymbolTable,
    expected_return: Option<&Type>,
    in_loop: bool,
) -> Result<TypedBlock, TypeError> {
    let mut typed_statements = Vec::new();

    // Analyze each statement in the block
    for statement in block.statements {
        let typed_stmt = analyse_statement(statement, symbols, expected_return, in_loop)?;
        typed_statements.push(typed_stmt);
    }

    // Analyze the return expression if present
    let typed_return_expr = if let Some(return_expr) = block.return_expression {
        Some(analyse_expression(return_expr, symbols, expected_return)?)
    } else {
        None
    };

    Ok(TypedBlock {
        statements: typed_statements,
        return_expression: typed_return_expr,
    })
}

fn analyse_statement(
    statement: Statement,
    symbols: &mut SymbolTable,
    expected_return: Option<&Type>,
    in_loop: bool,
) -> Result<TypedStatement, TypeError> {
    match statement {
        Statement::Let {
            mutable,
            name,
            type_annotation,
            value,
        } => {
            // Parse the optional type annotation (now returns Result)
            let expected_type = if let Some(t) = type_annotation.as_ref() {
                Some(parse_type(t)?)
            } else {
                None
            };

            // Analyze the value expression with context
            let typed_value = if let Some(expected) = &expected_type {
                analyse_expression_with_hint(value, symbols, expected)?
            } else {
                analyse_expression(value, symbols, None)?
            };

            let inferred_type = typed_value.get_type().clone();

            if let Some(declared_type) = expected_type {
                if declared_type != inferred_type {
                    return Err(TypeError {
                        msg: format!(
                            "Type mismatch for '{}': declared as {:?}, but value has type {:?}",
                            name, declared_type, inferred_type
                        ),
                    });
                }
            }

            // Add to symbol table
            symbols.insert(
                name.clone(),
                Symbol {
                    symbol_type: inferred_type.clone(),
                    mutable,
                },
            )?;

            Ok(TypedStatement::Let {
                mutable,
                name,
                type_annotation: inferred_type,
                value: typed_value,
            })
        }

        Statement::Assignment { name, value } => {
            // Check variable exists and is mutable
            let symbol = symbols.lookup(&name)?;
            if !symbol.mutable {
                return Err(TypeError {
                    msg: format!("Cannot assign to immutable variable '{}'", name),
                });
            }

            let var_type = symbol.symbol_type.clone();

            // Analyze value and check type matches
            let typed_value = analyse_expression(value, symbols, Some(&var_type))?;
            let value_type = typed_value.get_type();

            if &var_type != value_type {
                return Err(TypeError {
                    msg: format!(
                        "Type mismatch in assignment to '{}': expected {:?}, got {:?}",
                        name, var_type, value_type
                    ),
                });
            }

            Ok(TypedStatement::Assignment {
                name,
                value: typed_value,
            })
        }

        Statement::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        } => {
            if name == "print" {
                return Err(TypeError {
                    msg: "print is reserved".to_string(),
                });
            }

            // Build function type (parse parameter types, propagate errors)
            let mut param_types: Vec<Type> = Vec::new();
            for p in &parameters {
                param_types.push(parse_type(&p.type_annotation)?);
            }

            let ret_type = if let Some(t) = return_type.as_ref() {
                parse_type(t)?
            } else {
                Type::Unit
            };

            let func_type = Type::Function {
                param_types: param_types.clone(),
                return_type: Box::new(ret_type.clone()),
            };

            // Insert before the body so direct recursion type-checks.
            symbols.insert(
                name.clone(),
                Symbol {
                    symbol_type: func_type,
                    mutable: false,
                },
            )?;

            symbols.push_scope();
            let typed_body = analyse_function_body(&parameters, body, symbols, &ret_type);
            symbols.pop_scope();
            let typed_body = typed_body?;

            if let Some(actual) = trailing_value_type(&typed_body) {
                if actual != &ret_type {
                    return Err(TypeError {
                        msg: format!(
                            "Function '{}' return type mismatch: expected {:?}, got {:?}",
                            name, ret_type, actual
                        ),
                    });
                }
            }
            if ret_type != Type::Unit && !guarantees_return(&typed_body) {
                return Err(TypeError {
                    msg: format!("Function '{name}' might not return {ret_type:?}"),
                });
            }

            Ok(TypedStatement::FunctionDeclaration {
                name,
                parameters,
                return_type: ret_type,
                body: typed_body,
            })
        }

        Statement::Return { value } => {
            let typed_value = if let Some(expr) = value {
                Some(analyse_expression(expr, symbols, expected_return)?)
            } else {
                None
            };

            let actual = match &typed_value {
                Some(expr) => expr.get_type().clone(),
                None => Type::Unit,
            };
            if let Some(expected) = expected_return {
                if &actual != expected {
                    return Err(TypeError {
                        msg: format!(
                            "return type mismatch: expected {:?}, got {:?}",
                            expected, actual
                        ),
                    });
                }
            }

            Ok(TypedStatement::Return { value: typed_value })
        }

        Statement::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let typed_condition = analyse_expression(condition, symbols, None)?;
            if typed_condition.get_type() != &Type::Bool {
                return Err(TypeError {
                    msg: format!(
                        "if condition must be bool, got {:?}",
                        typed_condition.get_type()
                    ),
                });
            }
            let typed_then = analyse_in_new_scope(then_branch, symbols, expected_return, in_loop)?;
            let typed_else = if let Some(branch) = else_branch {
                Some(analyse_in_new_scope(branch, symbols, expected_return, in_loop)?)
            } else {
                None
            };
            Ok(TypedStatement::If {
                condition: typed_condition,
                then_branch: typed_then,
                else_branch: typed_else,
            })
        }

        Statement::While { condition, body } => {
            let typed_condition = analyse_expression(condition, symbols, None)?;
            if typed_condition.get_type() != &Type::Bool {
                return Err(TypeError {
                    msg: format!(
                        "while condition must be bool, got {:?}",
                        typed_condition.get_type()
                    ),
                });
            }
            let typed_body = analyse_in_new_scope(body, symbols, expected_return, true)?;
            Ok(TypedStatement::While {
                condition: typed_condition,
                body: typed_body,
            })
        }

        Statement::Expression(expr) => {
            let typed_expr = analyse_expression(expr, symbols, expected_return)?;
            Ok(TypedStatement::Expression(typed_expr))
        }
    }
}

fn analyse_expression_with_hint(
    expr: Expression,
    symbols: &SymbolTable,
    hint: &Type,
) -> Result<TypedExpression, TypeError> {
    analyse_expression(expr, symbols, Some(hint))
}

/// Lightweight REPL symbol table wrapper used by the interactive REPL.
///
/// This keeps a persistent `SymbolTable` across user inputs so declarations
/// (variables, functions) remain available between evaluations.
pub struct Repl {
    pub symbols: SymbolTable,
}

impl Repl {
    /// Create a new REPL with an empty global scope.
    pub fn new() -> Self {
        Repl {
            symbols: SymbolTable::new(),
        }
    }

    /// Analyze a full program (a sequence of statements), updating the REPL's symbol table.
    ///
    /// This uses the existing incremental `analyse_statement` implementation so the
    /// symbol table is updated as each statement is processed.
    pub fn feed_program(&mut self, program: Program) -> Result<TypedProgram, TypeError> {
        let saved = self.symbols.clone();
        let mut typed_statements = Vec::new();

        for stmt in program.statements {
            match analyse_statement(stmt, &mut self.symbols, None, false) {
                Ok(typed) => typed_statements.push(typed),
                Err(err) => {
                    self.symbols = saved;
                    return Err(err);
                }
            }
        }

        Ok(TypedProgram {
            statements: typed_statements,
        })
    }

    /// Re-analyse statements that already ran, after a runtime failure rolled
    /// the symbol table back to the state before this input.
    pub fn replay(&mut self, statements: &[Statement]) -> Result<(), TypeError> {
        for stmt in statements {
            analyse_statement(stmt.clone(), &mut self.symbols, None, false)?;
        }
        Ok(())
    }

    /// Reset the REPL to an empty global scope.
    pub fn reset(&mut self) {
        self.symbols = SymbolTable::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    fn analyse(src: &str) -> Result<TypedProgram, TypeError> {
        let program = parse_program(src).unwrap();
        Repl::new().feed_program(program)
    }

    #[test]
    fn if_condition_must_be_bool() {
        let err = analyse("if 1 { return 1 }").unwrap_err();
        assert!(err.msg.contains("bool"), "{}", err.msg);
    }

    #[test]
    fn return_must_match_function_type() {
        let err = analyse("func f() -> i64 { return true }").unwrap_err();
        assert!(err.msg.contains("return"), "{}", err.msg);
    }

    #[test]
    fn print_is_reserved() {
        let err = analyse("func print(x: i64) -> i64 { x }").unwrap_err();
        assert!(err.msg.contains("print"), "{}", err.msg);
    }

    #[test]
    fn failed_function_is_not_kept() {
        let mut repl = Repl::new();
        let program = parse_program("func f() -> i64 { return true }").unwrap();
        assert!(repl.feed_program(program).is_err());
        assert!(repl.symbols.lookup("f").is_err());
    }

    #[test]
    fn branch_locals_do_not_escape() {
        let err = analyse("if false { let x: i64 = 1 }\nx").unwrap_err();
        assert!(err.msg.contains("Undefined"), "{}", err.msg);
    }

    #[test]
    fn function_must_return_on_every_path() {
        let err = analyse("func f() -> i64 { if false { return 1 } }").unwrap_err();
        assert!(err.msg.contains("might not return"), "{}", err.msg);
    }

    #[test]
    fn bool_ordering_is_rejected() {
        let err = analyse("true < false").unwrap_err();
        assert!(err.msg.contains("not supported"), "{}", err.msg);
    }

    #[test]
    fn i32_literal_must_fit() {
        let err = analyse("let x: i32 = 2147483648").unwrap_err();
        assert!(err.msg.contains("does not fit"), "{}", err.msg);
    }
}
