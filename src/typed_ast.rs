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

pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

pub struct Symbol {
    pub name: String,
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

    /// Check if a symbol exists in the current scope only
    pub fn exists_in_current_scope(&self, name: &str) -> bool {
        self.scopes.last().unwrap().contains_key(name)
    }

    /// Update a symbol (for assignments)
    pub fn update(&mut self, name: &str, new_type: Type) -> Result<(), TypeError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(symbol) = scope.get_mut(name) {
                if !symbol.mutable {
                    return Err(TypeError {
                        msg: format!("Cannot assign to immutable variable '{}'", name),
                    });
                }
                symbol.symbol_type = new_type;
                return Ok(());
            }
        }

        Err(TypeError {
            msg: format!("Undefined symbol '{}'", name),
        })
    }

    /// Get the current scope depth (useful for debugging)
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }
}

#[derive(Debug)]
pub struct TypeError {
    pub msg: String,
}

pub fn analyze_program(program: Program) -> Result<TypedProgram, TypeError> {
    let mut symbols = SymbolTable::new();
    let mut typed_statements = Vec::new();

    for statement in program.statements {
        let typed_stmt = analyse_statement(statement, &mut symbols)?;
        typed_statements.push(typed_stmt);
    }

    Ok(TypedProgram {
        statements: typed_statements,
    })
}

#[derive(Debug)]
pub struct TypedProgram {
    pub statements: Vec<TypedStatement>,
}

pub fn analyse_expression(
    expr: Expression,
    symbols: &SymbolTable,
) -> Result<TypedExpression, TypeError> {
    match expr {
        Expression::Binary {
            left,
            operator,
            right,
        } => {
            let typed_left = analyse_expression(*left, symbols)?;
            let typed_right = analyse_expression(*right, symbols)?;

            let left_type = typed_left.get_type();
            let right_type = typed_right.get_type();

            // Check type compatibility
            if left_type != right_type {
                return Err(TypeError {
                    msg: format!(
                        "Type mismatch: {:?} {:?} {:?}",
                        left_type, operator, right_type
                    ),
                });
            }

            // Determine result type based on operator
            let expr_type = match operator {
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Modulo => left_type.clone(),

                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::LessThan
                | BinaryOperator::LessThanOrEqual
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterThanOrEqual => Type::Bool,
            };

            // Validate operator for type
            match (&left_type, &operator) {
                (Type::String, BinaryOperator::Add) => {}      // OK
                (Type::String, BinaryOperator::Equal) => {}    // OK
                (Type::String, BinaryOperator::NotEqual) => {} // OK
                (Type::String, _) => {
                    return Err(TypeError {
                        msg: format!("Operator {:?} not supported for strings", operator),
                    });
                }
                _ => {}
            }

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

        Expression::IntegerLiteral(i) => Ok(TypedExpression::IntegerLiteral {
            value: i,
            expr_type: Type::I64,
        }),

        Expression::FloatLiteral(f) => Ok(TypedExpression::FloatLiteral {
            value: f,
            expr_type: Type::F64,
        }),

        Expression::BooleanLiteral(b) => Ok(TypedExpression::BooleanLiteral {
            value: b,
            expr_type: Type::Bool,
        }),

        Expression::StringLiteral(s) => Ok(TypedExpression::StringLiteral {
            value: s,
            expr_type: Type::String,
        }),

        Expression::Unary { operator, operand } => {
            let typed_operand = analyse_expression(*operand, symbols)?;
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
                let typed_arg = analyse_expression(arg, symbols)?;
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

fn analyse_function_declaration(
    func_decl: Statement,
    symbols: &mut SymbolTable,
) -> Result<TypedStatement, TypeError> {
    // Extract function details
    let (name, parameters, return_type, body) = match func_decl {
        Statement::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        } => (name, parameters, return_type, body),
        _ => unreachable!(),
    };

    // Build function type
    let param_types: Vec<Type> = parameters
        .iter()
        .map(|p| parse_type(&p.type_annotation))
        .collect();

    let ret_type = return_type.map(|t| parse_type(&t)).unwrap_or(Type::Unit); // or however you handle void

    let func_type = Type::Function {
        param_types: param_types.clone(),
        return_type: Box::new(ret_type.clone()),
    };

    // Add function to symbol table
    symbols.insert(
        name.clone(),
        Symbol {
            name: name.clone(),
            symbol_type: func_type,
            mutable: false,
        },
    )?;

    // Create new scope for function body
    symbols.push_scope();

    // Add parameters to scope
    for param in &parameters {
        symbols.insert(
            param.name.clone(),
            Symbol {
                name: param.name.clone(),
                symbol_type: parse_type(&param.type_annotation),
                mutable: false,
            },
        )?;
    }

    // Analyze body
    let typed_body = analyse_block(body, symbols)?;

    // Validate all return paths
    validate_function_returns(&name, &ret_type, &typed_body)?;

    // Pop function scope
    symbols.pop_scope();

    Ok(TypedStatement::FunctionDeclaration {
        name,
        parameters,
        return_type: ret_type,
        body: typed_body,
    })
}

fn parse_type(type_str: &str) -> Type {
    match type_str {
        "i32" => Type::I32,
        "i64" => Type::I64,
        "f32" => Type::F32,
        "f64" => Type::F64,
        "bool" => Type::Bool,
        "string" => Type::String,
        "()" => Type::Unit,
        _ => panic!("Unknown type: {}", type_str),
    }
}

fn analyse_block(block: Block, symbols: &mut SymbolTable) -> Result<TypedBlock, TypeError> {
    let mut typed_statements = Vec::new();

    // Analyze each statement in the block
    for statement in block.statements {
        let typed_stmt = analyse_statement(statement, symbols)?;
        typed_statements.push(typed_stmt);
    }

    // Analyze the return expression if present
    let typed_return_expr = if let Some(return_expr) = block.return_expression {
        Some(analyse_expression(return_expr, symbols)?)
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
) -> Result<TypedStatement, TypeError> {
    match statement {
        Statement::Let {
            mutable,
            name,
            type_annotation,
            value,
        } => {
            let expected_type = type_annotation.as_ref().map(|t| parse_type(t));

            // Analyze the value expression with context
            let typed_value = if let Some(expected) = &expected_type {
                analyse_expression_with_hint(value, symbols, expected)?
            } else {
                analyse_expression(value, symbols)?
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
                    name: name.clone(),
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
            let typed_value = analyse_expression(value, symbols)?;
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
            // Build function type
            let param_types: Vec<Type> = parameters
                .iter()
                .map(|p| parse_type(&p.type_annotation))
                .collect();

            let ret_type = return_type
                .as_ref()
                .map(|t| parse_type(t))
                .unwrap_or(Type::Unit);

            let func_type = Type::Function {
                param_types: param_types.clone(),
                return_type: Box::new(ret_type.clone()),
            };

            // Add function to current scope
            symbols.insert(
                name.clone(),
                Symbol {
                    name: name.clone(),
                    symbol_type: func_type,
                    mutable: false,
                },
            )?;

            // Create new scope for function body
            symbols.push_scope();

            // Add parameters to the new scope
            for param in &parameters {
                symbols.insert(
                    param.name.clone(),
                    Symbol {
                        name: param.name.clone(),
                        symbol_type: parse_type(&param.type_annotation),
                        mutable: false,
                    },
                )?;
            }

            // Analyze function body
            let typed_body = analyse_block(body, symbols)?;

            // Check return type matches
            if let Some(return_expr) = &typed_body.return_expression {
                let actual_return_type = return_expr.get_type();
                if &ret_type != actual_return_type {
                    return Err(TypeError {
                        msg: format!(
                            "Function '{}' return type mismatch: expected {:?}, got {:?}",
                            name, ret_type, actual_return_type
                        ),
                    });
                }
            }

            // Pop function scope
            symbols.pop_scope();

            Ok(TypedStatement::FunctionDeclaration {
                name,
                parameters,
                return_type: ret_type,
                body: typed_body,
            })
        }

        Statement::Return { value } => {
            let typed_value = if let Some(expr) = value {
                Some(analyse_expression(expr, symbols)?)
            } else {
                None
            };

            Ok(TypedStatement::Return { value: typed_value })
        }

        Statement::Expression(expr) => {
            let typed_expr = analyse_expression(expr, symbols)?;
            Ok(TypedStatement::Expression(typed_expr))
        }
    }
}

fn validate_function_returns(
    func_name: &str,
    expected_type: &Type,
    block: &TypedBlock,
) -> Result<(), TypeError> {
    let has_return = block.return_expression.is_some()
        || block
            .statements
            .iter()
            .any(|s| matches!(s, TypedStatement::Return { .. }));

    if !has_return && expected_type != &Type::Unit {
        return Err(TypeError {
            msg: format!(
                "Function '{}' declared to return {:?}, but has no return expression",
                func_name, expected_type
            ),
        });
    }

    // Check return expression type
    if let Some(return_expr) = &block.return_expression {
        if return_expr.get_type() != expected_type {
            return Err(TypeError {
                msg: format!(
                    "Function '{}' return type mismatch: expected {:?}, got {:?}",
                    func_name,
                    expected_type,
                    return_expr.get_type()
                ),
            });
        }
    }

    // Check return statement types
    for stmt in &block.statements {
        if let TypedStatement::Return { value: Some(expr) } = stmt {
            if expr.get_type() != expected_type {
                return Err(TypeError {
                    msg: format!(
                        "Function '{}' return type mismatch: expected {:?}, got {:?}",
                        func_name,
                        expected_type,
                        expr.get_type()
                    ),
                });
            }
        }
    }

    Ok(())
}

fn collect_return_types(block: &TypedBlock) -> Vec<Type> {
    let mut types = Vec::new();

    // Check final return expression
    if let Some(return_expr) = &block.return_expression {
        types.push(return_expr.get_type().clone());
    }

    // Check all statements for early returns
    for stmt in &block.statements {
        match stmt {
            TypedStatement::Return { value: Some(expr) } => {
                types.push(expr.get_type().clone());
            }
            TypedStatement::Return { value: None } => {
                types.push(Type::Unit);
            }
            // Could recurse into nested blocks here if needed
            _ => {}
        }
    }

    types
}

fn analyse_expression_with_hint(
    expr: Expression,
    symbols: &SymbolTable,
    hint: &Type,
) -> Result<TypedExpression, TypeError> {
    match expr {
        Expression::IntegerLiteral(i) => {
            // Use the hint to determine integer type
            let expr_type = match hint {
                Type::I32 => Type::I32,
                Type::I64 => Type::I64,
                _ => Type::I32, // Default
            };
            Ok(TypedExpression::IntegerLiteral {
                value: i,
                expr_type,
            })
        }
        // For other expressions, recurse without hint or analyze normally
        _ => analyse_expression(expr, symbols),
    }
}
