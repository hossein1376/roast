use crate::ast::*;
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "grammar/roast.pest"]
pub struct RoastParser;

pub fn parse_program(input: &str) -> Result<Program, Box<dyn std::error::Error>> {
    let mut pairs = RoastParser::parse(Rule::program, input)?;
    let program_pair = pairs.next().unwrap();

    let mut statements = Vec::new();
    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::statement => {
                statements.push(build_statement(pair)?);
            }
            Rule::EOI => {}
            _ => {}
        }
    }

    Ok(Program { statements })
}

fn build_statement(pair: Pair<Rule>) -> Result<Statement, Box<dyn std::error::Error>> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::let_statement => build_let_statement(inner),
        Rule::assignment_statement => build_assignment_statement(inner),
        Rule::function_declaration => build_function_declaration(inner),
        Rule::return_statement => build_return_statement(inner),
        Rule::expression_statement => {
            let expr = build_expression(inner.into_inner().next().unwrap())?;
            Ok(Statement::Expression(expr))
        }
        _ => Err(format!("Unexpected statement rule: {:?}", inner.as_rule()).into()),
    }
}

fn build_let_statement(pair: Pair<Rule>) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut mutable = false;
    let mut name = String::new();
    let mut type_annotation = None;
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::mutability => mutable = true,
            Rule::identifier => name = inner.as_str().to_string(),
            Rule::type_annotation => type_annotation = Some(inner.as_str().to_string()),
            Rule::expression => value = Some(build_expression(inner)?),
            _ => {}
        }
    }

    Ok(Statement::Let {
        mutable,
        name,
        type_annotation,
        value: value.unwrap(),
    })
}

fn build_assignment_statement(pair: Pair<Rule>) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value = build_expression(inner.next().unwrap())?;

    Ok(Statement::Assignment { name, value })
}

fn build_function_declaration(pair: Pair<Rule>) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut name = String::new();
    let mut parameters = Vec::new();
    let mut return_type = None;
    let mut body = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => name = inner.as_str().to_string(),
            Rule::parameter_list => {
                for param_pair in inner.into_inner() {
                    parameters.push(build_parameter(param_pair)?);
                }
            }
            Rule::return_type => {
                return_type = Some(inner.into_inner().next().unwrap().as_str().to_string());
            }
            Rule::block => body = Some(build_block(inner)?),
            _ => {}
        }
    }

    Ok(Statement::FunctionDeclaration {
        name,
        parameters,
        return_type,
        body: body.unwrap(),
    })
}

fn build_parameter(pair: Pair<Rule>) -> Result<Parameter, Box<dyn std::error::Error>> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let type_annotation = inner.next().unwrap().as_str().to_string();

    Ok(Parameter {
        name,
        type_annotation,
    })
}

fn build_block(pair: Pair<Rule>) -> Result<Block, Box<dyn std::error::Error>> {
    let mut statements = Vec::new();
    let mut return_expression = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::statement => statements.push(build_statement(inner)?),
            Rule::return_expression => {
                return_expression = Some(build_expression(inner.into_inner().next().unwrap())?);
            }
            _ => {}
        }
    }

    Ok(Block {
        statements,
        return_expression,
    })
}

fn build_return_statement(pair: Pair<Rule>) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut value = None;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expression {
            value = Some(build_expression(inner)?);
        }
    }

    Ok(Statement::Return { value })
}

fn build_expression(pair: Pair<Rule>) -> Result<Expression, Box<dyn std::error::Error>> {
    match pair.as_rule() {
        Rule::expression | Rule::comparison | Rule::additive | Rule::multiplicative => {
            build_binary_expression(pair)
        }
        Rule::unary => build_unary_expression(pair),
        Rule::primary => build_primary_expression(pair),
        _ => Err(format!("Unexpected expression rule: {:?}", pair.as_rule()).into()),
    }
}

fn build_binary_expression(pair: Pair<Rule>) -> Result<Expression, Box<dyn std::error::Error>> {
    let rule = pair.as_rule();
    let mut inner = pair.into_inner();
    let mut left = match rule {
        Rule::comparison => build_expression(inner.next().unwrap())?,
        Rule::additive => build_expression(inner.next().unwrap())?,
        Rule::multiplicative => build_expression(inner.next().unwrap())?,
        Rule::expression => build_expression(inner.next().unwrap())?,
        _ => return Err("Invalid binary expression".into()),
    };

    while let Some(op_pair) = inner.next() {
        let operator = match op_pair.as_str() {
            "+" => BinaryOperator::Add,
            "-" => BinaryOperator::Subtract,
            "*" => BinaryOperator::Multiply,
            "/" => BinaryOperator::Divide,
            "%" => BinaryOperator::Modulo,
            "==" => BinaryOperator::Equal,
            "!=" => BinaryOperator::NotEqual,
            "<" => BinaryOperator::LessThan,
            "<=" => BinaryOperator::LessThanOrEqual,
            ">" => BinaryOperator::GreaterThan,
            ">=" => BinaryOperator::GreaterThanOrEqual,
            _ => return Err(format!("Unknown operator: {}", op_pair.as_str()).into()),
        };

        let right = build_expression(inner.next().unwrap())?;

        left = Expression::Binary {
            left: Box::new(left),
            operator,
            right: Box::new(right),
        };
    }

    Ok(left)
}

fn build_unary_expression(pair: Pair<Rule>) -> Result<Expression, Box<dyn std::error::Error>> {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();

    match first.as_rule() {
        Rule::unary_op => {
            let operator = match first.as_str() {
                "-" => UnaryOperator::Negate,
                "!" => UnaryOperator::Not,
                _ => return Err(format!("Unknown unary operator: {}", first.as_str()).into()),
            };
            let operand = build_expression(inner.next().unwrap())?;
            Ok(Expression::Unary {
                operator,
                operand: Box::new(operand),
            })
        }
        Rule::primary => build_primary_expression(first),
        _ => Err("Invalid unary expression".into()),
    }
}

fn build_primary_expression(pair: Pair<Rule>) -> Result<Expression, Box<dyn std::error::Error>> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::integer_literal => {
            let value = inner.as_str().parse::<i64>()?;
            Ok(Expression::IntegerLiteral(value))
        }
        Rule::float_literal => {
            let value = inner.as_str().parse::<f64>()?;
            Ok(Expression::FloatLiteral(value))
        }
        Rule::boolean_literal => {
            let value = inner.as_str() == "true";
            Ok(Expression::BooleanLiteral(value))
        }
        Rule::string_literal => {
            let s = inner.as_str();
            let value = s[1..s.len() - 1].to_string(); // Remove quotes
            Ok(Expression::StringLiteral(value))
        }
        Rule::identifier => Ok(Expression::Identifier(inner.as_str().to_string())),
        Rule::function_call => build_function_call(inner),
        Rule::expression => build_expression(inner),
        _ => Err(format!("Unexpected primary expression: {:?}", inner.as_rule()).into()),
    }
}

fn build_function_call(pair: Pair<Rule>) -> Result<Expression, Box<dyn std::error::Error>> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let mut arguments = Vec::new();

    if let Some(arg_list) = inner.next() {
        if arg_list.as_rule() == Rule::argument_list {
            for arg in arg_list.into_inner() {
                arguments.push(build_expression(arg)?);
            }
        }
    }

    Ok(Expression::FunctionCall { name, arguments })
}
