use criterion::{black_box, criterion_group, criterion_main, Criterion};
use parcel::join;
use parcel::parsers::character::{digit, expect_character, whitespace};
use parcel::prelude::v1::*;

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Additive(AdditiveKind),
    Multiplicative(MultiplicativeKind),
    Primary(Primary),
}

#[derive(Debug, PartialEq, Clone)]
pub enum AdditiveKind {
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum MultiplicativeKind {
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Primary {
    Int(i64),
}

pub fn expression<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], Expr> {
    additive()
}

enum AdditiveOp {
    Plus,
    Minus,
}

#[allow(clippy::redundant_closure)]
fn additive<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], Expr> {
    join(
        multiplicative(),
        parcel::zero_or_more(join(
            whitespace_wrapped(expect_character('+'))
                .map(|_| AdditiveOp::Plus)
                .or(|| whitespace_wrapped(expect_character('-')).map(|_| AdditiveOp::Minus)),
            multiplicative(),
        ))
        .map(unzip),
    )
    .map(|(first_expr, (operators, operands))| {
        operators
            .into_iter()
            .zip(operands.into_iter())
            .fold(first_expr, |lhs, (operator, rhs)| match operator {
                AdditiveOp::Plus => Expr::Additive(AdditiveKind::Add(Box::new(lhs), Box::new(rhs))),
                AdditiveOp::Minus => {
                    Expr::Additive(AdditiveKind::Sub(Box::new(lhs), Box::new(rhs)))
                }
            })
    })
    .or(|| primary())
}

enum MultiplicativeOp {
    Star,
    Slash,
}

#[allow(clippy::redundant_closure)]
fn multiplicative<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], Expr> {
    join(
        primary(),
        parcel::zero_or_more(join(
            whitespace_wrapped(expect_character('*'))
                .map(|_| MultiplicativeOp::Star)
                .or(|| whitespace_wrapped(expect_character('/')).map(|_| MultiplicativeOp::Slash)),
            primary(),
        ))
        .map(unzip),
    )
    .map(|(first_expr, (operators, operands))| {
        operators
            .into_iter()
            .zip(operands.into_iter())
            .fold(first_expr, |lhs, (operator, rhs)| match operator {
                MultiplicativeOp::Star => {
                    Expr::Multiplicative(MultiplicativeKind::Mul(Box::new(lhs), Box::new(rhs)))
                }
                MultiplicativeOp::Slash => {
                    Expr::Multiplicative(MultiplicativeKind::Div(Box::new(lhs), Box::new(rhs)))
                }
            })
    })
    .or(|| primary())
}

#[allow(clippy::redundant_closure)]
fn primary<'a>() -> impl parcel::Parser<'a, &'a [(usize, char)], Expr> {
    dec_i64().map(|i| Expr::Primary(Primary::Int(i)))
}

fn unzip<A, B>(pair: Vec<(A, B)>) -> (Vec<A>, Vec<B>) {
    pair.into_iter().unzip()
}

fn whitespace_wrapped<'a, P, B>(parser: P) -> impl Parser<'a, &'a [(usize, char)], B>
where
    B: 'a,
    P: Parser<'a, &'a [(usize, char)], B> + 'a,
{
    parcel::right(parcel::join(
        parcel::zero_or_more(whitespace()),
        parcel::left(parcel::join(parser, parcel::zero_or_more(whitespace()))),
    ))
}

fn dec_i64<'a>() -> impl Parser<'a, &'a [(usize, char)], i64> {
    move |input: &'a [(usize, char)]| {
        let preparsed_input = input;
        let res = parcel::join(
            whitespace_wrapped(expect_character('-')).optional(),
            parcel::one_or_more(digit(10)),
        )
        .map(|(negative, digits)| {
            let vd: String = if negative.is_some() {
                format!("-{}", digits.into_iter().collect::<String>())
            } else {
                digits.into_iter().collect()
            };
            vd.parse::<i64>()
        })
        .parse(input);

        match res {
            Ok(MatchStatus::Match {
                span,
                remainder,
                inner: Ok(u),
            }) => Ok(MatchStatus::Match {
                span,
                remainder,
                inner: u,
            }),

            Ok(MatchStatus::Match {
                span: _,
                remainder: _,
                inner: Err(_),
            }) => Ok(MatchStatus::NoMatch(preparsed_input)),

            Ok(MatchStatus::NoMatch(remainder)) => Ok(MatchStatus::NoMatch(remainder)),
            Err(e) => Err(e),
        }
    }
}

fn parse_basic_expression(c: &mut Criterion) {
    let input: Vec<(usize, char)> = "10 / 5 + 1".char_indices().collect();

    let expected = Expr::Additive(AdditiveKind::Add(
        Box::new(Expr::Multiplicative(MultiplicativeKind::Div(
            Box::new(Expr::Primary(Primary::Int(10))),
            Box::new(Expr::Primary(Primary::Int(5))),
        ))),
        Box::new(Expr::Primary(Primary::Int(1))),
    ));

    let expected = Ok(expected);

    c.bench_function("simple calculator expression parsing", |b| {
        b.iter(|| {
            let parse_tree = expression().parse(black_box(&input)).map(|ms| ms.unwrap());

            assert_eq!(&parse_tree, &expected)
        });
    });
}

fn parse_large_expression(c: &mut Criterion) {
    let input: Vec<(usize, char)> = ["10"]
        .into_iter()
        .chain(["/ 5", "+ 1", "- 2", "* 6"].into_iter().cycle())
        .take(100)
        .collect::<String>()
        .char_indices()
        .collect();

    c.bench_function("large expression parsing", |b| {
        b.iter(|| {
            let parse_tree = expression().parse(black_box(&input)).map(|ms| ms.unwrap());

            assert!(parse_tree.is_ok())
        });
    });
}

criterion_group!(benches, parse_basic_expression, parse_large_expression);
criterion_main!(benches);
