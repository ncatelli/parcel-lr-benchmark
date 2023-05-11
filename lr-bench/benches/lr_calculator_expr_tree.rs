use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lr_core::prelude::v1::*;
use lr_core::TerminalOrNonTerminal;
pub use lr_derive::Lr1;
pub use relex_derive::{Relex, VariantKind};

#[derive(VariantKind, Relex, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    #[matches(r"+")]
    Plus,
    #[matches(r"-")]
    Minus,
    #[matches(r"*")]
    Star,
    #[matches(r"/")]
    Slash,
    #[matches(r"[0-9]+", |lex: &str| { lex.parse::<i64>().ok() })]
    Int(i64),
    #[eoi]
    Eof,
}

impl std::fmt::Display for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Int(i) => write!(f, "{}", i),
            Self::Eof => write!(f, "<$>"),
        }
    }
}

impl TerminalRepresentable for Terminal {
    /// the associated type representing the variant kind.
    type Repr = <Self as VariantKindRepresentable>::Output;

    fn eof() -> Self::Repr {
        Self::Repr::Eof
    }

    fn to_variant_repr(&self) -> Self::Repr {
        self.to_variant_kind()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Star,
    Slash,
}

#[derive(Debug, PartialEq)]
pub struct BinaryExpr {
    pub lhs: NonTerminal,
    pub operator: BinaryOperator,
    pub rhs: NonTerminal,
}

impl BinaryExpr {
    pub fn new(lhs: NonTerminal, operator: BinaryOperator, rhs: NonTerminal) -> Self {
        Self { lhs, operator, rhs }
    }
}

#[derive(Debug, PartialEq)]
pub struct UnaryExpr {
    pub lhs: NonTerminal,
}

impl UnaryExpr {
    pub fn new(lhs: NonTerminal) -> Self {
        Self { lhs }
    }
}

#[derive(Debug, PartialEq)]
pub enum ExprInner {
    Unary(UnaryExpr),
    Binary(BinaryExpr),
}

type TermOrNonTerm = TerminalOrNonTerminal<Terminal, NonTerminal>;

#[allow(unused)]
fn reduce_primary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
        Ok(NonTerminal::Primary(term))
    } else {
        Err("expected terminal at top of stack in reducer.".to_string())
    }
}

#[allow(unused)]
fn reduce_expr_unary(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    // the only top level expr is an additive expr.
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Additive(inner))) = elems.pop() {
        Ok(NonTerminal::Expr(inner))
    } else {
        let err_msg = format!(
            "expected non-terminal at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_multiplicative_unary(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let inner = ExprInner::Unary(UnaryExpr::new(nonterm));

        Ok(NonTerminal::Multiplicative(Box::new(inner)))
    } else {
        let err_msg = format!(
            "expected non-terminal at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_additive_unary(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let inner = ExprInner::Unary(UnaryExpr::new(nonterm));

        Ok(NonTerminal::Additive(Box::new(inner)))
    } else {
        let err_msg = format!(
            "expected non-terminal at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_multiplicative_binary(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op {
            Terminal::Star => BinaryOperator::Star,
            Terminal::Slash => BinaryOperator::Slash,
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        let bin_expr = BinaryExpr::new(lhs, BinaryOperator::Slash, rhs);
        let inner = ExprInner::Binary(bin_expr);

        Ok(NonTerminal::Multiplicative(Box::new(inner)))
    } else {
        let err_msg = format!(
            "expected 3 elements at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_additive_binary(
    production_id: usize,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let bin_op = match op {
            Terminal::Plus => BinaryOperator::Plus,
            Terminal::Minus => BinaryOperator::Minus,
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        let bin_expr = BinaryExpr::new(lhs, bin_op, rhs);
        let inner = ExprInner::Binary(bin_expr);

        Ok(NonTerminal::Additive(Box::new(inner)))
    } else {
        let err_msg = format!(
            "expected 3 elements at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_goal(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Expr(inner))) = elems.pop() {
        Ok(NonTerminal::Expr(inner))
    } else {
        Err(format!("expected Expr non-terminal at top of stack.",))
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<Expr>", reduce_goal)]
    #[production(r"<Additive>", |elems| reduce_expr_unary(2, elems))]
    Expr(Box<ExprInner>),
    #[production(r"<Additive> Terminal::Plus <Multiplicative>", |elems| reduce_additive_binary(3, elems))]
    #[production(r"<Additive> Terminal::Minus <Multiplicative>", |elems| reduce_additive_binary(4, elems))]
    #[production(r"<Multiplicative>", |elems| reduce_additive_unary(5, elems))]
    Additive(Box<ExprInner>),
    #[production(r"<Multiplicative> Terminal::Star <Primary>", |elems| { reduce_multiplicative_binary(6, elems) })]
    #[production(r"<Multiplicative> Terminal::Slash <Primary>", |elems| { reduce_multiplicative_binary(7, elems) })]
    #[production(r"<Primary>", |elems| reduce_multiplicative_unary(8, elems))]
    Multiplicative(Box<ExprInner>),
    #[production(r"Terminal::Int", reduce_primary)]
    Primary(Terminal),
}

impl NonTerminalRepresentable for NonTerminal {
    type Terminal = Terminal;
}

fn parse_basic_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple calculator expression parsing (tree)");
    let input = "10 / 5 + 1";

    let expected = NonTerminal::Expr(Box::new(ExprInner::Binary(BinaryExpr::new(
        NonTerminal::Additive(Box::new(ExprInner::Unary(UnaryExpr::new(
            NonTerminal::Multiplicative(Box::new(ExprInner::Binary(BinaryExpr::new(
                NonTerminal::Multiplicative(Box::new(ExprInner::Unary(UnaryExpr::new(
                    NonTerminal::Primary(Terminal::Int(10)),
                )))),
                BinaryOperator::Slash,
                NonTerminal::Primary(Terminal::Int(5)),
            )))),
        )))),
        BinaryOperator::Plus,
        NonTerminal::Multiplicative(Box::new(ExprInner::Unary(UnaryExpr::new(
            NonTerminal::Primary(Terminal::Int(1)),
        )))),
    ))));

    let expected = Ok(expected);

    group.bench_function("with tokenization", |b| {
        b.iter(|| {
            let tokenizer = token_stream_from_input(black_box(input))
                .unwrap()
                .map(|token| token.to_variant())
                .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
                // append a single eof.
                .chain([Terminal::Eof].into_iter());

            let parse_tree = LrParseable::parse_input(tokenizer);
            assert_eq!(&parse_tree, &expected);
        });
    });

    group.bench_function("without tokenization", |b| {
        let tokenizer = token_stream_from_input(black_box(&input))
            .unwrap()
            .map(|token| token.to_variant())
            .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
            // append a single eof.
            .chain([Terminal::Eof].into_iter());

        let token_stream = tokenizer.collect::<Vec<_>>();

        b.iter(|| {
            let parse_tree = LrParseable::parse_input((&token_stream).iter().copied());
            assert_eq!(&parse_tree, &expected);
        });
    });
}

fn parse_large_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("large expression (tree)");
    let input = ["10"]
        .into_iter()
        .chain(["/ 5", "+ 1", "- 2", "* 6"].into_iter().cycle())
        .take(100)
        .collect::<String>();

    group.bench_function("with tokenization", |b| {
        b.iter(|| {
            let tokenizer = token_stream_from_input(black_box(&input))
                .unwrap()
                .map(|token| token.to_variant())
                .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
                // append a single eof.
                .chain([Terminal::Eof].into_iter());

            let parse_tree = NonTerminal::parse_input(tokenizer);
            assert!(parse_tree.is_ok());
        });
    });

    group.bench_function("without tokenization", |b| {
        let tokenizer = token_stream_from_input(black_box(&input))
            .unwrap()
            .map(|token| token.to_variant())
            .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
            // append a single eof.
            .chain([Terminal::Eof].into_iter());

        let token_stream = tokenizer.collect::<Vec<_>>();

        b.iter(|| {
            let parse_tree = NonTerminal::parse_input((&token_stream).iter().copied());
            assert!(parse_tree.is_ok());
        });
    });
}

criterion_group!(benches, parse_basic_expression, parse_large_expression);
criterion_main!(benches);
