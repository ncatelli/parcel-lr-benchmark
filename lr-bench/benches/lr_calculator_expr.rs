use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lr_core::{TerminalOrNonTerminal, TerminalRepresentable};
pub use lr_derive::Lr1;
pub use relex_derive::{Relex, VariantKind};

#[derive(VariantKind, Relex, Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, PartialEq)]
pub enum MultiplicativeTermKind {
    Mul(NonTerminal, NonTerminal),
    Div(NonTerminal, NonTerminal),
    Unary(NonTerminal),
}

#[derive(Debug, PartialEq)]
pub enum AdditiveTermKind {
    Add(NonTerminal, NonTerminal),
    Sub(NonTerminal, NonTerminal),
    Unary(NonTerminal),
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
fn reduce_expr_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        Ok(NonTerminal::Expr(Box::new(nonterm)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_multiplicative_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = MultiplicativeTermKind::Unary(nonterm);

        Ok(NonTerminal::Multiplicative(Box::new(non_term_kind)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
    }
}

#[allow(unused)]
fn reduce_additive_unary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(nonterm)) = elems.pop() {
        let non_term_kind = AdditiveTermKind::Unary(nonterm);

        Ok(NonTerminal::Additive(Box::new(non_term_kind)))
    } else {
        Err(format!(
            "expected non-terminal at top of stack in production 3 reducer.",
        ))
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
    let err_msg = format!(
        "expected 3 elements at top of stack in production {} reducer. got [{:?}, {:?}, {:?}]",
        production_id, &optional_lhs, &optional_term, &optional_rhs
    );

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op {
            Terminal::Star => MultiplicativeTermKind::Mul(lhs, rhs),
            Terminal::Slash => MultiplicativeTermKind::Div(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::Multiplicative(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_additive_binary(elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();
    let err_msg = format!(
        "expected 3 elements at top of stack in production  reducer. got [{:?}, {:?}, {:?}]",
        &optional_lhs, &optional_term, &optional_rhs
    );

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(lhs)), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(rhs))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op {
            Terminal::Plus => AdditiveTermKind::Add(lhs, rhs),
            Terminal::Minus => AdditiveTermKind::Sub(lhs, rhs),
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        Ok(NonTerminal::Additive(Box::new(non_term_kind)))
    } else {
        Err(err_msg)
    }
}

#[derive(Debug, Lr1, PartialEq)]
pub enum NonTerminal {
    #[goal(r"<Expr>", reduce_expr_unary)]
    #[production(r"<Additive>", reduce_expr_unary)]
    Expr(Box<NonTerminal>),
    #[production(r"<Additive> Terminal::Plus <Multiplicative>", reduce_additive_binary)]
    #[production(r"<Additive> Terminal::Minus <Multiplicative>", reduce_additive_binary)]
    #[production(r"<Multiplicative>", reduce_additive_unary)]
    Additive(Box<AdditiveTermKind>),
    #[production(r"<Multiplicative> Terminal::Star <Primary>", |elems| { reduce_multiplicative_binary(6, elems) })]
    #[production(r"<Multiplicative> Terminal::Slash <Primary>", |elems| { reduce_multiplicative_binary(7, elems) })]
    #[production(r"<Primary>", reduce_multiplicative_unary)]
    Multiplicative(Box<MultiplicativeTermKind>),
    #[production(r"Terminal::Int", reduce_primary)]
    Primary(Terminal),
}

fn parse_basic_expression(c: &mut Criterion) {
    let input = "10 / 5 + 1";

    let expected = NonTerminal::Expr(Box::new(NonTerminal::Additive(Box::new(
        AdditiveTermKind::Add(
            NonTerminal::Additive(Box::new(AdditiveTermKind::Unary(
                NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Div(
                    NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Unary(
                        NonTerminal::Primary(Terminal::Int(10)),
                    ))),
                    NonTerminal::Primary(Terminal::Int(5)),
                ))),
            ))),
            NonTerminal::Multiplicative(Box::new(MultiplicativeTermKind::Unary(
                NonTerminal::Primary(Terminal::Int(1)),
            ))),
        ),
    ))));

    let expected = Ok(expected);

    c.bench_function("simple_calculator_expr_parsing", |b| {
        b.iter(|| {
            let tokenizer = token_stream_from_input(black_box(input))
                .unwrap()
                .map(|token| token.to_variant())
                .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
                // append a single eof.
                .chain([Terminal::Eof].into_iter());

            let parse_tree = lr_parse_input(tokenizer);
            assert_eq!(&parse_tree, &expected);
        });
    });
}

criterion_group!(benches, parse_basic_expression,);
criterion_main!(benches);
