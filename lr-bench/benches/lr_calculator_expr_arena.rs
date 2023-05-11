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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryExpr {
    pub lhs: NonTerminalRef,
    pub operator: BinaryOperator,
    pub rhs: NonTerminalRef,
}

impl BinaryExpr {
    pub fn new(lhs: NonTerminalRef, operator: BinaryOperator, rhs: NonTerminalRef) -> Self {
        Self { lhs, operator, rhs }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryExpr {
    pub lhs: NonTerminalRef,
}

impl UnaryExpr {
    pub fn new(lhs: NonTerminalRef) -> Self {
        Self { lhs }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprInner {
    Unary(UnaryExpr),
    Binary(BinaryExpr),
}

type TermOrNonTerm = TerminalOrNonTerminal<Terminal, NonTerminal>;

#[allow(unused)]
fn reduce_primary(
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::Terminal(term)) = elems.pop() {
        let node = ParseTreeNode::Primary(term);
        let nt_ref = state.add_node_mut(node);

        Ok(NonTerminal::Primary(nt_ref))
    } else {
        Err("expected terminal at top of stack in reducer.".to_string())
    }
}

#[allow(unused)]
fn reduce_multiplicative_unary(
    production_id: usize,
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Primary(nt_ref))) = elems.pop() {
        let inner = ExprInner::Unary(UnaryExpr::new(nt_ref));
        let node = ParseTreeNode::Multiplicative(inner);
        let nt_ref = state.add_node_mut(node);

        Ok(NonTerminal::Multiplicative(nt_ref))
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
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Multiplicative(nt_ref))) = elems.pop() {
        let inner = ExprInner::Unary(UnaryExpr::new(nt_ref));
        let node = ParseTreeNode::Additive(inner);
        let nt_ref = state.add_node_mut(node);

        Ok(NonTerminal::Additive(nt_ref))
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
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(NonTerminal::Multiplicative(lhs_ref))), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(NonTerminal::Primary(rhs_ref)))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let non_term_kind = match op {
            Terminal::Star => BinaryOperator::Star,
            Terminal::Slash => BinaryOperator::Slash,
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        let bin_expr = BinaryExpr::new(lhs_ref, BinaryOperator::Slash, rhs_ref);
        let inner = ExprInner::Binary(bin_expr);
        let node = ParseTreeNode::Multiplicative(inner);
        let nt_ref = state.add_node_mut(node);

        Ok(NonTerminal::Multiplicative(nt_ref))
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
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    let optional_rhs = elems.pop();
    let optional_term = elems.pop();
    let optional_lhs = elems.pop();

    // reversed due to popping elements
    if let [Some(TermOrNonTerm::NonTerminal(NonTerminal::Additive(lhs_ref))), Some(TermOrNonTerm::Terminal(op)), Some(TerminalOrNonTerminal::NonTerminal(NonTerminal::Multiplicative(rhs_ref)))] =
        [optional_lhs, optional_term, optional_rhs]
    {
        let bin_op = match op {
            Terminal::Plus => BinaryOperator::Plus,
            Terminal::Minus => BinaryOperator::Minus,
            // Dispatcher should never reach this block of code due to parser guarantees.
            _ => unreachable!(),
        };

        let bin_expr = BinaryExpr::new(lhs_ref, BinaryOperator::Slash, rhs_ref);
        let inner = ExprInner::Binary(bin_expr);
        let node = ParseTreeNode::Additive(inner);
        let nt_ref = state.add_node_mut(node);

        Ok(NonTerminal::Additive(nt_ref))
    } else {
        let err_msg = format!(
            "expected 3 elements at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_expr_unary(
    production_id: usize,
    state: &mut State,
    elems: &mut Vec<TermOrNonTerm>,
) -> Result<NonTerminal, String> {
    // the only top level expr is an additive expr.
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Additive(nt_ref))) = elems.pop() {
        let unary_expr = UnaryExpr::new(nt_ref);
        let inner = ExprInner::Unary(unary_expr);
        let node = ParseTreeNode::Expr(inner);
        let nt_ref = state.add_node_mut(node);
        Ok(NonTerminal::Expr(nt_ref))
    } else {
        let err_msg = format!(
            "expected non-terminal at top of stack in production {} reducer.",
            production_id
        );

        Err(err_msg)
    }
}

#[allow(unused)]
fn reduce_goal(state: &mut State, elems: &mut Vec<TermOrNonTerm>) -> Result<NonTerminal, String> {
    if let Some(TermOrNonTerm::NonTerminal(NonTerminal::Expr(inner))) = elems.pop() {
        Ok(NonTerminal::Expr(inner))
    } else {
        Err(format!("expected Expr non-terminal at top of stack.",))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NonTerminalRef(usize);

impl NonTerminalRef {
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    arena: Vec<ParseTreeNode>,
}

impl State {
    const DEFAULT_CAPACITY: usize = 64;
}

impl State {
    fn next_nonterminal_ref(&self) -> NonTerminalRef {
        let idx = self.arena.len();

        NonTerminalRef(idx)
    }

    fn add_node_mut(&mut self, node: ParseTreeNode) -> NonTerminalRef {
        let nt_ref = self.next_nonterminal_ref();
        self.arena.push(node);

        nt_ref
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            arena: Vec::with_capacity(Self::DEFAULT_CAPACITY),
        }
    }
}

#[derive(Debug, Lr1, Clone, Copy, PartialEq)]
pub enum NonTerminal {
    #[state(State)]
    #[goal(r"<Expr>", reduce_goal)]
    #[production(r"<Additive>", |state, elems| reduce_expr_unary(2, state, elems))]
    Expr(NonTerminalRef),
    #[production(r"<Additive> Terminal::Plus <Multiplicative>", |state, elems| reduce_additive_binary(3, state, elems))]
    #[production(r"<Additive> Terminal::Minus <Multiplicative>", |state, elems| reduce_additive_binary(4, state, elems))]
    #[production(r"<Multiplicative>", |state, elems| reduce_additive_unary(5, state, elems))]
    Additive(NonTerminalRef),
    #[production(r"<Multiplicative> Terminal::Star <Primary>", |state, elems| { reduce_multiplicative_binary(6, state, elems) })]
    #[production(r"<Multiplicative> Terminal::Slash <Primary>", |state, elems| { reduce_multiplicative_binary(7, state, elems) })]
    #[production(r"<Primary>", |state, elems| reduce_multiplicative_unary(9, state, elems))]
    Multiplicative(NonTerminalRef),
    #[production(r"Terminal::Int", reduce_primary)]
    Primary(NonTerminalRef),
}

impl NonTerminalRepresentable for NonTerminal {
    type Terminal = Terminal;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseTreeNode {
    Expr(ExprInner),
    Additive(ExprInner),
    Multiplicative(ExprInner),
    Primary(Terminal),
}

fn parse_basic_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple calculator expression parsing");
    let input = "10 / 5 + 1";

    /*let expected = NonTerminal::Expr(Box::new(ExprInner::Binary(BinaryExpr::new(
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
    */

    group.bench_function("with tokenization", |b| {
        b.iter(|| {
            let mut state = State::default();

            let tokenizer = token_stream_from_input(black_box(input))
                .unwrap()
                .map(|token| token.to_variant())
                .take_while(|terminal| !matches!(&terminal, &Terminal::Eof))
                // append a single eof.
                .chain([Terminal::Eof].into_iter());

            let parse_tree = NonTerminal::parse_input(&mut state, tokenizer);

            assert!(parse_tree.is_ok())
            //assert_eq!(&parse_tree, &expected);
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
            let mut state = State::default();
            let parse_tree = NonTerminal::parse_input(&mut state, (&token_stream).iter().copied());

            assert!(parse_tree.is_ok())
            //assert_eq!(&parse_tree, &expected);
        });
    });
}

fn parse_large_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("large expression");
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

            let mut state = State::default();

            let parse_tree = NonTerminal::parse_input(&mut state, tokenizer);
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
            let mut state = State::default();
            let parse_tree = NonTerminal::parse_input(&mut state, (&token_stream).iter().copied());
            assert!(parse_tree.is_ok());
        });
    });
}

criterion_group!(benches, parse_basic_expression, parse_large_expression);
criterion_main!(benches);
