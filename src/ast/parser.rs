use nom::{IResult, Parser};
use nom_locate::{position, LocatedSpan};

use nom::{
    AsChar,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1, take_till},
    character::complete::{char, digit1, multispace0, line_ending},
    combinator::{map, opt, recognize},
    multi::{many0, separated_list0},
    sequence::delimited,
};

use crate::ast::{DrawDirective, Module, Rule, Stage, Term, TermContents, Rel};

/// Skips a line comment: # followed by everything until (but not including) newline or EOF
fn skip_line_comment(s: Span) -> IResult<Span, ()> {
    let (s, _) = char('#')(s)?;
    let (s, _) = take_till(|c| c == '\n' || c == '\r')(s)?;
    Ok((s, ()))
}

/// Skips whitespace and comments (zero or more). 
/// Replaces multispace0 but also handles # comments.
fn ws0(s: Span) -> IResult<Span, ()> {
    let mut s = s;
    loop {
        // Skip any whitespace first
        let (s2, _) = multispace0(s)?;
        s = s2;
        
        // Try to skip a comment
        if let Ok((s3, _)) = skip_line_comment(s) {
            s = s3;
            // After comment, continue to skip more whitespace/comments
        } else {
            break;
        }
    }
    Ok((s, ()))
}

/// Skips whitespace and comments, requiring at least one whitespace char or comment.
/// Replaces multispace1.
fn ws1(s: Span) -> IResult<Span, ()> {
    // Must have at least one whitespace char OR a comment
    let start = s;
    let (s, _) = ws0(s)?;
    
    // Check we actually consumed something
    if s.location_offset() == start.location_offset() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Space)));
    }
    Ok((s, ()))
}

/// Skips optional inline comment at end of line (before line_ending).
/// Use this after parsing content that may have a trailing comment.
fn skip_trailing_comment(s: Span) -> IResult<Span, ()> {
    let (s, _) = take_while(|c| c == ' ' || c == '\t')(s)?;
    let (s, _) = opt(skip_line_comment).parse(s)?;
    Ok((s, ()))
}

fn user_rel(name: &str) -> Rel {
    Rel::UserRel { name: name.to_string() }
}

pub type Span<'a> = LocatedSpan<&'a str>;

pub fn parse_rule(s: Span) -> IResult<Span, Rule> {
    let (s, _) = position(s)?;

    let (s, _) = tag("Rule")(s)?;
    let (s, _) = ws1(s)?;

    let (s, name) = parse_identifier(s)?;

    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = ws0(s)?;
    let (s, premise) = parse_term(s)?;
    let (s, _) = skip_trailing_comment(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = ws0(s)?;
    let (s, _) = take_while1(|c| c == '-')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = ws0(s)?;
    let (s, conclusion) = parse_term(s)?;
    let (s, _) = skip_trailing_comment(s)?;

    Ok((s, Rule {
        name: name.to_string(),
        premise,
        conclusion,
    }))
}

fn parse_state_constraints(s: Span) -> IResult<Span, Vec<Term>> {
    let (s, _) = tag("Begin State Constraints:")(s)?;
    let (s, _) = line_ending(s)?;

    let (s, constraints) = many0(|s| {
        let (s, _) = ws0(s)?;
        let (s, term) = parse_term(s)?;
        let (s, _) = skip_trailing_comment(s)?;
        let (s, _) = line_ending(s)?;
        Ok((s, term))
    }).parse(s)?;

    let (s, _) = ws0(s)?;
    let (s, _) = tag("End State Constraints")(s)?;

    Ok((s, constraints))
}

fn is_draw_terminator(s: Span) -> bool {
    let fragment = s.fragment();
    fragment.starts_with("With")
        || fragment.starts_with("Draw")
        || fragment.starts_with("Rule")
        || fragment.starts_with("Begin")
        || fragment.starts_with("End")
}

fn parse_with_clause(s: Span) -> IResult<Span, Term> {
    let (s, _) = tag("With")(s)?;
    let (s, _) = line_ending(s)?;
    let (s, _) = ws0(s)?;
    let (s, term) = parse_term(s)?;
    let (s, _) = skip_trailing_comment(s)?;
    let (s, _) = line_ending(s)?;
    let (s, _) = ws0(s)?;
    Ok((s, term))
}

fn parse_draw_item(s: Span) -> IResult<Span, Term> {
    let (s, _) = ws0(s)?;
    if is_draw_terminator(s) {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Tag)));
    }
    let (s, term) = parse_term(s)?;
    let (s, _) = skip_trailing_comment(s)?;
    let (s, _) = line_ending(s)?;
    Ok((s, term))
}

fn parse_draw_directive(s: Span) -> IResult<Span, DrawDirective> {
    let (s, _) = ws0(s)?;
    let (s, condition) = opt(parse_with_clause).parse(s)?;
    let (s, _) = tag("Draw")(s)?;
    let (s, _) = line_ending(s)?;
    let (s, draws) = many0(parse_draw_item).parse(s)?;

    Ok((s, DrawDirective { condition, draws }))
}

pub fn parse_stage(s: Span) -> IResult<Span, Stage> {
    let (s, _) = position(s)?;

    let (s, _) = tag("Begin Stage")(s)?;
    let (s, _) = ws1(s)?;

    let (s, name) = parse_identifier(s)?;

    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, rules) = many0(|s| {
        let (s, _) = ws0(s)?;
        let (s, rule) = parse_rule(s)?;
        let (s, _) = ws0(s)?;
        Ok((s, rule))
    }).parse(s)?;

    let (s, _) = ws0(s)?;
    let (s, state_constraints) = opt(parse_state_constraints).parse(s)?;
    let state_constraints = state_constraints.unwrap_or_default();

    let (s, _) = ws0(s)?;
    let (s, draw_directives) = many0(parse_draw_directive).parse(s)?;

    let (s, _) = ws0(s)?;
    let (s, _) = tag("End Stage")(s)?;
    let (s, _) = ws1(s)?;

    let (s, end_name) = parse_identifier(s)?;

    if name != end_name {
        return Err(nom::Err::Error(nom::error::Error::new(
            s,
            nom::error::ErrorKind::Verify,
        )));
    }

    Ok((s, Stage {
        name: name.to_string(),
        rules,
        state_constraints,
        draw_directives,
    }))
}

fn parse_state_var(s: Span) -> IResult<Span, String> {
    let (s, _) = tag("StateVar")(s)?;
    let (s, _) = ws1(s)?;
    let (s, name) = parse_identifier(s)?;

    if !name.chars().next().unwrap().is_uppercase() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Verify)));
    }

    Ok((s, name.to_string()))
}

enum FactOrStateVar {
    Fact(Term),
    StateVar(String),
}

pub fn parse_module(s: Span) -> IResult<Span, Module> {
    let (s, _) = position(s)?;
    let (s, _) = ws0(s)?;

    let (s, _) = tag("Begin Facts:")(s)?;
    let (s, _) = line_ending(s)?;

    let (s, items) = many0(|s| {
        let (s, _) = ws0(s)?;
        let (s, item) = alt((
            |s| {
                let (s, sv) = parse_state_var(s)?;
                Ok((s, FactOrStateVar::StateVar(sv)))
            },
            |s| {
                let (s, term) = parse_term(s)?;
                Ok((s, FactOrStateVar::Fact(term)))
            },
        )).parse(s)?;
        let (s, _) = skip_trailing_comment(s)?;
        let (s, _) = line_ending(s)?;
        Ok((s, item))
    }).parse(s)?;

    let mut state_vars = Vec::new();
    let mut facts = Vec::new();
    for item in items {
        match item {
            FactOrStateVar::StateVar(sv) => state_vars.push(sv),
            FactOrStateVar::Fact(t) => facts.push(t),
        }
    }

    let (s, _) = ws0(s)?;
    let (s, _) = tag("End Facts")(s)?;
    let (s, _) = ws1(s)?;

    let (s, _) = tag("Begin Global:")(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = position(s)?;
    let (s, global_rules) = many0(|s| {
        let (s, _) = ws0(s)?;
        let (s, rule) = parse_rule(s)?;
        let (s, _) = ws0(s)?;
        Ok((s, rule))
    }).parse(s)?;

    let (s, _) = ws0(s)?;
    let (s, _) = tag("End Global")(s)?;
    let (s, _) = ws0(s)?;

    let global_stage = Stage {
        name: "Global".to_string(),
        rules: global_rules,
        state_constraints: Vec::new(),
        draw_directives: Vec::new(),
    };

    let (s, stages) = many0(|s| {
        let (s, _) = ws0(s)?;
        let (s, stage) = parse_stage(s)?;
        let (s, _) = ws0(s)?;
        Ok((s, stage))
    }).parse(s)?;

    Ok((s, Module {
        state_vars,
        facts,
        global_stage,
        stages,
    }))
}

fn is_alpha_or_underscore(c: char) -> bool {
    c.is_alpha() || c == '_'
}

fn is_alphanumeric_or_underscore(c: char) -> bool {
    c.is_alphanum() || c == '_'
}

fn parse_identifier(s: Span<'_>) -> IResult<Span<'_>, &str> {
    let (s, result) = recognize((
        take_while1(is_alpha_or_underscore),
        take_while(is_alphanumeric_or_underscore),
    )).parse(s)?;
    Ok((s, *result.fragment()))
}

fn parse_int(s: Span) -> IResult<Span, Term> {
    let (s, _) = position(s)?;
    let (s, sign) = opt(char('-')).parse(s)?;
    let (s, digits) = digit1(s)?;

    let val_str = if sign.is_some() {
        format!("-{}", *digits.fragment())
    } else {
        (*digits.fragment()).to_string()
    };
    let val = val_str.parse::<i32>().unwrap();

    Ok((s, Term {
        contents: TermContents::Int { val },
    }))
}

fn parse_float(s: Span) -> IResult<Span, Term> {
    let (s, _) = position(s)?;
    let (s, sign) = opt(char('-')).parse(s)?;
    let (s, (int_part, _, frac_part)) = (digit1, char('.'), digit1).parse(s)?;

    let val_str = if sign.is_some() {
        format!("-{}.{}", *int_part.fragment(), *frac_part.fragment())
    } else {
        format!("{}.{}", *int_part.fragment(), *frac_part.fragment())
    };
    let val = val_str.parse::<f32>().unwrap();

    Ok((s, Term {
        contents: TermContents::Float { val },
    }))
}

fn parse_var(s: Span) -> IResult<Span, Term> {
    let (s, _) = position(s)?;
    let (s, name) = parse_identifier(s)?;

    if !name.chars().next().unwrap().is_uppercase() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Verify)));
    }

    Ok((s, Term {
        contents: TermContents::Var { name: name.to_string() },
    }))
}

fn parse_atom(s: Span) -> IResult<Span, Term> {
    let (s, _) = position(s)?;
    let (s, text) = parse_identifier(s)?;

    if !text.chars().next().unwrap().is_lowercase() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Verify)));
    }

    Ok((s, Term {
        contents: TermContents::Atom { text: text.to_string() },
    }))
}

fn parse_app(s: Span) -> IResult<Span, Term> {
    let (s, _) = position(s)?;
    let (s, rel_name) = parse_identifier(s)?;
    let (s, _) = ws0(s)?;

    fn ws_term(s: Span) -> IResult<Span, Term> {
        let (s, _) = ws0(s)?;
        let (s, term) = parse_term(s)?;
        let (s, _) = ws0(s)?;
        Ok((s, term))
    }

    fn ws_comma(s: Span) -> IResult<Span, char> {
        let (s, _) = ws0(s)?;
        let (s, c) = char(',')(s)?;
        let (s, _) = ws0(s)?;
        Ok((s, c))
    }

    let (s, args) = delimited(
        char('('),
        separated_list0(ws_comma, ws_term),
        char(')'),
    ).parse(s)?;

    let rel = Rel::UserRel { name: rel_name.to_string() };

    Ok((s, Term {
        contents: TermContents::App { rel, args },
    }))
}

fn parse_paren_term(s: Span) -> IResult<Span, Term> {
    let (s, _) = char('(')(s)?;
    let (s, _) = ws0(s)?;
    let (s, term) = parse_term(s)?;
    let (s, _) = ws0(s)?;
    let (s, _) = char(')')(s)?;
    Ok((s, term))
}

fn parse_primary(s: Span) -> IResult<Span, Term> {
    alt((
        parse_float,
        parse_int,
        parse_app,
        parse_var,
        parse_atom,
        parse_paren_term,
    )).parse(s)
}

fn parse_not_prefix(s: Span) -> IResult<Span, ()> {
    let (s, _) = alt((char('¬'), char('!'))).parse(s)?;
    let (s, _) = ws0(s)?;
    Ok((s, ()))
}

fn parse_unary(s: Span) -> IResult<Span, Term> {
    let (s, nots) = many0(parse_not_prefix).parse(s)?;
    let (s, mut term) = parse_primary(s)?;

    for _ in 0..nots.len() {
        term = Term {
            contents: TermContents::App {
                rel: user_rel("not"),
                args: vec![term],
            },
        };
    }

    Ok((s, term))
}

fn parse_cmp_op(s: Span) -> IResult<Span, Rel> {
    alt((
        map(tag(".=="), |_| user_rel("real_eq")),
        map(tag(".<="), |_| user_rel("real_le")),
        map(tag(".>="), |_| user_rel("real_ge")),
        map(tag(".<"), |_| user_rel("real_lt")),
        map(tag(".>"), |_| user_rel("real_gt")),
        map(tag("=="), |_| user_rel("int_eq")),
        map(tag("<="), |_| user_rel("int_le")),
        map(tag(">="), |_| user_rel("int_ge")),
        map(tag("<"), |_| user_rel("int_lt")),
        map(tag(">"), |_| user_rel("int_gt")),
    )).parse(s)
}

fn parse_eq_op(s: Span) -> IResult<Span, Rel> {
    let (s, _) = char('=')(s)?;
    // Make sure this isn't `==` (int_eq)
    if s.fragment().starts_with('=') {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Char)));
    }
    Ok((s, user_rel("eq")))
}

fn parse_eq(s: Span) -> IResult<Span, Term> {
    let (mut s, mut left) = parse_unary(s)?;
    loop {
        let (s2, _) = ws0(s)?;
        if let Ok((s3, rel)) = parse_eq_op(s2) {
            let (s4, _) = ws0(s3)?;
            let (s5, right) = parse_unary(s4)?;
            left = Term {
                contents: TermContents::App {
                    rel,
                    args: vec![left, right],
                },
            };
            s = s5;
        } else {
            break;
        }
    }
    Ok((s, left))
}

fn parse_cmp(s: Span) -> IResult<Span, Term> {
    let (mut s, mut left) = parse_eq(s)?;
    loop {
        let (s2, _) = ws0(s)?;
        if let Ok((s3, rel)) = parse_cmp_op(s2) {
            let (s4, _) = ws0(s3)?;
            let (s5, right) = parse_eq(s4)?;
            left = Term {
                contents: TermContents::App {
                    rel,
                    args: vec![left, right],
                },
            };
            s = s5;
        } else {
            break;
        }
    }
    Ok((s, left))
}

fn parse_and_op(s: Span) -> IResult<Span, Rel> {
    let (s, _) = alt((char('∧'), char('&'))).parse(s)?;
    Ok((s, user_rel("and")))
}

fn parse_and(s: Span) -> IResult<Span, Term> {
    let (mut s, mut left) = parse_cmp(s)?;
    loop {
        let (s2, _) = ws0(s)?;
        if let Ok((s3, rel)) = parse_and_op(s2) {
            let (s4, _) = ws0(s3)?;
            let (s5, right) = parse_cmp(s4)?;
            left = Term {
                contents: TermContents::App {
                    rel,
                    args: vec![left, right],
                },
            };
            s = s5;
        } else {
            break;
        }
    }
    Ok((s, left))
}

fn parse_or_op(s: Span) -> IResult<Span, Rel> {
    let (s, _) = alt((char('∨'), char('|'))).parse(s)?;
    Ok((s, user_rel("or")))
}

fn parse_or(s: Span) -> IResult<Span, Term> {
    let (mut s, mut left) = parse_and(s)?;
    loop {
        let (s2, _) = ws0(s)?;
        if let Ok((s3, rel)) = parse_or_op(s2) {
            let (s4, _) = ws0(s3)?;
            let (s5, right) = parse_and(s4)?;
            left = Term {
                contents: TermContents::App {
                    rel,
                    args: vec![left, right],
                },
            };
            s = s5;
        } else {
            break;
        }
    }
    Ok((s, left))
}

pub fn parse_term(s: Span) -> IResult<Span, Term> {
    parse_or(s)
}
