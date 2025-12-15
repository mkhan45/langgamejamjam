use nom::{IResult, Parser};
use nom_locate::{position, LocatedSpan};

use nom::{
    AsChar,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, line_ending},
    combinator::{opt, recognize},
    multi::{many0, many1, separated_list0},
    sequence::delimited,
};

use crate::ast::{Module, Rule, Stage, Term, TermContents, Rel};

type Span<'a> = LocatedSpan<&'a str>;

pub fn parse_rule(s: Span) -> IResult<Span, Rule> {
    let (s, start_pos) = position(s)?;

    // Parse "Rule" keyword
    let (s, _) = tag("Rule")(s)?;
    let (s, _) = multispace1(s)?;

    // Parse rule name
    let (s, name) = parse_identifier(s)?;

    // Parse ":"
    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    // Parse premise (with optional leading whitespace)
    let (s, _) = multispace0(s)?;
    let (s, premise) = parse_term(s)?;
    let (s, _) = line_ending(s)?;

    // Parse divider line (3+ dashes with optional whitespace)
    let (s, _) = multispace0(s)?;
    let (s, _) = take_while1(|c| c == '-')(s)?;
    let (s, _) = line_ending(s)?;

    // Parse conclusion (with optional leading whitespace)
    let (s, _) = multispace0(s)?;
    let (s, conclusion) = parse_term(s)?;

    Ok((s, Rule {
        span: start_pos,
        name,
        premise,
        conclusion,
    }))
}

pub fn parse_stage(s: Span) -> IResult<Span, Stage> {
    let (s, start_pos) = position(s)?;

    // Parse "Begin Stage" keyword
    let (s, _) = tag("Begin Stage")(s)?;
    let (s, _) = multispace1(s)?;

    // Parse stage name
    let (s, name) = parse_identifier(s)?;

    // Parse ":"
    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    // Parse zero or more rules (with optional whitespace between them)
    let (s, rules) = many0(|s| {
        let (s, _) = multispace0(s)?;
        let (s, rule) = parse_rule(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, rule))
    }).parse(s)?;

    // Parse "End Stage" keyword
    let (s, _) = multispace0(s)?;
    let (s, _) = tag("End Stage")(s)?;
    let (s, _) = multispace1(s)?;

    // Parse stage name again (should match the opening name)
    let (s, end_name) = parse_identifier(s)?;

    // Verify the names match
    if name != end_name {
        return Err(nom::Err::Error(nom::error::Error::new(
            s,
            nom::error::ErrorKind::Verify,
        )));
    }

    Ok((s, Stage {
        span: start_pos,
        name,
        rules,
    }))
}

pub fn parse_module(s: Span) -> IResult<Span, Module> {
    let (s, start_pos) = position(s)?;

    // Parse zero or more stages (with optional whitespace between them)
    let (s, stages) = many1(|s| {
        let (s, _) = multispace0(s)?;
        let (s, stage) = parse_stage(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, stage))
    }).parse(s)?;

    Ok((s, Module {
        span: start_pos,
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
    let (s, start_pos) = position(s)?;
    let (s, sign) = opt(char('-')).parse(s)?;
    let (s, digits) = digit1(s)?;

    let val_str = if sign.is_some() {
        format!("-{}", *digits.fragment())
    } else {
        (*digits.fragment()).to_string()
    };
    let val = val_str.parse::<i32>().unwrap();

    Ok((s, Term {
        span: start_pos,
        contents: TermContents::Int { val },
    }))
}

fn parse_float(s: Span) -> IResult<Span, Term> {
    let (s, start_pos) = position(s)?;
    let (s, sign) = opt(char('-')).parse(s)?;
    let (s, (int_part, _, frac_part)) = (digit1, char('.'), digit1).parse(s)?;

    let val_str = if sign.is_some() {
        format!("-{}.{}", *int_part.fragment(), *frac_part.fragment())
    } else {
        format!("{}.{}", *int_part.fragment(), *frac_part.fragment())
    };
    let val = val_str.parse::<f32>().unwrap();

    Ok((s, Term {
        span: start_pos,
        contents: TermContents::Float { val },
    }))
}

fn parse_var(s: Span) -> IResult<Span, Term> {
    let (s, start_pos) = position(s)?;
    let (s, name) = parse_identifier(s)?;

    // Check if first character is uppercase
    if !name.chars().next().unwrap().is_uppercase() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Verify)));
    }

    Ok((s, Term {
        span: start_pos,
        contents: TermContents::Var { name },
    }))
}

fn parse_atom(s: Span) -> IResult<Span, Term> {
    let (s, start_pos) = position(s)?;
    let (s, text) = parse_identifier(s)?;

    // Check if first character is lowercase
    if !text.chars().next().unwrap().is_lowercase() {
        return Err(nom::Err::Error(nom::error::Error::new(s, nom::error::ErrorKind::Verify)));
    }

    Ok((s, Term {
        span: start_pos,
        contents: TermContents::Atom { text },
    }))
}

fn parse_app(s: Span) -> IResult<Span, Term> {
    let (s, start_pos) = position(s)?;
    let (s, rel_name) = parse_identifier(s)?;
    let (s, _) = multispace0(s)?;

    // Helper parser for a term surrounded by whitespace
    fn ws_term(s: Span) -> IResult<Span, Term> {
        let (s, _) = multispace0(s)?;
        let (s, term) = parse_term(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, term))
    }

    // Helper parser for comma separator surrounded by whitespace
    fn ws_comma(s: Span) -> IResult<Span, char> {
        let (s, _) = multispace0(s)?;
        let (s, c) = char(',')(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, c))
    }

    let (s, args) = delimited(
        char('('),
        separated_list0(ws_comma, ws_term),
        char(')'),
    ).parse(s)?;

    // For now, treat all relations as UserRel
    let rel = Rel::UserRel { name: rel_name };

    Ok((s, Term {
        span: start_pos,
        contents: TermContents::App { rel, args },
    }))
}

pub fn parse_term(s: Span) -> IResult<Span, Term> {
    alt((
        parse_float,  // Try float before int (since float contains digits too)
        parse_int,
        parse_app,    // Try app before var/atom (since it starts with identifier)
        parse_var,
        parse_atom,
    )).parse(s)
}
