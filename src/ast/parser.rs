use nom::{IResult, Parser};
use nom_locate::{position, LocatedSpan};

use nom::{
    AsChar,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, line_ending},
    combinator::{opt, recognize},
    multi::{many0, separated_list0},
    sequence::delimited,
};

use crate::ast::{Module, Rule, Stage, Term, TermContents, Rel};

pub type Span<'a> = LocatedSpan<&'a str>;

pub fn parse_rule(s: Span) -> IResult<Span, Rule> {
    let (s, _) = position(s)?;

    let (s, _) = tag("Rule")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, name) = parse_identifier(s)?;

    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = multispace0(s)?;
    let (s, premise) = parse_term(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = multispace0(s)?;
    let (s, _) = take_while1(|c| c == '-')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = multispace0(s)?;
    let (s, conclusion) = parse_term(s)?;

    Ok((s, Rule {
        name: name.to_string(),
        premise,
        conclusion,
    }))
}

pub fn parse_stage(s: Span) -> IResult<Span, Stage> {
    let (s, _) = position(s)?;

    let (s, _) = tag("Begin Stage")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, name) = parse_identifier(s)?;

    let (s, _) = char(':')(s)?;
    let (s, _) = line_ending(s)?;

    let (s, rules) = many0(|s| {
        let (s, _) = multispace0(s)?;
        let (s, rule) = parse_rule(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, rule))
    }).parse(s)?;

    let (s, _) = multispace0(s)?;
    let (s, _) = tag("End Stage")(s)?;
    let (s, _) = multispace1(s)?;

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
    }))
}

pub fn parse_module(s: Span) -> IResult<Span, Module> {
    let (s, _) = position(s)?;
    let (s, _) = multispace0(s)?;

    let (s, _) = tag("Begin Facts:")(s)?;
    let (s, _) = line_ending(s)?;

    let (s, facts) = many0(|s| {
        let (s, _) = multispace0(s)?;
        let (s, term) = parse_term(s)?;
        let (s, _) = line_ending(s)?;
        Ok((s, term))
    }).parse(s)?;

    let (s, _) = multispace0(s)?;
    let (s, _) = tag("End Facts")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("Begin Global:")(s)?;
    let (s, _) = line_ending(s)?;

    let (s, _) = position(s)?;
    let (s, global_rules) = many0(|s| {
        let (s, _) = multispace0(s)?;
        let (s, rule) = parse_rule(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, rule))
    }).parse(s)?;

    let (s, _) = multispace0(s)?;
    let (s, _) = tag("End Global")(s)?;
    let (s, _) = multispace0(s)?;

    let global_stage = Stage {
        name: "Global".to_string(),
        rules: global_rules,
    };

    let (s, stages) = many0(|s| {
        let (s, _) = multispace0(s)?;
        let (s, stage) = parse_stage(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, stage))
    }).parse(s)?;

    Ok((s, Module {
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
    let (s, _) = multispace0(s)?;

    fn ws_term(s: Span) -> IResult<Span, Term> {
        let (s, _) = multispace0(s)?;
        let (s, term) = parse_term(s)?;
        let (s, _) = multispace0(s)?;
        Ok((s, term))
    }

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

    let rel = Rel::UserRel { name: rel_name.to_string() };

    Ok((s, Term {
        contents: TermContents::App { rel, args },
    }))
}

pub fn parse_term(s: Span) -> IResult<Span, Term> {
    alt((
        parse_float,
        parse_int,
        parse_app,
        parse_var,
        parse_atom,
    )).parse(s)
}
