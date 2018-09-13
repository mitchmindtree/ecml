//! Demonstrates using genetic programming to evolve a function that can predict the next ascii
//! character, given *n* preceding characters.
//!
//! 1. What is the "Terminal Set"?
//!
//! - The list of *n* preceding characters
//! - Some random constants in the range 0..256.
//!
//! 2. What is the "Function Set"?
//!
//! - `*` `/` `+` `-` '%'
//! - `count` (the number of elements ina list)
//! - `sum`
//! - `product`
//! - `nth`
//!
//! 3. What is the "Fitness Measure"?
//!
//! - The number of correct guesses out of the total number of guesses.
//! - The simplicity of the program.
//!
//! 4. What are the "Control Parameters"?
//!
//! - 100 guesses per individual.
//! - 1000 individuals.
//! - 100 generations.
//!
//! 5. What is the "Termination Criterion"?
//!
//! - The best guessing program after all generations OR
//! - The first program to guess 100% of characters.

extern crate ecml;
extern crate rand;
extern crate walkdir;

use ecml::{ga, gp};
use rand::{Rng, SeedableRng};
use rand::prng::XorShiftRng;
use std::collections::HashMap;
use std::fs;
use walkdir::WalkDir;

// Constants.

const MIN_NODE_COUNT: usize = 2;
const EXPRESSION_DEPTH: u32 = 3;
const PRECEDING_CHARS: usize = 6;
const RANDOM_CONSTANTS: usize = VALID_CHARS as _;
const GUESSES_PER_INDIVIDUAL: usize = 1_000;
const CHARS_TO_GUESS: usize = PRECEDING_CHARS;
const INDIVIDUALS_PER_GENERATION: usize = 100;
const GENERATIONS: usize = 10;
const HARRY_POTTER_PATH: &'static str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
const VALID_CHARS: u8 = 31;

// Model.

/// A simplified set of characters that we use for prediction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Char {
    A = 0,
    E = 1,
    I = 2,
    O = 3,
    U = 4,
    B = 5,
    C = 6,
    D = 7,
    F = 8,
    G = 9,
    H = 10,
    J = 11,
    K = 12,
    L = 13,
    M = 14,
    N = 15,
    P = 16,
    Q = 17,
    R = 18,
    S = 19,
    T = 20,
    V = 21,
    W = 22,
    X = 23,
    Y = 24,
    Z = 25,
    Dot = 26,
    Exclamation = 27,
    Comma = 28,
    Space = 29,
    Newline = 30,
}

/// The three categories of character types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum CharType {
    Vowel,
    Consonant,
    Other,
}

/// The environment in which an expression's fitness is tested.
struct Environment<'a> {
    char_weights: &'a HashMap<Char, f32>,
    char_ty_weights: &'a HashMap<CharType, f32>,
    data: Vec<&'a [Char]>,
}

/// The process by which new individuals are selected for a new generation.
struct GeneticOperator;

/// The possible leaves of the AST.
#[derive(Clone, Debug)]
enum Terminal {
    /// An index into the slice of preceding chars.
    PrecedingCharacterIndex(usize),
    /// A random constant within the char byte range.
    RandomConstant(Char),
}

/// Operation functions.
#[derive(Clone, Debug)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

#[derive(Clone, Debug)]
enum Condition {
    IfGtElse,
    IfLtElse,
}

/// Functions that can composed together within the generated guessing program.
#[derive(Clone, Debug)]
enum Function {
    Operator(Operator),
    Condition(Condition),
}

/// The node type used within the expression tree.
#[derive(Clone, Debug)]
enum Node {
    Terminal(Terminal),
    Function(Function),
}

/// The tree type representing the guessing program.
type Expr = gp::expr::DiGraph<Node>;

// Arity impls.

impl gp::expr::gen::Arity for Operator {
    fn arity(&self) -> u32 {
        2
    }
}

impl gp::expr::gen::Arity for Condition {
    fn arity(&self) -> u32 {
        4
    }
}

impl gp::expr::gen::Arity for Function {
    fn arity(&self) -> u32 {
        match *self {
            Function::Operator(ref f) => f.arity(),
            Function::Condition(ref f) => f.arity(),
        }
    }
}

impl gp::expr::gen::Arity for Node {
    fn arity(&self) -> u32 {
        match *self {
            Node::Function(ref f) => f.arity(),
            Node::Terminal(_) => 0,
        }
    }
}

// Function impls.

impl gp::expr::gen::Function for Operator {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        match rng.gen_range(0, 6) {
            0 => Operator::Add,
            1 => Operator::Sub,
            2 => Operator::Mul,
            3 => Operator::Div,
            4 => Operator::Rem,
            5 => Operator::Pow,
            _ => unreachable!(),
        }
    }
}

impl gp::expr::gen::Function for Condition {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        match rng.gen_range(0, 2) {
            0 => Condition::IfGtElse,
            1 => Condition::IfLtElse,
            _ => unreachable!(),
        }
    }
}

impl gp::expr::gen::Function for Function {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        match rng.gen_range(0, 2) {
            0 => Function::Operator(Operator::generate(rng)),
            1 => Function::Condition(Condition::generate(rng)),
            _ => unreachable!(),
        }
    }
}

impl gp::expr::gen::Function for Node {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        Node::Function(Function::generate(rng))
    }
}

// Terminal impls.

impl gp::expr::gen::Terminal for Terminal {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        let n = rng.gen_range(0, PRECEDING_CHARS + RANDOM_CONSTANTS);
        if n < PRECEDING_CHARS {
            Terminal::PrecedingCharacterIndex(n)
        } else {
            let byte = rng.gen_range(0, VALID_CHARS);
            let ch = Char::from_byte(byte).unwrap();
            Terminal::RandomConstant(ch)
        }
    }
}

impl gp::expr::gen::Terminal for Node {
    fn generate<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        Node::Terminal(Terminal::generate(rng))
    }
}

// Evaluate impl.

impl<'a> gp::expr::Evaluate<&'a [Char]> for Node {
    type Value = Char;
    fn evaluate(&self, inputs: &[&Self::Value], seq: &&'a [Char]) -> Self::Value {
        fn u32_to_char(u: u32) -> Char {
            let byte = (u.wrapping_add(VALID_CHARS as u32) % VALID_CHARS as u32) as u8;
            Char::from_byte(byte)
                .unwrap_or_else(|| panic!("could not convert {} to Char", byte))
        }

        match *self {
            Node::Terminal(ref t) => match *t {
                Terminal::PrecedingCharacterIndex(ix) => seq[ix],
                Terminal::RandomConstant(ch) => ch,
            }
            Node::Function(ref f) => match *f {
                Function::Operator(ref op) => {
                    let a = *inputs[0] as u8 as u32;
                    let b = *inputs[1] as u8 as u32;
                    let u = match *op {
                        Operator::Add => a.wrapping_add(b),
                        Operator::Sub => a.wrapping_sub(b),
                        Operator::Mul => a.wrapping_mul(b),
                        Operator::Div => if b == 0 { 1 } else { a / b },
                        Operator::Rem => a % if b == 0 { 1 } else { b },
                        Operator::Pow => {
                            match (a, b) {
                                (0, _) => 0,
                                (_, 0) => 1,
                                (a, 1) => a,
                                _ => {
                                    let mut u = a;
                                    for _ in 2..b {
                                        u = u.wrapping_mul(a);
                                    }
                                    u
                                }
                            }
                        }
                    };
                    u32_to_char(u)
                }
                Function::Condition(ref cond) => {
                    let a = *inputs[0] as u8 as u32;
                    let b = *inputs[1] as u8 as u32;
                    let c = *inputs[2] as u8 as u32;
                    let d = *inputs[3] as u8 as u32;
                    let u = match *cond {
                        Condition::IfGtElse => if a > b { c } else { d }
                        Condition::IfLtElse => if a < b { c } else { d }
                    };
                    u32_to_char(u)
                }
            }
        }
    }
}

// Implementations.

impl Char {
    fn from_char(ch: char) -> Option<Char> {
        let ch = match ch {
            'a' | 'A' => Char::A,
            'b' | 'B' => Char::B,
            'c' | 'C' => Char::C,
            'd' | 'D' => Char::D,
            'e' | 'E' => Char::E,
            'f' | 'F' => Char::F,
            'g' | 'G' => Char::G,
            'h' | 'H' => Char::H,
            'i' | 'I' => Char::I,
            'j' | 'J' => Char::J,
            'k' | 'K' => Char::K,
            'l' | 'L' => Char::L,
            'm' | 'M' => Char::M,
            'n' | 'N' => Char::N,
            'o' | 'O' => Char::O,
            'p' | 'P' => Char::P,
            'q' | 'Q' => Char::Q,
            'r' | 'R' => Char::R,
            's' | 'S' => Char::S,
            't' | 'T' => Char::T,
            'u' | 'U' => Char::U,
            'v' | 'V' => Char::V,
            'w' | 'W' => Char::W,
            'x' | 'X' => Char::X,
            'y' | 'Y' => Char::Y,
            'z' | 'Z' => Char::Z,
            '.' => Char::Dot,
            '!' => Char::Exclamation,
            ',' => Char::Comma,
            ' ' => Char::Space,
            '\n' | '\r' => Char::Newline,
            _ => return None,
        };
        Some(ch)
    }

    fn from_byte(byte: u8) -> Option<Self> {
        let ch = match byte {
            0 => Char::A,
            1 => Char::E,
            2 => Char::I,
            3 => Char::O,
            4 => Char::U,
            5 => Char::B,
            6 => Char::C,
            7 => Char::D,
            8 => Char::F,
            9 => Char::G,
            10 => Char::H,
            11 => Char::J,
            12 => Char::K,
            13 => Char::L,
            14 => Char::M,
            15 => Char::N,
            16 => Char::P,
            17 => Char::Q,
            18 => Char::R,
            19 => Char::S,
            20 => Char::T,
            21 => Char::V,
            22 => Char::W,
            23 => Char::X,
            24 => Char::Y,
            25 => Char::Z,
            26 => Char::Dot,
            27 => Char::Exclamation,
            28 => Char::Comma,
            29 => Char::Space,
            30 => Char::Newline,
            _ => return None,
        };
        Some(ch)
    }

    fn to_char_lowercase(&self) -> char {
        match *self {
            Char::A => 'a',
            Char::B => 'b',
            Char::C => 'c',
            Char::D => 'd',
            Char::E => 'e',
            Char::F => 'f',
            Char::G => 'g',
            Char::H => 'h',
            Char::I => 'i',
            Char::J => 'j',
            Char::K => 'k',
            Char::L => 'l',
            Char::M => 'm',
            Char::N => 'n',
            Char::O => 'o',
            Char::P => 'p',
            Char::Q => 'q',
            Char::R => 'r',
            Char::S => 's',
            Char::T => 't',
            Char::U => 'u',
            Char::V => 'v',
            Char::W => 'w',
            Char::X => 'x',
            Char::Y => 'y',
            Char::Z => 'z',
            Char::Dot => '.' ,
            Char::Exclamation => '!',
            Char::Comma => ',',
            Char::Space => ' ',
            Char::Newline => '\n',
        }
    }

    fn to_type(&self) -> CharType {
        match *self {
            Char::A |
            Char::E |
            Char::I |
            Char::O |
            Char::U => CharType::Vowel,
            Char::B |
            Char::C |
            Char::D |
            Char::F |
            Char::G |
            Char::H |
            Char::J |
            Char::K |
            Char::L |
            Char::M |
            Char::N |
            Char::P |
            Char::Q |
            Char::R |
            Char::S |
            Char::T |
            Char::V |
            Char::W |
            Char::X |
            Char::Y |
            Char::Z => CharType::Consonant,
            Char::Dot |
            Char::Exclamation |
            Char::Comma |
            Char::Space |
            Char::Newline => CharType::Other,
        }
    }
}

impl<'a> Environment<'a> {
    fn generate<R: Rng>(
        rng: &mut R,
        data: &'a [Char],
        char_weights: &'a HashMap<Char, f32>,
        char_ty_weights: &'a HashMap<CharType, f32>,
    ) -> Self {
        // Collect the random data slices.
        let data = (0..GUESSES_PER_INDIVIDUAL)
            .map(|_| random_data_slice(rng, &data))
            .collect::<Vec<_>>();
        Environment { char_weights, char_ty_weights, data }
    }
}

impl<'a> ga::GeneticOperator<Expr, Environment<'a>> for GeneticOperator {
    fn generate_individual<R: Rng>(
        &self,
        rng: &mut R,
        op: ga::GeneticOperation<Expr, Environment<'a>>,
    ) -> Expr {
        // Select the process type from:
        enum Kind { Select, Crossover, Mutate, Random }

        const SELECT_WEIGHT: f32 = 0.1;
        const CROSSOVER_WEIGHT: f32 = 0.4;
        const MUTATE_WEIGHT: f32 = 0.2;
        const RANDOM_WEIGHT: f32 = 1.0 - CROSSOVER_WEIGHT - SELECT_WEIGHT - MUTATE_WEIGHT;

        let random_index = (RANDOM_WEIGHT * op.population.len() as f32) as usize;
        let mutate_index =
            (random_index as f32 + MUTATE_WEIGHT * op.population.len() as f32) as usize;
        let crossover_index =
            (mutate_index as f32 + CROSSOVER_WEIGHT * op.population.len() as f32) as usize;

        // The kind of reproduction used to generate this individual.
        let kind = if op.index < random_index {
            Kind::Random
        } else if op.index < mutate_index {
            Kind::Mutate
        } else if op.index < crossover_index {
            Kind::Crossover
        } else {
            Kind::Select
        };

        match kind {
            Kind::Random => gp::expr::gen(rng, EXPRESSION_DEPTH),
            Kind::Mutate => {
                let a = &op.population[op.index];
                let b = gp::expr::gen(rng, EXPRESSION_DEPTH);
                crossover_exprs(rng, &a.0, &b)
            }
            Kind::Crossover => {
                let a = &op.population[op.index];
                let b_index = rng.gen_range(op.index, op.population.len());
                let b = &op.population[b_index];
                crossover_exprs(rng, &a.0, &b.0)
            }
            Kind::Select => op.population[op.index].0.clone(),
        }
    }
}

fn crossover_exprs<R>(rng: &mut R, a: &Expr, b: &Expr) -> Expr
where
    R: Rng,
{
    let a_len = a.node_count();
    let b_len = b.node_count();
    let a_crossover = gp::expr::NodeIndex::new(rng.gen_range(0, a_len));
    let b_crossover = gp::expr::NodeIndex::new(rng.gen_range(0, b_len));

    let b_subexpr = gp::expr::clone_subtree(b, b_crossover);
    let mut new_expr = gp::expr::replace_subtree(&a, a_crossover, b_subexpr);
    gp::expr::trim_tree_to_depth(rng, &mut new_expr, EXPRESSION_DEPTH);
    new_expr
}

impl<'a> ga::Individual<Environment<'a>> for Expr {
    type Fitness = f32;
    fn fitness(&self, env: &Environment<'a>) -> Self::Fitness {
        // Sum up the prediction score and check for repetitions.
        let mut score = 0.0;
        let mut rep_count = 0;
        let mut last_ch = Char::from_char(' ').unwrap();
        for slice in &env.data {
            let mut guesses = [(None, None); CHARS_TO_GUESS];
            for (i, guess) in guesses.iter_mut().enumerate() {
                let preceding = &slice[i..slice.len() - (CHARS_TO_GUESS - i)];
                let expected = *slice
                    .iter()
                    .last()
                    .expect("must be at least one char per training slice");

                // Guess the next character.
                let eval = gp::expr::eval(self, &preceding);
                assert!(!eval.is_empty(), "eval map for expr was empty:\n{:#?}", self);
                let ch = eval[&gp::expr::NodeIndex::new(0)];

                // Penalise frequent repetitions.
                if ch == last_ch {
                    rep_count += 1;
                    if rep_count > 3 {
                        score -= rep_count as f32;
                    }
                } else {
                    rep_count = 0;
                }
                last_ch = ch;

                // If the character type is correct, score.
                let char_ty = ch.to_type();
                if expected.to_type() == char_ty {
                    guess.0 = Some(char_ty);
                }

                // If the character itself was correct, score.
                if expected == ch {
                    guess.1 = Some(ch);
                }
            }

            // Add score for 1. n consecutive correct guesses and 2. correct guesses.
            let mut n_consecutive_correct = 0usize;
            let mut last_correct = false;
            for &(ty, ch) in &guesses {
                match ch {
                    None => last_correct = false,
                    Some(ch) => {
                        if last_correct {
                            n_consecutive_correct += 1;
                        }
                        last_correct = true;
                        score += env.char_weights[&ch];
                    }
                }
                if let Some(ty) = ty {
                    score += env.char_ty_weights[&ty] * 0.2;
                }
            }
            score += n_consecutive_correct.pow(2) as f32;
        }

        // Determine the weight based on tree size.
        let size_weight = if self.node_count() < MIN_NODE_COUNT {
            0.1
        } else {
            1.0
        };

        size_weight * score
    }
}

// fns

// A function for retrieving a random data slice.
fn random_data_slice<'a, 'b, R>(rng: &'a mut R, data: &'b [Char]) -> &'b [Char]
where
    R: Rng,
{
    let n_chars = PRECEDING_CHARS + CHARS_TO_GUESS; // preceding chars + 1 for testing.
    let start = rng.gen_range(0, data.len() - n_chars);
    let end = start + n_chars;
    let range = start..end;
    &data[range]
}

// Exe.

fn main() {
    // The file or directory from which the data should be loaded.
    let path = match std::env::args().nth(1) {
        Some(path) => std::path::PathBuf::from(path),
        None => std::path::PathBuf::from(HARRY_POTTER_PATH),
    };

    // A random number generator with a unique seed.
    let seed = rand::random();
    let mut rng = XorShiftRng::from_seed(seed);
    println!("RNG seed: {:?}", seed);

    println!("Collecting ASCII text data from {:?}", path);

    // Load the data into one long utf8 string.
    let mut data = Vec::new();
    // data.extend((0..10_000).map(|_| rng.gen::<u8>()).filter_map(Char::from_byte));
    if path.is_dir() {
        for entry in WalkDir::new(&path) {
            let entry = entry.unwrap();
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let mut string = fs::read_to_string(path).expect("failed to read file to utf8 string");
            string.push_str("\n");
            let chars = string.chars().filter_map(Char::from_char);
            data.extend(chars);
        }
    } else {
        let string = fs::read_to_string(path).expect("failed to read file to utf8 string");
        let chars = string.chars().filter_map(Char::from_char);
        data.extend(chars);
    };

    println!("Measuring character frequency...");

    // Collect the letter frequencies. We want to make sure that the GA does not just optimise for
    // the most frequently appearing letter.
    let mut char_occurrences = HashMap::new();
    let mut char_ty_occurrences = HashMap::new();
    for &ch in data.iter() {
        *char_occurrences.entry(ch).or_insert(0) += 1;
        *char_ty_occurrences.entry(ch.to_type()).or_insert(0) += 1;
    }
    let total_chars = data.len();
    let char_weights = char_occurrences
        .into_iter()
        .map(|(ch, occurrences)| (ch, 1.0 - occurrences as f32 / total_chars as f32))
        .collect::<HashMap<_, _>>();
    let char_ty_weights = char_ty_occurrences
        .into_iter()
        .map(|(ty, occurrences)| (ty, 1.0 - occurrences as f32 / total_chars as f32))
        .collect::<HashMap<_, _>>();

    println!("Initialising simulation...");

    // Initialise the simulation.
    let mut simulation = {
        let pop = (0..INDIVIDUALS_PER_GENERATION)
            .map(|_| gp::expr::gen(&mut rng, EXPRESSION_DEPTH))
            .collect::<Vec<_>>();
        let env = &Environment::generate(&mut rng, &data, &char_weights, &char_ty_weights);
        let n_threads = 1;
        ga::Simulation::with_num_threads(pop, env, n_threads)
    };

    println!("Running simulation...");

    // Fun the simulation.
    let start = std::time::Instant::now();
    let mut last = start;
    for g in 0..GENERATIONS {
        // Step the simulation forward one generation.
        {
            let env = &Environment::generate(&mut rng, &data, &char_weights, &char_ty_weights);
            simulation.step(&mut rng, env, &GeneticOperator);
        }

        // The duration taken by the generation.
        let now = std::time::Instant::now();
        let interval = now.duration_since(last);
        last = now;

        let &(_, greatest) = simulation.most_fit();
        let sum: f32 = simulation
            .population()
            .iter()
            .map(|&(_, f)| f)
            .sum();
        let average = sum / simulation.population().len() as f32;
        println!("Generation {}: Greatest: {:?}, Average: {:?}, Duration: {:?}",
                 g, greatest, average, interval);
    }
    println!("Total duration: {:?}", std::time::Instant::now().duration_since(start));

    let &(ref fittest, fitness) = simulation.most_fit();
    println!("Fittest expression ({}):\n{:#?}", fitness, fittest);

    // Generate text.
    let mut slice = random_data_slice(&mut rng, &data).to_vec();
    let mut text: Vec<Char> = slice.iter().cloned().collect();
    for _ in 0..1_000 {
        let eval = gp::expr::eval(fittest, &&slice[..]);
        let next_char = eval[&gp::expr::NodeIndex::new(0)];
        text.push(next_char);
        slice.push(next_char);
        slice.remove(0);
    }
    let source_text = text[..slice.len()]
        .iter()
        .map(Char::to_char_lowercase)
        .collect::<String>();
    let generated_text = text[slice.len()..]
        .iter()
        .map(Char::to_char_lowercase)
        .collect::<String>();
    println!("Source text: {:?}", source_text);
    println!("<begin generated text>\n{}\n<end generated text>", generated_text);
}
