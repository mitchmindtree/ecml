//! A module for abstracting common processes related to Genetic Algorithms.
//!
//! # Genetic Algorithms
//!
//! The genetic algorithm process can be described as follows:
//!
//! 1. Initialise a *Population* of *Individual*s.
//! 2. Evaluate the *Fitness* of each of the *Individual*s.
//! 3. Based on the *Fitness*, create a new generation via applying some **GeneticOperator** (e.g.
//!    Mutation, Crossover and Selection).
//! 4. If the terminal condition is met, we're done.
//! 5. GOTO 2.

use num_cpus;
use rand::{Rng, SeedableRng};
use rand::prng::XorShiftRng;
use scoped_threadpool::Pool as ThreadPool;
use std::cmp::Ordering;
use std::mem;
use std::sync::mpsc;

// Traits.

/// An **Individual** (sometimes referred to as "Phenotype") within a population.
///
/// `E` is the environment in which the individual's fitness is tested.
pub trait Individual<E>: Send + Sync {
    /// The measurement of fitness.
    type Fitness: Fitness;
    /// Evaluate the fitness of the individual within the given environment.
    fn fitness(&self, environment: &E) -> Self::Fitness;
}

/// Types representing a measurement of fitness.
pub trait Fitness: Send + Sync + PartialOrd {}

/// An operator used to guide the algorithm towards a solution.
///
/// Specifically, a genetic operator is responsible for generating individuals for the new
/// generation's population.
pub trait GeneticOperator<I, E>: Sync
where
    I: Individual<E>,
{
    /// Generate a new individual.
    fn generate_individual<R: Rng>(&self, rng: &mut R, op: GeneticOperation<I, E>) -> I;
}

// Model.

/// The simulation in which the genetic algorithm is run.
pub struct Simulation<I, E>
where
    I: Individual<E>,
{
    thread_pool: ThreadPool,
    // Stores the population alongside their fitness.
    population: Vec<(I, I::Fitness)>,
    // For collecting new individuals as they are generated.
    new_population_buffer: Vec<I>,
}

/// The context provided for a genetic operator to create a new individual.
#[derive(Copy, Clone, Debug)]
pub struct GeneticOperation<'a, I, E>
where
    I: 'a + Individual<E>,
    I::Fitness: 'a,
    E: 'a,
{
    /// The previous population along with the fitness for each individual.
    pub population: &'a [(I, I::Fitness)],
    /// The environment in which the last population's fitness was evaluated.
    pub environment: &'a E,
    /// The index of the newly generated individual within the `population` being generated.
    pub index: usize,
}

// Impls.

impl<I, E> Simulation<I, E>
where
    I: Individual<E>,
    E: Sync,
{
    /// Initialise the simulation with the initial state of the population.
    pub fn new<Is>(individuals: Is, environment: &E) -> Self
    where
        Is: IntoIterator<Item = I>,
    {
        Self::with_num_threads(individuals, environment, num_cpus::get() as _)
    }

    /// Initialise the simulation with the initial state of the population.
    ///
    /// Also allows for specifying the number of threads to use.
    pub fn with_num_threads<Is>(individuals: Is, environment: &E, num_threads: u32) -> Self
    where
        Is: IntoIterator<Item = I>,
    {
        // Use a threadpool for evaluating fitness.
        let mut thread_pool = ThreadPool::new(num_threads);

        // Calculate the fitness of the population.
        let (tx, rx) = mpsc::channel();
        thread_pool.scoped(|scoped| {
            for indv in individuals {
                let tx = tx.clone();
                scoped.execute(move || {
                    let fit = indv.fitness(environment);
                    tx.send((indv, fit)).unwrap();
                });
            }
        });
        mem::drop(tx);
        let mut population = rx.iter().collect::<Vec<_>>();

        // Sort the population by fitness.
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less));

        let new_population_buffer = Vec::with_capacity(population.len());
        Simulation {
            thread_pool,
            population,
            new_population_buffer,
        }
    }

    /// Step forward the simulation by a single generation.
    pub fn step<R, G>(&mut self, rng: &mut R, environment: &E, genetic_operator: &G)
    where
        R: Rng,
        G: GeneticOperator<I, E>,
    {
        let Simulation {
            ref mut thread_pool,
            ref mut population,
            ref mut new_population_buffer,
        } = *self;

        // 1. Generate new population.
        let (tx, rx) = mpsc::channel();
        thread_pool.scoped(|scoped| {
            let population = &*population;
            for (index, _) in population.iter().enumerate() {
                let mut rng = XorShiftRng::from_seed(rng.gen());
                let tx = tx.clone();
                scoped.execute(move || {
                    let op = GeneticOperation { population, environment, index };
                    let new = genetic_operator.generate_individual(&mut rng, op);
                    tx.send(new).unwrap();
                });
            }
        });
        mem::drop(tx);
        new_population_buffer.extend(rx.iter());

        // 2. Evaluate the fitness of the new population.
        let (tx, rx) = mpsc::channel();
        thread_pool.scoped(|scoped| {
            for indv in new_population_buffer.drain(..) {
                let tx = tx.clone();
                scoped.execute(move || {
                    let fit = indv.fitness(environment);
                    tx.send((indv, fit)).unwrap();
                });
            }
        });
        mem::drop(tx);
        population.clear();
        population.extend(rx.iter());

        // 3. Sort the population by fitness.
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less));
    }

    /// View the current generation's population and their fitness.
    ///
    /// This slice will always be sorted by its fitness.
    pub fn population(&self) -> &[(I, I::Fitness)] {
        &self.population
    }

    /// The individual with the greatest fitness.
    pub fn most_fit(&self) -> &(I, I::Fitness) {
        self.population()
            .iter()
            .last()
            .expect("must be at least one individual")
    }

    /// The individual with the worst fitness.
    pub fn least_fit(&self) -> &(I, I::Fitness) {
        self.population()
            .iter()
            .next()
            .expect("must be at least one individual")
    }
}

// Fitness

impl<T> Fitness for T where T: Send + Sync + PartialOrd {}
