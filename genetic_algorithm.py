"""
genetic_algorithm.py
--------------------
Pure-Python / NumPy genetic algorithm for optimal EV-charging-station placement.

Chromosome representation
  A chromosome is a list of `n_stations` (row, col) integer tuples
  representing grid indices inside the density matrix.

Fitness function
  We want stations to be placed where population density is HIGH and
  where each high-density cell is *covered* by at least one nearby station.

  fitness(chromosome) = Σ_{cell} density[cell] * exp(-d_min(cell) / scale)

  where d_min(cell) is the Euclidean grid distance to the nearest station.
  Higher fitness ← stations closer to dense cells.

GA operators
  Selection  : tournament selection (size 3)
  Crossover  : uniform crossover on the list of station indices
  Mutation   : random re-position of a randomly chosen station
  Elitism    : top-k individuals always survive unchanged
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Station  = Tuple[int, int]          # (row, col)
Chromosome = List[Station]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    n_stations:      int   = 10     # number of EV stations to place
    population_size: int   = 50     # number of candidate solutions
    n_generations:   int   = 100    # total evolution cycles
    mutation_rate:   float = 0.15   # probability of mutating each station
    elite_fraction:  float = 0.10   # fraction of top individuals kept each gen
    tournament_size: int   = 3      # tournament selection pool size
    coverage_scale:  float = 15.0   # grid-cell "reach" of one station (σ)
    seed:            int | None = 42


# ---------------------------------------------------------------------------
# Distance-based coverage kernel
# ---------------------------------------------------------------------------
def _build_coverage_map(
    stations: Chromosome,
    grid_size: int,
    scale: float,
) -> np.ndarray:
    """
    Return a (grid_size, grid_size) float array where each cell value is
    the maximum coverage contribution from any station:

        coverage[r,c] = max_i exp( -||cell - station_i||² / (2·scale²) )
    """
    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    R, C = np.meshgrid(rows, cols, indexing="ij")  # (G, G)

    coverage = np.zeros((grid_size, grid_size), dtype=np.float64)
    for sr, sc in stations:
        dist_sq = (R - sr) ** 2 + (C - sc) ** 2
        contrib  = np.exp(-dist_sq / (2.0 * scale ** 2))
        coverage = np.maximum(coverage, contrib)
    return coverage


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------
def evaluate_fitness(
    chromosome: Chromosome,
    density: np.ndarray,
    scale: float,
) -> float:
    """Higher is better."""
    grid_size = density.shape[0]
    coverage  = _build_coverage_map(chromosome, grid_size, scale)
    return float(np.sum(density * coverage))


# ---------------------------------------------------------------------------
# Chromosome initialisation
# ---------------------------------------------------------------------------
def random_chromosome(n_stations: int, grid_size: int, rng: np.random.Generator) -> Chromosome:
    rows = rng.integers(0, grid_size, size=n_stations)
    cols = rng.integers(0, grid_size, size=n_stations)
    return list(zip(rows.tolist(), cols.tolist()))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def tournament_select(
    population: List[Chromosome],
    fitnesses:  List[float],
    rng:        np.random.Generator,
    k:          int,
) -> Chromosome:
    """Return a deep copy of the winner of a k-way tournament."""
    indices  = rng.integers(0, len(population), size=k).tolist()
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return list(population[best_idx])  # shallow copy of list of tuples


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------
def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng:      np.random.Generator,
) -> Chromosome:
    """
    Uniform crossover: each station position is taken from either parent
    with equal probability.
    """
    n = len(parent_a)
    mask = rng.integers(0, 2, size=n, dtype=bool)
    child: Chromosome = [
        parent_a[i] if mask[i] else parent_b[i]
        for i in range(n)
    ]
    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def mutate(
    chromosome: Chromosome,
    mutation_rate: float,
    grid_size:     int,
    rng:           np.random.Generator,
) -> Chromosome:
    """
    Each station is independently re-randomised with probability mutation_rate.
    """
    mutated: Chromosome = []
    for (r, c) in chromosome:
        if rng.random() < mutation_rate:
            r = int(rng.integers(0, grid_size))
            c = int(rng.integers(0, grid_size))
        mutated.append((r, c))
    return mutated


# ---------------------------------------------------------------------------
# Main GA runner
# ---------------------------------------------------------------------------
@dataclass
class GAResult:
    best_chromosome: Chromosome
    best_fitness:    float
    fitness_history: List[float] = field(default_factory=list)


def run_genetic_algorithm(
    density: np.ndarray,
    config:  GAConfig,
    verbose: bool = True,
) -> GAResult:
    """
    Run the genetic algorithm and return the best chromosome found.

    Parameters
    ----------
    density : normalised (G×G) density matrix (values in [0,1])
    config  : GAConfig instance
    verbose : print per-generation progress

    Returns
    -------
    GAResult with the best chromosome and fitness history
    """
    rng       = np.random.default_rng(config.seed)
    grid_size = density.shape[0]
    n_elite   = max(1, int(config.elite_fraction * config.population_size))

    # ---- initialise random population ----
    population: List[Chromosome] = [
        random_chromosome(config.n_stations, grid_size, rng)
        for _ in range(config.population_size)
    ]

    best_chromosome: Chromosome = population[0]
    best_fitness:    float      = -np.inf
    fitness_history: List[float] = []

    for gen in range(config.n_generations):

        # ---- evaluate ----
        fitnesses: List[float] = [
            evaluate_fitness(ch, density, config.coverage_scale)
            for ch in population
        ]

        # ---- track global best ----
        gen_best_idx     = int(np.argmax(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness    = gen_best_fitness
            best_chromosome = list(population[gen_best_idx])

        fitness_history.append(best_fitness)

        if verbose and (gen % 25 == 0 or gen == config.n_generations - 1):
            print(f"  Gen {gen+1:>4}/{config.n_generations}  |  "
                  f"Best fitness: {best_fitness:.4f}  |  "
                  f"Gen best: {gen_best_fitness:.4f}")

        # ---- elitism: keep top-n unchanged ----
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elites: List[Chromosome] = [list(population[i]) for i in sorted_indices[:n_elite]]

        # ---- breed next generation ----
        next_pop: List[Chromosome] = elites[:]
        while len(next_pop) < config.population_size:
            p1 = tournament_select(population, fitnesses, rng, config.tournament_size)
            p2 = tournament_select(population, fitnesses, rng, config.tournament_size)
            child = uniform_crossover(p1, p2, rng)
            child = mutate(child, config.mutation_rate, grid_size, rng)
            next_pop.append(child)

        population = next_pop

    return GAResult(
        best_chromosome=best_chromosome,
        best_fitness=best_fitness,
        fitness_history=fitness_history,
    )
