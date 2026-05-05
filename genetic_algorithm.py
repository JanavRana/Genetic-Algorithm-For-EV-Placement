import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# Type aliases
Station  = Tuple[int, int]          # (row, col)
Chromosome = List[Station]


# Configuration dataclass
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


# Distance-based coverage kernel

# Return a (grid_size, grid_size) float array where each cell value is
# the maximum coverage contribution from any station:

# coverage[r,c] = max_i exp( -||cell - station_i||² / (2·scale²) )

def _build_coverage_map(
    stations: Chromosome,
    grid_size: int,
    scale: float,
) -> np.ndarray:

    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    R, C = np.meshgrid(rows, cols, indexing="ij")  # (G, G)

    coverage = np.zeros((grid_size, grid_size), dtype=np.float64)
    for sr, sc in stations:
        dist_sq = (R - sr) ** 2 + (C - sc) ** 2
        contrib  = np.exp(-dist_sq / (2.0 * scale ** 2))
        coverage = np.maximum(coverage, contrib)
    return coverage


# Fitness evaluation
def evaluate_fitness(
    chromosome: Chromosome,
    density: np.ndarray,
    scale: float,
) -> float:

    grid_size = density.shape[0]
    coverage  = _build_coverage_map(chromosome, grid_size, scale)

    #  MAIN TERM: weighted population coverage 
    population_score = np.sum((density ** 1.5) * coverage)

    #  NEW: penalty for spreading too far 
    penalty = 0.0
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            r1, c1 = chromosome[i]
            r2, c2 = chromosome[j]
            dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            penalty += dist

    penalty = penalty / (len(chromosome) + 1)

    # --- FINAL SCORE ---
    return float(population_score - 0.1 * penalty)


# Chromosome initialisation
def random_chromosome(n_stations: int, grid_size: int, rng: np.random.Generator) -> Chromosome:
    rows = rng.integers(0, grid_size, size=n_stations)
    cols = rng.integers(0, grid_size, size=n_stations)
    return list(zip(rows.tolist(), cols.tolist()))


# Selection
def tournament_select(
    population: List[Chromosome],
    fitnesses:  List[float],
    rng:        np.random.Generator,
    k:          int,
) -> Chromosome:
    # Return a deep copy of the winner of a k-way tournament.
    indices  = rng.integers(0, len(population), size=k).tolist()
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return list(population[best_idx])  # shallow copy of list of tuples


# Crossover
def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng:      np.random.Generator,
) -> Chromosome:

    # each station position is taken from either parent with equal probability.
    n = len(parent_a)
    mask = rng.integers(0, 2, size=n, dtype=bool)
    child: Chromosome = [
        parent_a[i] if mask[i] else parent_b[i]
        for i in range(n)
    ]
    return child


# Mutation
def mutate(
    chromosome: Chromosome,
    mutation_rate: float,
    grid_size:     int,
    rng:           np.random.Generator,
) -> Chromosome:
    # Stations is independently re-randomised with probability mutation_rate.

    mutated: Chromosome = []
    for (r, c) in chromosome:
        if rng.random() < mutation_rate:
            r = int(rng.integers(0, grid_size))
            c = int(rng.integers(0, grid_size))
        mutated.append((r, c))
    return mutated


# Main GA runner
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

    rng       = np.random.default_rng(config.seed)
    grid_size = density.shape[0]
    n_elite   = max(1, int(config.elite_fraction * config.population_size))

    # initialise random population 
    population: List[Chromosome] = [
        random_chromosome(config.n_stations, grid_size, rng)
        for _ in range(config.population_size)
    ]

    best_chromosome: Chromosome = population[0]
    best_fitness:    float      = -np.inf
    fitness_history: List[float] = []

    for gen in range(config.n_generations):

        #  evaluate 
        fitnesses: List[float] = [
            evaluate_fitness(ch, density, config.coverage_scale)
            for ch in population
        ]

        # track global best 
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

        # seletion: top-n  
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elites: List[Chromosome] = [list(population[i]) for i in sorted_indices[:n_elite]]

        # breed next generation 
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
