"""
CyberCore-QC: Genetic Algorithm Optimizer
==========================================
Evolves Fuzzy Inference System membership function parameters
to maximize decision accuracy against ground truth.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable
import copy
from tqdm import tqdm
import random


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing FIS membership function parameters.
    
    Chromosome Encoding:
    Each chromosome represents the triangular membership function parameters [a, b, c]
    for all input fuzzy sets. The output sets remain fixed.
    
    Genes (18 values total):
    - defect_prob_low: [a, b, c]
    - defect_prob_medium: [a, b, c]
    - defect_prob_high: [a, b, c]
    - material_fragility_low: [a, b, c]
    - material_fragility_medium: [a, b, c]
    - material_fragility_high: [a, b, c]
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elite_size: int = 5,
        tournament_size: int = 3
    ):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of elite individuals to preserve
            tournament_size: Size of tournament selection
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Evolution history
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': []
        }
        
        # Best solution found
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        
    def _create_random_chromosome(self) -> Dict[str, List[float]]:
        """
        Create a random chromosome (FIS parameters).
        
        Returns:
            Dictionary of triangular membership parameters
        """
        chromosome = {}
        
        # Defect probability membership functions (0.0 - 1.0 range)
        # Low: should peak near 0
        chromosome['defect_prob_low'] = sorted([
            0.0,
            np.random.uniform(0.0, 0.2),
            np.random.uniform(0.2, 0.5)
        ])
        
        # Medium: should peak in middle
        chromosome['defect_prob_medium'] = sorted([
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.3, 0.6),
            np.random.uniform(0.6, 0.9)
        ])
        
        # High: should peak near 1
        chromosome['defect_prob_high'] = sorted([
            np.random.uniform(0.5, 0.7),
            np.random.uniform(0.7, 1.0),
            1.0
        ])
        
        # Material fragility membership functions (0.0 - 1.0 range)
        chromosome['material_fragility_low'] = sorted([
            0.0,
            np.random.uniform(0.0, 0.2),
            np.random.uniform(0.2, 0.5)
        ])
        
        chromosome['material_fragility_medium'] = sorted([
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.3, 0.6),
            np.random.uniform(0.6, 0.9)
        ])
        
        chromosome['material_fragility_high'] = sorted([
            np.random.uniform(0.5, 0.7),
            np.random.uniform(0.7, 1.0),
            1.0
        ])
        
        return chromosome
    
    def _initialize_population(self) -> List[Dict]:
        """
        Initialize random population.
        
        Returns:
            List of chromosomes
        """
        population = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            population.append(chromosome)
        
        return population
    
    def _evaluate_fitness(
        self,
        chromosome: Dict,
        fitness_function: Callable
    ) -> float:
        """
        Evaluate fitness of a chromosome.
        
        Args:
            chromosome: FIS parameters
            fitness_function: Function to evaluate fitness
            
        Returns:
            Fitness score (higher is better)
        """
        return fitness_function(chromosome)
    
    def _tournament_selection(
        self,
        population: List[Dict],
        fitness_scores: List[float]
    ) -> Dict:
        """
        Select individual using tournament selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: List of fitness scores
            
        Returns:
            Selected chromosome
        """
        # Randomly select tournament participants
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Select best from tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return copy.deepcopy(population[winner_idx])
    
    def _crossover(
        self,
        parent1: Dict,
        parent2: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        offspring1 = {}
        offspring2 = {}
        
        # Single-point crossover for each membership function
        for key in parent1.keys():
            if random.random() < 0.5:
                # Swap entire membership function
                offspring1[key] = copy.deepcopy(parent2[key])
                offspring2[key] = copy.deepcopy(parent1[key])
            else:
                # Crossover within the [a, b, c] parameters
                alpha = random.random()
                offspring1[key] = [
                    alpha * parent1[key][i] + (1 - alpha) * parent2[key][i]
                    for i in range(3)
                ]
                offspring2[key] = [
                    (1 - alpha) * parent1[key][i] + alpha * parent2[key][i]
                    for i in range(3)
                ]
                
                # Ensure sorted order (a <= b <= c)
                offspring1[key] = sorted(offspring1[key])
                offspring2[key] = sorted(offspring2[key])
        
        return offspring1, offspring2
    
    def _mutate(self, chromosome: Dict) -> Dict:
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = copy.deepcopy(chromosome)
        
        for key in mutated.keys():
            if random.random() < self.mutation_rate:
                # Mutate one of the three parameters
                param_idx = random.randint(0, 2)
                
                # Add gaussian noise
                mutation_strength = 0.1
                mutated[key][param_idx] += np.random.normal(0, mutation_strength)
                
                # Clip to valid range
                if 'defect_prob' in key or 'material_fragility' in key:
                    mutated[key][param_idx] = np.clip(mutated[key][param_idx], 0.0, 1.0)
                
                # Ensure sorted order
                mutated[key] = sorted(mutated[key])
        
        return mutated
    
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """
        Calculate population diversity.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Diversity metric
        """
        if len(population) < 2:
            return 0.0
        
        # Flatten all chromosomes to vectors
        vectors = []
        for chromosome in population:
            vector = []
            for key in sorted(chromosome.keys()):
                vector.extend(chromosome[key])
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances.append(dist)
        
        # Return average distance
        return np.mean(distances) if distances else 0.0
    
    def evolve(
        self,
        fitness_function: Callable,
        verbose: bool = True
    ) -> Dict:
        """
        Run the genetic algorithm.
        
        Args:
            fitness_function: Function to evaluate chromosome fitness
                              Should accept a chromosome dict and return float
            verbose: Whether to show progress
            
        Returns:
            Best chromosome found
        """
        # Initialize population
        population = self._initialize_population()
        
        if verbose:
            print("\n🧬 Starting Genetic Algorithm Evolution...")
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {self.generations}")
            print("=" * 70)
        
        # Evolution loop
        pbar = tqdm(range(self.generations), desc="Evolving") if verbose else range(self.generations)
        
        for generation in pbar:
            # Evaluate fitness for entire population
            fitness_scores = []
            for chromosome in population:
                fitness = self._evaluate_fitness(chromosome, fitness_function)
                fitness_scores.append(fitness)
            
            # Track statistics
            best_gen_fitness = max(fitness_scores)
            avg_gen_fitness = np.mean(fitness_scores)
            worst_gen_fitness = min(fitness_scores)
            diversity = self._calculate_diversity(population)
            
            self.history['best_fitness'].append(best_gen_fitness)
            self.history['avg_fitness'].append(avg_gen_fitness)
            self.history['worst_fitness'].append(worst_gen_fitness)
            self.history['diversity'].append(diversity)
            
            # Update global best
            best_gen_idx = np.argmax(fitness_scores)
            if fitness_scores[best_gen_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_gen_idx]
                self.best_chromosome = copy.deepcopy(population[best_gen_idx])
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'Best': f'{best_gen_fitness:.4f}',
                    'Avg': f'{avg_gen_fitness:.4f}',
                    'Diversity': f'{diversity:.4f}'
                })
            
            # Create next generation
            next_population = []
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_population.append(copy.deepcopy(population[idx]))
            
            # Generate offspring
            while len(next_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                # Add to next generation
                next_population.append(offspring1)
                if len(next_population) < self.population_size:
                    next_population.append(offspring2)
            
            # Replace population
            population = next_population[:self.population_size]
        
        if verbose:
            print(f"\n✅ Evolution Complete!")
            print(f"Best Fitness Achieved: {self.best_fitness:.4f}")
            print("=" * 70)
        
        return self.best_chromosome


if __name__ == "__main__":
    # Test the Genetic Algorithm
    
    # Simple test fitness function (maximize sum of all parameters)
    def test_fitness(chromosome: Dict) -> float:
        total = 0
        for key, values in chromosome.items():
            total += sum(values)
        return total
    
    ga = GeneticAlgorithm(
        population_size=30,
        generations=20,
        crossover_rate=0.8,
        mutation_rate=0.2
    )
    
    best_solution = ga.evolve(test_fitness, verbose=True)
    
    print("\nBest Solution:")
    for key, values in best_solution.items():
        print(f"  {key}: {[f'{v:.3f}' for v in values]}")
