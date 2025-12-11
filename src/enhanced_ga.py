"""
Enhanced Genetic Algorithm with Advanced UI and Visualization
Real-time evolution tracking, animated mutations, performance optimizations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time
from rich.console import Console
from rich.progress import Progress
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

from ui_components import CyberpunkUI, GeneticAlgorithmDisplay, create_advanced_progress
from logger import get_logger


class EnhancedGeneticOptimizer:
    """
    Enhanced genetic algorithm for FIS parameter optimization.
    Features real-time visualization, adaptive parameters, and robust error handling.
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        n_params: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 30,  # Reduced from 50 for faster convergence
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 5,  # Reduced from 10 proportionally
        console: Optional[Console] = None,
        enable_logging: bool = True,
        adaptive: bool = True
    ):
        """
        Initialize enhanced GA.
        
        Args:
            fitness_function: Function to evaluate fitness
            n_params: Number of parameters to optimize
            bounds: List of (min, max) for each parameter
            population_size: Size of population
            mutation_rate: Initial mutation rate
            crossover_rate: Crossover probability
            elite_size: Number of elites to preserve
            console: Rich console for output
            enable_logging: Enable logging
            adaptive: Use adaptive mutation/crossover rates
        """
        self.fitness_func = fitness_function
        self.n_params = n_params
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.adaptive = adaptive
        
        self.console = console or Console()
        self.ui = CyberpunkUI(self.console)
        self.ga_display = GeneticAlgorithmDisplay(self.console)
        
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.logger = get_logger('GeneticAlgorithm')
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.zeros(population_size)
        
        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.generation = 0
        
        # Best solution
        self.best_solution = None
        self.best_fitness = -np.inf
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        population = np.zeros((self.population_size, self.n_params))
        
        for i in range(self.n_params):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)
        
        if self.enable_logging:
            self.logger.info(f"Initialized population: {self.population_size} individuals, {self.n_params} params")
        
        return population
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for entire population with progress bar."""
        with create_advanced_progress() as progress:
            task = progress.add_task(
                "[cyan]Evaluating population...[/cyan]",
                total=self.population_size
            )
            
            for i in range(self.population_size):
                try:
                    self.fitness_scores[i] = self.fitness_func(self.population[i])
                except Exception as e:
                    if self.enable_logging:
                        self.logger.warning(f"Fitness evaluation failed for individual {i}: {e}")
                    self.fitness_scores[i] = -np.inf
                
                progress.update(task, advance=1)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity (genetic variance)."""
        # Normalize population to [0, 1]
        normalized_pop = np.zeros_like(self.population)
        for i in range(self.n_params):
            min_val, max_val = self.bounds[i]
            if max_val > min_val:
                normalized_pop[:, i] = (self.population[:, i] - min_val) / (max_val - min_val)
        
        # Calculate average pairwise distance
        diversity = np.mean(np.std(normalized_pop, axis=0))
        
        return float(diversity)
    
    def _select_parents(self, n_parents: int) -> np.ndarray:
        """
        Tournament selection for parent selection.
        
        Args:
            n_parents: Number of parents to select
            
        Returns:
            Selected parent indices
        """
        tournament_size = max(3, self.population_size // 10)
        parent_indices = []
        
        for _ in range(n_parents):
            # Random tournament
            tournament_idx = np.random.choice(self.population_size, tournament_size, replace=False)
            tournament_fitness = self.fitness_scores[tournament_idx]
            
            # Select winner
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parent_indices.append(winner_idx)
        
        return np.array(parent_indices)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        mask = np.random.random(self.n_params) < 0.5
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        offspring1[mask] = parent2[mask]
        offspring2[mask] = parent1[mask]
        
        return offspring1, offspring2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation with adaptive rate.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # Adaptive mutation rate based on diversity
        if self.adaptive:
            diversity = self.diversity_history[-1] if self.diversity_history else 0.5
            adaptive_rate = self.mutation_rate * (1 + (1 - diversity))  # Higher rate when diversity is low
        else:
            adaptive_rate = self.mutation_rate
        
        # Apply mutation
        for i in range(self.n_params):
            if np.random.random() < adaptive_rate:
                min_val, max_val = self.bounds[i]
                
                # Gaussian mutation with bounds
                sigma = (max_val - min_val) * 0.1  # 10% of range
                mutation = np.random.normal(0, sigma)
                
                mutated[i] = np.clip(mutated[i] + mutation, min_val, max_val)
        
        return mutated
    
    def _create_new_generation(self) -> np.ndarray:
        """Create new generation using selection, crossover, and mutation."""
        new_population = np.zeros_like(self.population)
        
        # Elitism: preserve best individuals
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        new_population[:self.elite_size] = self.population[elite_indices]
        
        # Generate offspring
        n_offspring = self.population_size - self.elite_size
        
        for i in range(self.elite_size, self.population_size, 2):
            # Select parents
            parent_indices = self._select_parents(2)
            parent1 = self.population[parent_indices[0]]
            parent2 = self.population[parent_indices[1]]
            
            # Crossover
            offspring1, offspring2 = self._crossover(parent1, parent2)
            
            # Mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            
            # Add to new population
            new_population[i] = offspring1
            if i + 1 < self.population_size:
                new_population[i + 1] = offspring2
        
        return new_population
    
    def _display_generation_summary(self):
        """Display beautiful generation summary."""
        best_idx = np.argmax(self.fitness_scores)
        best_fitness = self.fitness_scores[best_idx]
        avg_fitness = np.mean(self.fitness_scores)
        diversity = self.diversity_history[-1]
        
        # Create population panel
        panel = self.ga_display.create_population_visualization(
            self.generation,
            self.population_size,
            best_fitness,
            avg_fitness,
            diversity
        )
        
        self.console.print(panel)
        
        # Log metrics using standard logging
        if self.enable_logging:
            self.logger.info(
                f"Gen {self.generation}: Best={best_fitness:.4f}, "
                f"Avg={avg_fitness:.4f}, Diversity={diversity:.2%}"
            )
    
    def optimize(
        self,
        n_generations: int = 100,
        target_fitness: Optional[float] = None,
        patience: int = 20,
        show_animation: bool = True
    ) -> Dict:
        """
        Run genetic algorithm optimization.
        
        Args:
            n_generations: Number of generations
            target_fitness: Target fitness to reach (early stopping)
            patience: Generations without improvement before stopping
            show_animation: Show mutation animations
            
        Returns:
            Optimization results
        """
        # Show header
        header = self.ui.create_title_panel(
            "GENETIC ALGORITHM OPTIMIZATION",
            f"Generations: {n_generations} | Population: {self.population_size} | Params: {self.n_params}",
            style="primary"
        )
        self.console.print(header)
        
        if self.enable_logging:
            self.logger.info(f"Starting GA optimization: {n_generations} generations")
        
        start_time = time.time()
        generations_without_improvement = 0
        
        try:
            for gen in range(1, n_generations + 1):
                self.generation = gen
                gen_start = time.time()
                
                # Evaluate population
                self._evaluate_population()
                
                # Track metrics
                best_idx = np.argmax(self.fitness_scores)
                best_fitness = self.fitness_scores[best_idx]
                avg_fitness = np.mean(self.fitness_scores)
                diversity = self._calculate_diversity()
                
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                self.diversity_history.append(diversity)
                
                # Update best solution
                if best_fitness > self.best_fitness:
                    self.best_fitness = best_fitness
                    self.best_solution = self.population[best_idx].copy()
                    generations_without_improvement = 0
                    
                    if self.enable_logging:
                        self.logger.info(f"New best fitness: {best_fitness:.4f} at generation {gen}")
                else:
                    generations_without_improvement += 1
                
                # Display summary every 5 generations or on first/last
                if gen % 5 == 0 or gen == 1 or gen == n_generations:
                    self._display_generation_summary()
                
                # Show mutation animation occasionally
                if show_animation and gen % 20 == 0:
                    self.ga_display.show_mutation_animation(self.n_params)
                
                # Early stopping checks
                if target_fitness and best_fitness >= target_fitness:
                    self.console.print(f"\n[green]✓[/green] Target fitness {target_fitness:.4f} reached!")
                    if self.enable_logging:
                        self.logger.info(f"Target fitness reached at generation {gen}")
                    break
                
                if generations_without_improvement >= patience:
                    self.console.print(f"\n[yellow]⚠[/yellow] No improvement for {patience} generations. Stopping early.")
                    if self.enable_logging:
                        self.logger.info(f"Early stopping at generation {gen}")
                    break
                
                # Create new generation
                self.population = self._create_new_generation()
                
                gen_time = time.time() - gen_start
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Optimization interrupted by user[/yellow]")
            if self.enable_logging:
                self.logger.warning("Optimization interrupted")
        
        except Exception as e:
            self.console.print(f"\n[red]❌ Optimization failed: {str(e)}[/red]")
            if self.enable_logging:
                self.logger.error(f"Optimization failed", exc_info=True)
            raise
        
        total_time = time.time() - start_time
        
        # Show completion
        completion_panel = self.ui.create_title_panel(
            "OPTIMIZATION COMPLETE",
            f"Best Fitness: {self.best_fitness:.4f} | Generations: {self.generation} | Time: {total_time/60:.1f}min",
            style="success"
        )
        self.console.print("\n", completion_panel)
        
        # Return results
        results = {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'generations': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'total_time': total_time
        }
        
        return results
    
    @property
    def history(self) -> Dict[str, List[float]]:
        """Return evolution history for visualization."""
        return {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'diversity': self.diversity_history,
            'generations': list(range(1, len(self.best_fitness_history) + 1))
        }


# Export
__all__ = ['EnhancedGeneticOptimizer']
