"""
Advanced UI Components for dt-QualityCtrl System
Cyberpunk-themed, animated, professional interface elements
"""

import time
import random
from typing import List, Dict, Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    TaskProgressColumn
)
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text
from rich.align import Align
from rich.columns import Columns


class CyberpunkUI:
    """Advanced cyberpunk-themed UI components."""
    
    # Color palette
    COLORS = {
        'primary': '#00ff9f',      # Neon green
        'secondary': '#ff00ff',    # Neon magenta
        'accent': '#00d4ff',       # Neon cyan
        'warning': '#ffff00',      # Neon yellow
        'danger': '#ff0055',       # Neon red
        'success': '#00ff00',      # Bright green
        'text': '#e0e0e0',         # Light gray
        'dim': '#808080',          # Gray
        'bg': '#0a0a0a'            # Near black
    }
    
    GLYPHS = {
        'arrow_right': '▶',
        'arrow_left': '◀',
        'bullet': '●',
        'diamond': '◆',
        'star': '★',
        'lightning': '⚡',
        'gear': '⚙',
        'target': '◎',
        'wave': '∿',
        'infinity': '∞',
        'delta': '△',
        'gradient': '▓',
        'block': '█',
        'shade_light': '░',
        'shade_medium': '▒',
        'shade_dark': '▓',
        'box_h': '─',
        'box_v': '│',
        'corner_tl': '╭',
        'corner_tr': '╮',
        'corner_bl': '╰',
        'corner_br': '╯'
    }
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize UI components."""
        self.console = console or Console()
    
    def create_title_panel(self, title: str, subtitle: str = "", style: str = "primary") -> Panel:
        """Create an animated title panel."""
        color = self.COLORS[style]
        
        title_text = Text()
        title_text.append(f"{self.GLYPHS['lightning']} ", style=f"bold {color}")
        title_text.append(title, style=f"bold {color}")
        title_text.append(f" {self.GLYPHS['lightning']}", style=f"bold {color}")
        
        if subtitle:
            title_text.append("\n")
            title_text.append(subtitle, style=f"dim {color}")
        
        panel = Panel(
            Align.center(title_text),
            border_style=color,
            box=box.DOUBLE,
            padding=(1, 2)
        )
        
        return panel
    
    def create_metrics_panel(self, metrics: Dict[str, float], title: str = "Metrics") -> Panel:
        """Create a live metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold magenta", justify="right")
        
        for key, value in metrics.items():
            # Add visual indicator based on metric name
            if 'loss' in key.lower():
                indicator = self.GLYPHS['arrow_left'] if value < 0.5 else self.GLYPHS['arrow_right']
                color = "green" if value < 0.5 else "red"
            elif 'acc' in key.lower() or 'accuracy' in key.lower():
                indicator = self.GLYPHS['target']
                color = "green" if value > 70 else "yellow" if value > 50 else "red"
            else:
                indicator = self.GLYPHS['bullet']
                color = "cyan"
            
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.add_row(
                f"{indicator} {key}",
                f"[{color}]{formatted_value}[/{color}]"
            )
        
        return Panel(
            table,
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def create_progress_panel(
        self,
        current: int,
        total: int,
        label: str,
        metrics: Optional[Dict] = None
    ) -> Panel:
        """Create a custom progress panel with metrics."""
        # Progress bar
        bar_width = 40
        filled = int((current / total) * bar_width)
        bar = f"[cyan]{self.GLYPHS['block'] * filled}[/cyan]"
        bar += f"[dim]{self.GLYPHS['shade_light'] * (bar_width - filled)}[/dim]"
        
        percentage = (current / total) * 100
        
        content = Text()
        content.append(f"{label}\n", style="bold yellow")
        content.append(bar + "\n")
        content.append(f"{current}/{total} ({percentage:.1f}%)", style="cyan")
        
        if metrics:
            content.append("\n\n")
            for key, val in metrics.items():
                content.append(f"{key}: ", style="dim")
                content.append(f"{val:.4f}", style="bold magenta")
                content.append("\n")
        
        return Panel(
            content,
            border_style="cyan",
            box=box.DOUBLE
        )
    
    def animated_text(self, text: str, style: str = "cyan", delay: float = 0.03):
        """Display text with typing animation."""
        displayed = ""
        for char in text:
            displayed += char
            self.console.print(f"[{style}]{displayed}[/{style}]", end="\r")
            time.sleep(delay)
        self.console.print()
    
    def glitch_text(self, text: str, iterations: int = 3):
        """Create a glitch effect on text."""
        glitch_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
        original = text
        
        for _ in range(iterations):
            # Randomize some characters
            glitched = ""
            for char in original:
                if random.random() < 0.3 and char != ' ':
                    glitched += random.choice(glitch_chars)
                else:
                    glitched += char
            
            self.console.print(f"[bold magenta]{glitched}[/bold magenta]", end="\r")
            time.sleep(0.05)
        
        # Show original
        self.console.print(f"[bold cyan]{original}[/bold cyan]")
    
    def spinning_loader(self, message: str, duration: float = 2.0):
        """Show a spinning loader."""
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        start = time.time()
        idx = 0
        
        while time.time() - start < duration:
            frame = frames[idx % len(frames)]
            self.console.print(
                f"[cyan]{frame}[/cyan] [bold]{message}[/bold]",
                end="\r"
            )
            time.sleep(0.1)
            idx += 1
        
        self.console.print(f"[green]✓[/green] [bold]{message}[/bold]")


class TrainingProgressDisplay:
    """Advanced training progress display with live updates."""
    
    def __init__(self, console: Console):
        self.console = console
        self.ui = CyberpunkUI(console)
    
    def create_training_layout(self) -> Layout:
        """Create multi-panel layout for training."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=7)
        )
        
        layout["main"].split_row(
            Layout(name="progress"),
            Layout(name="metrics")
        )
        
        return layout
    
    def get_epoch_progress(
        self,
        epoch: int,
        total_epochs: int,
        batch: int,
        total_batches: int,
        metrics: Dict
    ) -> Progress:
        """Create rich progress bars for training."""
        progress = Progress(
            SpinnerColumn(spinner_name="dots12"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        )
        
        return progress
    
    def create_metrics_table(self, train_metrics: Dict, val_metrics: Dict) -> Table:
        """Create beautiful metrics comparison table."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            title="[bold magenta]Training Metrics[/bold magenta]"
        )
        
        table.add_column("Metric", style="yellow", width=20)
        table.add_column("Train", justify="right", style="green")
        table.add_column("Validation", justify="right", style="magenta")
        table.add_column("Δ", justify="center", style="cyan")
        
        for key in train_metrics:
            if key in val_metrics:
                train_val = train_metrics[key]
                val_val = val_metrics[key]
                delta = val_val - train_val
                
                delta_str = f"{delta:+.4f}" if abs(delta) > 0.0001 else "~"
                delta_color = "green" if delta > 0 and 'acc' in key.lower() else "red" if delta < 0 and 'acc' in key.lower() else "yellow"
                
                table.add_row(
                    key,
                    f"{train_val:.4f}",
                    f"{val_val:.4f}",
                    f"[{delta_color}]{delta_str}[/{delta_color}]"
                )
        
        return table
    
    def create_live_epoch_display(
        self,
        epoch: int,
        total_epochs: int,
        current_metrics: Dict,
        best_metrics: Dict
    ) -> Panel:
        """Create live updating epoch display."""
        
        # Header
        header = Text()
        header.append("EPOCH ", style="dim cyan")
        header.append(f"{epoch}/{total_epochs}", style="bold cyan")
        header.append(" " * 5)
        header.append("⚡", style="yellow")
        
        # Metrics grid
        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold magenta", justify="right")
        grid.add_column(style="dim")
        grid.add_column(style="bold green", justify="right")
        
        for key, current_val in current_metrics.items():
            best_val = best_metrics.get(key, current_val)
            is_best = current_val >= best_val if 'acc' in key.lower() else current_val <= best_val
            
            indicator = "★" if is_best else "●"
            
            grid.add_row(
                f"{indicator} {key}:",
                f"{current_val:.4f}",
                "│ Best:",
                f"{best_val:.4f}"
            )
        
        panel = Panel(
            grid,
            title=header,
            border_style="cyan",
            box=box.HEAVY
        )
        
        return panel


class GeneticAlgorithmDisplay:
    """Advanced display for genetic algorithm evolution."""
    
    def __init__(self, console: Console):
        self.console = console
        self.ui = CyberpunkUI(console)
    
    def create_population_visualization(
        self,
        generation: int,
        population_size: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float
    ) -> Panel:
        """Create visual representation of population."""
        
        # Create fitness distribution bar
        fitness_bar_width = 50
        best_pos = int((best_fitness / 10.0) * fitness_bar_width)
        avg_pos = int((avg_fitness / 10.0) * fitness_bar_width)
        
        bar = ['░'] * fitness_bar_width
        if avg_pos < fitness_bar_width:
            bar[avg_pos] = '▒'
        if best_pos < fitness_bar_width:
            bar[best_pos] = '█'
        
        fitness_visual = ''.join(bar)
        
        content = Table.grid(padding=1)
        content.add_column(style="cyan")
        content.add_column(style="bold magenta")
        
        content.add_row("Generation:", f"{generation}")
        content.add_row("Population:", f"{population_size}")
        content.add_row("Best Fitness:", f"[green]{best_fitness:.4f}[/green]")
        content.add_row("Avg Fitness:", f"[yellow]{avg_fitness:.4f}[/yellow]")
        content.add_row("Diversity:", f"[cyan]{diversity:.2%}[/cyan]")
        content.add_row("", "")
        content.add_row("Fitness Range:", f"[dim]0[/dim] {fitness_visual} [bold]10[/bold]")
        
        return Panel(
            content,
            title=f"[bold cyan]🧬 Evolution Status[/bold cyan]",
            border_style="cyan",
            box=box.DOUBLE
        )
    
    def show_mutation_animation(self, chromosome_length: int = 20):
        """Show animated mutation process."""
        genes = ['0', '1'] * (chromosome_length // 2)
        
        self.console.print("\n[cyan]Original:[/cyan] ", end="")
        self.console.print(''.join(genes), style="dim")
        
        time.sleep(0.3)
        
        # Mutate
        mutation_points = random.sample(range(chromosome_length), 3)
        for point in mutation_points:
            genes[point] = '!' if genes[point] == '0' else '?'
            
            self.console.print("[yellow]Mutating:[/yellow] ", end="")
            self.console.print(''.join(genes), style="bold magenta", end="\r")
            time.sleep(0.2)
            
            genes[point] = '1' if genes[point] == '!' else '0'
        
        self.console.print("[green]Mutated: [/green] ", end="")
        self.console.print(''.join(genes), style="bold green")
    
    def create_evolution_graph(
        self,
        generations: List[int],
        best_fitness: List[float],
        avg_fitness: List[float]
    ) -> str:
        """Create ASCII graph of evolution."""
        height = 15
        width = 60
        
        if not generations:
            return ""
        
        # Normalize values
        max_val = max(max(best_fitness, default=1), max(avg_fitness, default=1))
        min_val = min(min(best_fitness, default=0), min(avg_fitness, default=0))
        
        graph = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot best fitness
        for i, gen in enumerate(generations):
            if i >= width:
                break
            val = best_fitness[i]
            y = int((1 - (val - min_val) / (max_val - min_val + 0.001)) * (height - 1))
            y = max(0, min(height - 1, y))
            graph[y][i] = '█'
        
        # Plot average fitness
        for i, gen in enumerate(generations):
            if i >= width:
                break
            val = avg_fitness[i]
            y = int((1 - (val - min_val) / (max_val - min_val + 0.001)) * (height - 1))
            y = max(0, min(height - 1, y))
            if graph[y][i] == ' ':
                graph[y][i] = '▒'
        
        # Convert to string
        result = []
        for row in graph:
            result.append(''.join(row))
        
        return '\n'.join(result)


def create_advanced_progress() -> Progress:
    """Create advanced progress bar with all metrics."""
    return Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            bar_width=None,
            complete_style="green",
            finished_style="bold green",
            pulse_style="cyan"
        ),
        MofNCompleteColumn(),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        expand=True
    )


# Export main classes
__all__ = [
    'CyberpunkUI',
    'TrainingProgressDisplay',
    'GeneticAlgorithmDisplay',
    'create_advanced_progress'
]
