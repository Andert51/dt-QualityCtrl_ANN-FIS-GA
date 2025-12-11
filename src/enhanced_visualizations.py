"""
Enhanced Visualization Module with Animations and GIFs
Creates beautiful, informative plots and animated outputs
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class AnimatedVisualizer:
    """Create animated visualizations for training progress."""
    
    def __init__(self, output_dir: Path, style: str = 'dark_background'):
        """
        Initialize animator.
        
        Args:
            output_dir: Directory to save animations
            style: Matplotlib style
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use(style)
        
        # Cyberpunk color palette
        self.colors = {
            'cyan': '#00d4ff',
            'magenta': '#ff00ff',
            'yellow': '#ffff00',
            'green': '#00ff9f',
            'red': '#ff0055',
            'blue': '#0055ff',
            'purple': '#9f00ff'
        }
    
    def create_training_animation(
        self,
        history: Dict[str, List[float]],
        output_filename: str = 'training_animation.gif',
        fps: int = 10,
        duration_per_epoch: float = 0.5
    ) -> Path:
        """
        Create animated GIF of training progress.
        
        Args:
            history: Training history dict
            output_filename: Name of output GIF file
            fps: Frames per second
            duration_per_epoch: Duration to show each epoch
            
        Returns:
            Path to saved GIF
        """
        # Validate history
        if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
            self.console.print("[yellow]⚠️  No training history available for animation[/yellow]")
            return None
        
        epochs = len(history['train_loss'])
        
        # Ensure duration_per_epoch is numeric
        try:
            duration_per_epoch = float(duration_per_epoch)
        except (TypeError, ValueError):
            duration_per_epoch = 0.5
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#0a0a0a')
        
        def animate(frame):
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
                ax.set_facecolor('#1a1a1a')
            
            # Current epoch (frame)
            current_epoch = min(frame + 1, epochs)
            epoch_range = range(1, current_epoch + 1)
            
            # Plot 1: Loss curves
            ax1 = axes[0, 0]
            ax1.plot(epoch_range, history['train_loss'][:current_epoch], 
                    color=self.colors['cyan'], linewidth=2, marker='o', 
                    markersize=4, label='Train Loss')
            ax1.plot(epoch_range, history['val_loss'][:current_epoch], 
                    color=self.colors['magenta'], linewidth=2, marker='s', 
                    markersize=4, label='Val Loss')
            ax1.set_xlabel('Epoch', fontsize=12, color='white')
            ax1.set_ylabel('Loss', fontsize=12, color='white')
            ax1.set_title('Training Loss', fontsize=14, fontweight='bold', 
                         color=self.colors['yellow'])
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(colors='white')
            
            # Plot 2: Accuracy curves
            ax2 = axes[0, 1]
            ax2.plot(epoch_range, history['train_acc'][:current_epoch], 
                    color=self.colors['green'], linewidth=2, marker='o', 
                    markersize=4, label='Train Acc')
            ax2.plot(epoch_range, history['val_acc'][:current_epoch], 
                    color=self.colors['magenta'], linewidth=2, marker='s', 
                    markersize=4, label='Val Acc')
            ax2.set_xlabel('Epoch', fontsize=12, color='white')
            ax2.set_ylabel('Accuracy (%)', fontsize=12, color='white')
            ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold', 
                         color=self.colors['yellow'])
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(colors='white')
            
            # Plot 3: Defect loss
            ax3 = axes[1, 0]
            ax3.plot(epoch_range, history['train_defect_loss'][:current_epoch], 
                    color=self.colors['red'], linewidth=2, marker='o', 
                    markersize=4, label='Train Defect')
            ax3.plot(epoch_range, history['val_defect_loss'][:current_epoch], 
                    color=self.colors['blue'], linewidth=2, marker='s', 
                    markersize=4, label='Val Defect')
            ax3.set_xlabel('Epoch', fontsize=12, color='white')
            ax3.set_ylabel('Defect Loss', fontsize=12, color='white')
            ax3.set_title('Defect Detection Loss', fontsize=14, fontweight='bold', 
                         color=self.colors['yellow'])
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.tick_params(colors='white')
            
            # Plot 4: Learning rate
            if 'learning_rates' in history and len(history['learning_rates']) > 0:
                ax4 = axes[1, 1]
                ax4.plot(epoch_range, history['learning_rates'][:current_epoch], 
                        color=self.colors['purple'], linewidth=2, marker='D', 
                        markersize=4)
                ax4.set_xlabel('Epoch', fontsize=12, color='white')
                ax4.set_ylabel('Learning Rate', fontsize=12, color='white')
                ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', 
                             color=self.colors['yellow'])
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.tick_params(colors='white')
            
            # Add epoch counter
            fig.text(0.5, 0.95, f'Epoch {current_epoch}/{epochs}', 
                    ha='center', fontsize=16, fontweight='bold', 
                    color=self.colors['cyan'])
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Create animation
        frames = int(epochs * duration_per_epoch * fps)
        frame_indices = np.linspace(0, epochs - 1, frames, dtype=int)
        
        anim = animation.FuncAnimation(
            fig, animate, frames=frame_indices, 
            interval=1000/fps, repeat=True
        )
        
        # Save as GIF
        output_path = self.output_dir / output_filename
        anim.save(
            output_path, 
            writer='pillow', 
            fps=fps,
            dpi=100
        )
        
        plt.close(fig)
        
        return output_path
    
    def create_ga_evolution_animation(
        self,
        generations: List[int],
        best_fitness: List[float],
        avg_fitness: List[float],
        diversity: List[float],
        fps: int = 10
    ) -> Path:
        """
        Create animated GIF of GA evolution.
        
        Args:
            generations: Generation numbers
            best_fitness: Best fitness per generation
            avg_fitness: Average fitness per generation
            diversity: Population diversity per generation
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a0a')
        
        def animate(frame):
            for ax in axes:
                ax.clear()
                ax.set_facecolor('#1a1a1a')
            
            current_gen = min(frame + 1, len(generations))
            gen_range = generations[:current_gen]
            
            # Plot 1: Fitness evolution
            ax1 = axes[0]
            ax1.plot(gen_range, best_fitness[:current_gen], 
                    color=self.colors['green'], linewidth=3, marker='*', 
                    markersize=8, label='Best Fitness')
            ax1.plot(gen_range, avg_fitness[:current_gen], 
                    color=self.colors['cyan'], linewidth=2, marker='o', 
                    markersize=4, label='Avg Fitness', alpha=0.7)
            ax1.fill_between(gen_range, best_fitness[:current_gen], avg_fitness[:current_gen],
                            color=self.colors['green'], alpha=0.2)
            ax1.set_xlabel('Generation', fontsize=12, color='white')
            ax1.set_ylabel('Fitness', fontsize=12, color='white')
            ax1.set_title('Genetic Algorithm Evolution', fontsize=14, fontweight='bold', 
                         color=self.colors['yellow'])
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(colors='white')
            
            # Plot 2: Diversity
            ax2 = axes[1]
            ax2.plot(gen_range, diversity[:current_gen], 
                    color=self.colors['magenta'], linewidth=2, marker='D', 
                    markersize=4)
            ax2.set_xlabel('Generation', fontsize=12, color='white')
            ax2.set_ylabel('Diversity', fontsize=12, color='white')
            ax2.set_title('Population Diversity', fontsize=14, fontweight='bold', 
                         color=self.colors['yellow'])
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(colors='white')
            ax2.set_ylim([0, 1])
            
            # Add generation counter
            fig.text(0.5, 0.95, f'Generation {current_gen}/{len(generations)}', 
                    ha='center', fontsize=16, fontweight='bold', 
                    color=self.colors['cyan'])
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Create animation
        frames = len(generations)
        anim = animation.FuncAnimation(
            fig, animate, frames=frames, 
            interval=1000/fps, repeat=True
        )
        
        # Save as GIF
        output_path = self.output_dir / 'ga_evolution_animation.gif'
        anim.save(
            output_path, 
            writer='pillow', 
            fps=fps,
            dpi=100
        )
        
        plt.close(fig)
        
        return output_path
    
    def create_detailed_metrics_plot(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'detailed_metrics.png'
    ) -> Path:
        """Create comprehensive metrics visualization."""
        
        epochs = len(history['train_loss'])
        epoch_range = range(1, epochs + 1)
        
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('#0a0a0a')
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Main loss plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#1a1a1a')
        ax1.plot(epoch_range, history['train_loss'], color=self.colors['cyan'], 
                linewidth=2, marker='o', markersize=3, label='Train Loss', alpha=0.8)
        ax1.plot(epoch_range, history['val_loss'], color=self.colors['magenta'], 
                linewidth=2, marker='s', markersize=3, label='Val Loss', alpha=0.8)
        ax1.fill_between(epoch_range, history['train_loss'], history['val_loss'],
                        alpha=0.2, color=self.colors['cyan'])
        ax1.set_xlabel('Epoch', color='white', fontsize=11)
        ax1.set_ylabel('Loss', color='white', fontsize=11)
        ax1.set_title('Training & Validation Loss', color=self.colors['yellow'], 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(colors='white')
        
        # 2. Main accuracy plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.set_facecolor('#1a1a1a')
        ax2.plot(epoch_range, history['train_acc'], color=self.colors['green'], 
                linewidth=2, marker='o', markersize=3, label='Train Acc', alpha=0.8)
        ax2.plot(epoch_range, history['val_acc'], color=self.colors['magenta'], 
                linewidth=2, marker='s', markersize=3, label='Val Acc', alpha=0.8)
        ax2.axhline(y=max(history['val_acc']), color=self.colors['yellow'], 
                   linestyle='--', linewidth=1, label=f'Best: {max(history["val_acc"]):.2f}%')
        ax2.set_xlabel('Epoch', color='white', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', color='white', fontsize=11)
        ax2.set_title('Training & Validation Accuracy', color=self.colors['yellow'], 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', facecolor='#1a1a1a', edgecolor='white')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(colors='white')
        
        # 3. Defect loss
        ax3 = fig.add_subplot(gs[2, :2])
        ax3.set_facecolor('#1a1a1a')
        ax3.plot(epoch_range, history['train_defect_loss'], color=self.colors['red'], 
                linewidth=2, marker='o', markersize=3, label='Train Defect', alpha=0.8)
        ax3.plot(epoch_range, history['val_defect_loss'], color=self.colors['blue'], 
                linewidth=2, marker='s', markersize=3, label='Val Defect', alpha=0.8)
        ax3.set_xlabel('Epoch', color='white', fontsize=11)
        ax3.set_ylabel('Defect Loss', color='white', fontsize=11)
        ax3.set_title('Defect Detection Loss', color=self.colors['yellow'], 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(colors='white')
        
        # 4. Learning rate schedule
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            ax4 = fig.add_subplot(gs[0, 2])
            ax4.set_facecolor('#1a1a1a')
            ax4.plot(epoch_range, history['learning_rates'], color=self.colors['purple'], 
                    linewidth=2, marker='D', markersize=3)
            ax4.set_xlabel('Epoch', color='white', fontsize=10)
            ax4.set_ylabel('LR', color='white', fontsize=10)
            ax4.set_title('Learning Rate', color=self.colors['yellow'], 
                         fontsize=11, fontweight='bold')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.tick_params(colors='white', labelsize=8)
        
        # 5. Overfitting gauge
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor('#1a1a1a')
        overfitting = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        ax5.plot(epoch_range, overfitting, color=self.colors['red'], linewidth=2)
        ax5.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax5.fill_between(epoch_range, 0, overfitting, 
                        where=[x > 0 for x in overfitting],
                        color=self.colors['red'], alpha=0.3)
        ax5.set_xlabel('Epoch', color='white', fontsize=10)
        ax5.set_ylabel('Gap (%)', color='white', fontsize=10)
        ax5.set_title('Train-Val Gap', color=self.colors['yellow'], 
                     fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.tick_params(colors='white', labelsize=8)
        
        # 6. Epoch time
        if 'epoch_times' in history and len(history['epoch_times']) > 0:
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.set_facecolor('#1a1a1a')
            ax6.bar(epoch_range, history['epoch_times'], color=self.colors['cyan'], alpha=0.7)
            ax6.set_xlabel('Epoch', color='white', fontsize=10)
            ax6.set_ylabel('Time (s)', color='white', fontsize=10)
            ax6.set_title('Epoch Duration', color=self.colors['yellow'], 
                         fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax6.tick_params(colors='white', labelsize=8)
        
        # Overall title
        best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
        fig.suptitle(
            f'Training Analytics | Best Epoch: {best_epoch} | Best Acc: {max(history["val_acc"]):.2f}%',
            fontsize=16, fontweight='bold', color=self.colors['cyan'], y=0.98
        )
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=200, facecolor='#0a0a0a', bbox_inches='tight')
        plt.close(fig)
        
        return output_path


# Export
__all__ = ['AnimatedVisualizer']
