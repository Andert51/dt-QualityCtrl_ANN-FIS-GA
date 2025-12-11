"""
CyberCore-QC: Advanced Visualization System
============================================
High-end visualization suite using Matplotlib, Seaborn, and Plotly.
Includes CNN, FIS, GA, and system-wide visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import skfuzzy as fuzz


# Set style
plt.style.use('dark_background')
sns.set_palette("husl")


class VisualizationHub:
    """
    Comprehensive visualization hub for the CyberCore-QC system.
    Creates cyberpunk-themed, publication-quality visualizations.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualization hub.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cyberpunk color scheme
        self.colors = {
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'yellow': '#FFFF00',
            'green': '#00FF00',
            'red': '#FF0000',
            'blue': '#0080FF',
            'purple': '#8000FF',
            'orange': '#FF8000'
        }
        
    # ==================== CNN Visualizations ====================
    
    def plot_training_curves(self, history: Dict, save_name: str = "training_curves.png"):
        """
        Plot CNN training learning curves.
        
        Args:
            history: Training history dictionary
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, history['train_loss'], label='Train Loss', 
                color=self.colors['cyan'], linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, history['val_loss'], label='Val Loss', 
                color=self.colors['magenta'], linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax1.legend(fontsize=10, framealpha=0.8)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['train_acc'], label='Train Accuracy', 
                color=self.colors['green'], linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, history['val_acc'], label='Val Accuracy', 
                color=self.colors['orange'], linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax2.legend(fontsize=10, framealpha=0.8)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Defect detection loss
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, history['train_defect_loss'], label='Train Defect Loss', 
                color=self.colors['purple'], linewidth=2, marker='o', markersize=4)
        ax3.plot(epochs, history['val_defect_loss'], label='Val Defect Loss', 
                color=self.colors['red'], linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Defect Loss (BCE)', fontsize=12, fontweight='bold')
        ax3.set_title('Defect Probability Loss', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax3.legend(fontsize=10, framealpha=0.8)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('CNN Training Analytics', fontsize=18, fontweight='bold', 
                    color=self.colors['cyan'], y=1.02)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved training curves to {save_name}")
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str], save_name: str = "confusion_matrix.png"):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_name: Filename to save plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax, linewidths=1, linecolor='cyan')
        
        ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax.set_ylabel('True Class', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', 
                    color=self.colors['yellow'], pad=20)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved confusion matrix to {save_name}")
    
    def plot_network_topology(self, model, save_name: str = "network_topology.png"):
        """
        Visualize CNN network architecture as a graph.
        
        Args:
            model: CNN model
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        G = nx.DiGraph()
        
        # Define layers
        layers = [
            ("Input\n(224×224×3)", 0),
            ("Conv1\n(64)", 1),
            ("Conv2\n(128)", 2),
            ("Conv3\n(256)", 3),
            ("Conv4\n(512)", 4),
            ("GlobalAvgPool", 5),
            ("FC-256", 6),
            ("FC-128", 7),
            ("Classification\n(6 classes)", 8),
            ("Defect Prob\n(Binary)", 8)
        ]
        
        # Add nodes
        for layer_name, level in layers:
            G.add_node(layer_name, level=level)
        
        # Add edges
        edges = [
            ("Input\n(224×224×3)", "Conv1\n(64)"),
            ("Conv1\n(64)", "Conv2\n(128)"),
            ("Conv2\n(128)", "Conv3\n(256)"),
            ("Conv3\n(256)", "Conv4\n(512)"),
            ("Conv4\n(512)", "GlobalAvgPool"),
            ("GlobalAvgPool", "FC-256"),
            ("FC-256", "FC-128"),
            ("FC-128", "Classification\n(6 classes)"),
            ("FC-128", "Defect Prob\n(Binary)")
        ]
        
        G.add_edges_from(edges)
        
        # Create hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=self.colors['cyan'], 
                              node_size=3000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=self.colors['magenta'], 
                              width=2, alpha=0.6, arrows=True, 
                              arrowsize=20, arrowstyle='->', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                               font_color='white', ax=ax)
        
        ax.set_title('CNN Network Topology (ResNet18 Backbone)', 
                    fontsize=16, fontweight='bold', color=self.colors['cyan'], pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved network topology to {save_name}")
    
    def plot_activation_heatmap(self, image: torch.Tensor, feature_maps: Dict,
                               save_name: str = "activation_heatmap.png"):
        """
        Plot activation heatmaps for CNN layers.
        
        Args:
            image: Input image tensor
            feature_maps: Dictionary of feature maps from different layers
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)
        
        # Original image
        ax_img = fig.add_subplot(gs[0, :2])
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        ax_img.imshow(img_np)
        ax_img.set_title('Original Image', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax_img.axis('off')
        
        # Plot feature maps from different layers
        layer_positions = {
            'layer1': (0, 2),
            'layer2': (0, 3),
            'layer3': (0, 4),
            'layer4': (1, 0)
        }
        
        for layer_name, (row, col) in layer_positions.items():
            if layer_name in feature_maps:
                ax = fig.add_subplot(gs[row, col])
                
                # Average across channels
                feat_map = feature_maps[layer_name][0].cpu().numpy()
                feat_map_avg = np.mean(feat_map, axis=0)
                
                im = ax.imshow(feat_map_avg, cmap='hot', interpolation='bilinear')
                ax.set_title(f'{layer_name.capitalize()}', fontsize=12, 
                           fontweight='bold', color=self.colors['yellow'])
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('CNN Activation Heatmaps', fontsize=18, fontweight='bold', 
                    color=self.colors['magenta'], y=0.98)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved activation heatmap to {save_name}")
    
    # ==================== FIS Visualizations ====================
    
    def plot_membership_functions(self, fis, save_name: str = "membership_functions.png"):
        """
        Plot fuzzy membership functions.
        
        Args:
            fis: FuzzyQualityController instance
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Defect Probability
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fis.defect_prob.universe, fis.defect_prob['low'].mf, 
                color=self.colors['green'], linewidth=2, label='Low')
        ax1.plot(fis.defect_prob.universe, fis.defect_prob['medium'].mf, 
                color=self.colors['yellow'], linewidth=2, label='Medium')
        ax1.plot(fis.defect_prob.universe, fis.defect_prob['high'].mf, 
                color=self.colors['red'], linewidth=2, label='High')
        ax1.set_xlabel('Defect Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Membership', fontsize=12, fontweight='bold')
        ax1.set_title('Defect Probability MFs', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax1.legend(fontsize=10, framealpha=0.8)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([-0.05, 1.05])
        
        # Material Fragility
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(fis.material_fragility.universe, fis.material_fragility['low'].mf, 
                color=self.colors['green'], linewidth=2, label='Low')
        ax2.plot(fis.material_fragility.universe, fis.material_fragility['medium'].mf, 
                color=self.colors['yellow'], linewidth=2, label='Medium')
        ax2.plot(fis.material_fragility.universe, fis.material_fragility['high'].mf, 
                color=self.colors['red'], linewidth=2, label='High')
        ax2.set_xlabel('Material Fragility', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Membership', fontsize=12, fontweight='bold')
        ax2.set_title('Material Fragility MFs', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax2.legend(fontsize=10, framealpha=0.8)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([-0.05, 1.05])
        
        # Severity Score (Output)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(fis.severity.universe, fis.severity['accept'].mf, 
                color=self.colors['green'], linewidth=2, label='Accept')
        ax3.plot(fis.severity.universe, fis.severity['rework'].mf, 
                color=self.colors['yellow'], linewidth=2, label='Rework')
        ax3.plot(fis.severity.universe, fis.severity['reject'].mf, 
                color=self.colors['red'], linewidth=2, label='Reject')
        ax3.set_xlabel('Severity Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Membership', fontsize=12, fontweight='bold')
        ax3.set_title('Severity Score MFs', fontsize=14, fontweight='bold', color=self.colors['cyan'])
        ax3.legend(fontsize=10, framealpha=0.8)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([-0.05, 1.05])
        
        # Add color zones for severity
        ax3.axvspan(0, 3, alpha=0.2, color=self.colors['green'])
        ax3.axvspan(3, 7, alpha=0.2, color=self.colors['yellow'])
        ax3.axvspan(7, 10, alpha=0.2, color=self.colors['red'])
        
        plt.suptitle('Fuzzy Membership Functions', fontsize=18, fontweight='bold', 
                    color=self.colors['magenta'], y=0.98)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved membership functions to {save_name}")
    
    def plot_fuzzy_surface_3d(self, fis, save_name: str = "fuzzy_surface_3d.html"):
        """
        Create interactive 3D surface plot of fuzzy decision space.
        
        Args:
            fis: FuzzyQualityController instance
            save_name: Filename to save plot
        """
        # Create grid
        defect_range = np.linspace(0, 1, 50)
        fragility_range = np.linspace(0, 1, 50)
        
        X, Y = np.meshgrid(defect_range, fragility_range)
        Z = np.zeros_like(X)
        
        # Calculate severity for each point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    result = fis.predict(X[i, j], Y[i, j])
                    Z[i, j] = result['severity_score']
                except:
                    Z[i, j] = 5.0  # Default value if prediction fails
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Severity Score', side='right')
            )
        )])
        
        fig.update_layout(
            title='Fuzzy Decision Surface (3D)',
            scene=dict(
                xaxis_title='Defect Probability',
                yaxis_title='Material Fragility',
                zaxis_title='Severity Score',
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333'),
                yaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333'),
                zaxis=dict(backgroundcolor='#1a1a1a', gridcolor='#333333')
            ),
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            font=dict(color='#00FFFF', size=12),
            width=1000,
            height=800
        )
        
        fig.write_html(self.output_dir / save_name)
        print(f"✅ Saved 3D fuzzy surface to {save_name}")
    
    # ==================== GA Visualizations ====================
    
    def plot_ga_evolution(self, ga_history: Dict, save_name: str = "ga_evolution.png"):
        """
        Plot genetic algorithm evolution progress.
        
        Args:
            ga_history: GA history dictionary
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        
        generations = range(1, len(ga_history['best_fitness']) + 1)
        
        # Fitness evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(generations, ga_history['best_fitness'], 
                label='Best Fitness', color=self.colors['cyan'], linewidth=2.5, marker='o')
        ax1.plot(generations, ga_history['avg_fitness'], 
                label='Avg Fitness', color=self.colors['yellow'], linewidth=2, marker='s')
        ax1.plot(generations, ga_history['worst_fitness'], 
                label='Worst Fitness', color=self.colors['red'], linewidth=1.5, marker='^', alpha=0.7)
        ax1.fill_between(generations, ga_history['worst_fitness'], ga_history['best_fitness'],
                        alpha=0.2, color=self.colors['cyan'])
        ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
        ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax1.legend(fontsize=10, framealpha=0.8)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Population diversity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(generations, ga_history['diversity'], 
                color=self.colors['magenta'], linewidth=2.5, marker='D')
        ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Population Diversity', fontsize=12, fontweight='bold')
        ax2.set_title('Population Diversity Over Time', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.fill_between(generations, 0, ga_history['diversity'], alpha=0.3, color=self.colors['magenta'])
        
        plt.suptitle('Genetic Algorithm Evolution', fontsize=18, fontweight='bold', 
                    color=self.colors['green'], y=1.02)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved GA evolution to {save_name}")
    
    # ==================== System-Wide Visualizations ====================
    
    def plot_results_grid(self, images: List, predictions: List[Dict], 
                         save_name: str = "results_grid.png", max_samples: int = 16):
        """
        Plot grid of sample results with predictions.
        
        Args:
            images: List of image arrays
            predictions: List of prediction dictionaries
            save_name: Filename to save plot
            max_samples: Maximum number of samples to display
        """
        n_samples = min(len(images), max_samples)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for idx in range(n_samples):
            ax = axes[idx]
            
            # Display image
            img = images[idx]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy().transpose(1, 2, 0)
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            # Add prediction info
            pred = predictions[idx]
            decision = pred['decision']
            severity = pred['severity_score']
            color = pred['color']
            
            # Color mapping
            border_color = {'green': self.colors['green'], 
                          'yellow': self.colors['yellow'], 
                          'red': self.colors['red']}[color]
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)
            
            # Add text
            ax.set_title(f"{decision}\nSeverity: {severity:.2f}", 
                        fontsize=11, fontweight='bold', color=border_color)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Quality Control Results', fontsize=18, fontweight='bold', 
                    color=self.colors['cyan'], y=1.0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved results grid to {save_name}")
    
    def plot_dataset_overview(self, dataset_info: Dict, save_name: str = "dataset_overview.png"):
        """
        Plot dataset statistics overview.
        
        Args:
            dataset_info: Dataset information dictionary
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Class distribution
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Handle different metadata formats
        if 'classes' in dataset_info and isinstance(dataset_info['classes'], dict):
            # Synthetic dataset format: {0: 'Normal', 1: 'Scratches', ...}
            class_names = list(dataset_info['classes'].values())
        elif 'class_names' in dataset_info:
            # Kaggle dataset format: ['Normal', 'Scratches', ...]
            class_names = dataset_info['class_names']
        else:
            # Fallback: get from samples keys
            class_names = list(dataset_info['samples'].keys())
        
        class_counts = [len(dataset_info['samples'][name]) for name in class_names]
        
        bars = ax1.bar(class_names, class_counts, color=[self.colors['cyan'], self.colors['magenta'],
                                                         self.colors['yellow'], self.colors['green'],
                                                         self.colors['red'], self.colors['blue']])
        ax1.set_xlabel('Defect Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Split distribution (if available)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Count samples per split
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        for class_samples in dataset_info['samples'].values():
            for sample in class_samples:
                split_counts[sample['split']] += 1
        
        wedges, texts, autotexts = ax2.pie(split_counts.values(), labels=split_counts.keys(),
                                            autopct='%1.1f%%', startangle=90,
                                            colors=[self.colors['green'], self.colors['yellow'], self.colors['red']])
        
        for text in texts:
            text.set_color('white')
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax2.set_title('Train/Val/Test Split', fontsize=14, fontweight='bold', color=self.colors['yellow'])
        
        plt.suptitle('Dataset Overview', fontsize=18, fontweight='bold', 
                    color=self.colors['magenta'], y=1.0)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        print(f"✅ Saved dataset overview to {save_name}")
