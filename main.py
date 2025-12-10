"""
CyberCore-QC: Hybrid Intelligent Quality Control System
========================================================
Main orchestrator combining CNN, FIS, and GA for industrial quality control.

Author: CyberCore AI Lab
Version: 1.0.0
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich import box
from rich.text import Text
from rich.align import Align
import pyfiglet
import questionary
from questionary import Style

# Import custom modules
sys.path.append(str(Path(__file__).parent / 'src'))
from data_generator import SyntheticDefectGenerator
from cnn_model import DefectCNN, DefectCNNTrainer, DefectDataset, get_data_transforms
from fuzzy_system import FuzzyQualityController
from genetic_algorithm import GeneticAlgorithm
from visualizations import VisualizationHub


class CyberCoreQC:
    """
    Main orchestrator for the CyberCore-QC system.
    Combines CNN, Fuzzy Logic, and Genetic Algorithm for quality control.
    """
    
    def __init__(self, workspace_dir: Path = Path('.')):
        """
        Initialize the CyberCore-QC system.
        
        Args:
            workspace_dir: Root workspace directory
        """
        self.workspace_dir = Path(workspace_dir)
        self.console = Console()
        
        # Directory structure
        self.input_dir = self.workspace_dir / 'input' / 'dataset'
        self.output_dir = self.workspace_dir / 'output'
        self.models_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.viz_dir = self.results_dir / 'visualizations'
        
        # Create directories
        for d in [self.input_dir, self.models_dir, self.results_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # System components
        self.cnn_model: Optional[DefectCNN] = None
        self.cnn_trainer: Optional[DefectCNNTrainer] = None
        self.fis: Optional[FuzzyQualityController] = None
        self.ga: Optional[GeneticAlgorithm] = None
        self.viz_hub: Optional[VisualizationHub] = None
        
        # Data
        self.dataset_metadata: Optional[Dict] = None
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cyberpunk style for questionary
        self.custom_style = Style([
            ('qmark', 'fg:#00FFFF bold'),
            ('question', 'fg:#FFFF00 bold'),
            ('answer', 'fg:#00FF00 bold'),
            ('pointer', 'fg:#FF00FF bold'),
            ('highlighted', 'fg:#00FFFF bold'),
            ('selected', 'fg:#00FF00'),
            ('separator', 'fg:#666666'),
            ('instruction', 'fg:#888888'),
        ])
        
    def show_banner(self):
        """Display cyberpunk ASCII banner."""
        self.console.clear()
        
        banner = r"""
       ____      ____              ___ __        ________       __
  ____/ / /_    / __ \__  ______ _/ (_) /___  __/ ____/ /______/ /
 / __  / __/   / / / / / / / __ `/ / / __/ / / / /   / __/ ___/ / 
/ /_/ / /_    / /_/ / /_/ / /_/ / / / /_/ /_/ / /___/ /_/ /  / /  
\__,_/\__/____\___\_\__,_/\__,_/_/_/\__/\__, /\____/\__/_/  /_/   
        /_____/                        /____/                     
        """
        
        banner_text = Text(banner, style="bold cyan")
        
        subtitle = Text(
            "\nHybrid Intelligence AI Lab",
            style="bold magenta",
            justify="center"
        )
        
        info = Text(
            f"Device: {self.device.upper()} | PyTorch {torch.__version__} | Scikit-Fuzzy Ready",
            style="dim yellow",
            justify="center"
        )
        
        from rich.console import Group
        content = Align.center(
            Group(banner_text, subtitle, Text("\n"), info)
        )
        
        panel = Panel(
            content,
            border_style="bright_cyan",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
            title="[bold dim]v1.0[/]",
            title_align="right"
        )
        
        self.console.print(panel)
        self.console.print()
    
   
    
    def show_status_dashboard(self):
        """Display system status dashboard."""
        table = Table(title="🖥️  System Status", border_style="cyan", box=box.ROUNDED)
        
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        # CNN Status
        cnn_status = "Loaded" if self.cnn_model else "Not Initialized"
        cnn_color = "green" if self.cnn_model else "yellow"
        table.add_row("CNN Model", f"[{cnn_color}]{cnn_status}[/{cnn_color}]", 
                     f"ResNet18 Backbone" if self.cnn_model else "—")
        
        # FIS Status
        fis_status = "Ready" if self.fis else "Not Initialized"
        fis_color = "green" if self.fis else "yellow"
        table.add_row("Fuzzy System", f"[{fis_color}]{fis_status}[/{fis_color}]", 
                     "6 Input MFs, 3 Output MFs" if self.fis else "—")
        
        # Dataset Status
        dataset_status = "Loaded" if self.dataset_metadata else "Not Loaded"
        dataset_color = "green" if self.dataset_metadata else "yellow"
        dataset_detail = f"{self.dataset_metadata['total_samples']} samples" if self.dataset_metadata else "—"
        table.add_row("Dataset", f"[{dataset_color}]{dataset_status}[/{dataset_color}]", dataset_detail)
        
        # Visualization Hub
        viz_status = "Active" if self.viz_hub else "Inactive"
        viz_color = "green" if self.viz_hub else "yellow"
        table.add_row("Viz Hub", f"[{viz_color}]{viz_status}[/{viz_color}]", 
                     str(self.viz_dir) if self.viz_hub else "—")
        
        self.console.print(table)
        self.console.print()
    
    def initialize_system(self):
        """Initialize or load system components."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            
            task1 = progress.add_task("[cyan]Initializing System...", total=5)
            
            # Initialize Visualization Hub
            progress.update(task1, description="[cyan]Setting up Visualization Hub...")
            self.viz_hub = VisualizationHub(self.viz_dir)
            progress.advance(task1)
            
            # Check for existing dataset
            progress.update(task1, description="[cyan]Checking for dataset...")
            metadata_path = self.input_dir / 'metadata.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.dataset_metadata = json.load(f)
                self.console.print("✅ [green]Loaded existing dataset metadata")
            else:
                # Generate synthetic dataset
                progress.update(task1, description="[yellow]Generating synthetic dataset...")
                generator = SyntheticDefectGenerator(img_size=(224, 224))
                self.dataset_metadata = generator.generate_dataset(
                    self.input_dir,
                    samples_per_class=100,  # Smaller for quick demo
                    split_ratios=(0.7, 0.15, 0.15)
                )
                self.console.print("✅ [green]Generated synthetic dataset")
            
            progress.advance(task1)
            
            # Load data into PyTorch
            progress.update(task1, description="[cyan]Creating data loaders...")
            self._create_data_loaders()
            progress.advance(task1)
            
            # Initialize CNN
            progress.update(task1, description="[cyan]Initializing CNN...")
            self.cnn_model = DefectCNN(num_classes=6, use_pretrained=True)
            self.cnn_trainer = DefectCNNTrainer(self.cnn_model, device=self.device)
            progress.advance(task1)
            
            # Initialize FIS
            progress.update(task1, description="[cyan]Initializing Fuzzy System...")
            self.fis = FuzzyQualityController()
            progress.advance(task1)
        
        self.console.print("\n✨ [bold green]System Initialized Successfully!")
        
        # Show dataset overview visualization
        if self.viz_hub and self.dataset_metadata:
            self.viz_hub.plot_dataset_overview(self.dataset_metadata)
    
    def _create_data_loaders(self):
        """Create PyTorch data loaders from dataset."""
        transforms = get_data_transforms(img_size=224)
        
        # Collect paths and labels for each split
        splits_data = {'train': {'paths': [], 'labels': [], 'severities': []},
                      'val': {'paths': [], 'labels': [], 'severities': []},
                      'test': {'paths': [], 'labels': [], 'severities': []}}
        
        for class_name, samples in self.dataset_metadata['samples'].items():
            for sample in samples:
                split = sample['split']
                splits_data[split]['paths'].append(sample['path'])
                splits_data[split]['labels'].append(sample['class_id'])
                splits_data[split]['severities'].append(sample['severity'])
        
        # Create datasets
        train_dataset = DefectDataset(
            splits_data['train']['paths'],
            splits_data['train']['labels'],
            transform=transforms['train'],
            severity_scores=splits_data['train']['severities']
        )
        
        val_dataset = DefectDataset(
            splits_data['val']['paths'],
            splits_data['val']['labels'],
            transform=transforms['val'],
            severity_scores=splits_data['val']['severities']
        )
        
        test_dataset = DefectDataset(
            splits_data['test']['paths'],
            splits_data['test']['labels'],
            transform=transforms['val'],
            severity_scores=splits_data['test']['severities']
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=0
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=0
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=0
        )
    
    def train_cnn(self):
        """Train the CNN model."""
        self.console.print("\n[bold cyan]╔══════════════════════════════════════════════╗")
        self.console.print("[bold cyan]║     🧠 CNN TRAINING SEQUENCE INITIATED      ║")
        self.console.print("[bold cyan]╚══════════════════════════════════════════════╝\n")
        
        # Training parameters
        epochs = questionary.text(
            "Enter number of epochs:",
            default="15",
            style=self.custom_style
        ).ask()
        
        epochs = int(epochs)
        
        # Train
        history = self.cnn_trainer.train(
            self.train_loader,
            self.val_loader,
            epochs=epochs,
            lr=0.001
        )
        
        # Save model
        model_path = self.models_dir / 'best_cnn_model.pth'
        self.cnn_trainer.save_checkpoint(str(model_path))
        
        # Visualizations
        self.console.print("\n[yellow]Generating training visualizations...")
        self.viz_hub.plot_training_curves(history)
        self.viz_hub.plot_network_topology(self.cnn_model)
        
        # Validation predictions for confusion matrix
        val_metrics = self.cnn_trainer.validate(self.val_loader)
        class_names = list(self.dataset_metadata['classes'].values())
        self.viz_hub.plot_confusion_matrix(
            val_metrics['labels'],
            val_metrics['predictions'],
            class_names
        )
        
        self.console.print("✅ [green]CNN training complete and visualizations saved!")
    
    def run_genetic_optimization(self):
        """Run genetic algorithm to optimize FIS parameters."""
        self.console.print("\n[bold magenta]╔══════════════════════════════════════════════╗")
        self.console.print("[bold magenta]║    🧬 GENETIC OPTIMIZATION INITIATED        ║")
        self.console.print("[bold magenta]╚══════════════════════════════════════════════╝\n")
        
        # Ensure CNN is trained
        if not self.cnn_model:
            self.console.print("[red]⚠️  CNN not initialized. Please train CNN first.")
            return
        
        # Get validation predictions from CNN
        self.console.print("[cyan]Collecting CNN predictions for optimization...")
        self.cnn_model.eval()
        
        defect_probs = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels, severities, _ in self.val_loader:
                images = images.to(self.device)
                _, defect_prob = self.cnn_model(images)
                defect_probs.extend(defect_prob.cpu().numpy().flatten())
                true_labels.extend(labels.numpy())
        
        # Simulate material fragility (in real scenario, would come from sensors)
        material_fragilities = np.random.uniform(0.2, 0.8, len(defect_probs))
        
        # Define fitness function
        def fitness_function(chromosome: Dict) -> float:
            """
            Evaluate fitness of FIS parameters.
            Higher accuracy = higher fitness.
            """
            # Create temporary FIS with these parameters
            temp_fis = FuzzyQualityController()
            temp_fis.update_from_triangular_params(chromosome)
            
            # Make predictions
            correct = 0
            total = len(defect_probs)
            
            for defect_prob, mat_frag, true_label in zip(defect_probs, material_fragilities, true_labels):
                try:
                    result = temp_fis.predict(float(defect_prob), float(mat_frag))
                    severity = result['severity_score']
                    
                    # Ground truth: if true_label > 0, it should be rework or reject
                    if true_label == 0:  # Normal
                        expected_decision = 'Accept'
                    elif severity < 0.5:  # Low severity defects
                        expected_decision = 'Rework'
                    else:
                        expected_decision = 'Reject'
                    
                    # Simple accuracy metric
                    if true_label == 0 and result['decision'] == 'Accept':
                        correct += 1
                    elif true_label > 0 and result['decision'] != 'Accept':
                        correct += 1
                except:
                    pass
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
        
        # Initialize and run GA
        self.ga = GeneticAlgorithm(
            population_size=40,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=5
        )
        
        best_chromosome = self.ga.evolve(fitness_function, verbose=True)
        
        # Update FIS with best parameters
        self.fis.update_from_triangular_params(best_chromosome)
        
        # Save optimized FIS
        fis_path = self.models_dir / 'optimized_fis.pkl'
        self.fis.save(str(fis_path))
        
        # Visualizations
        self.console.print("\n[yellow]Generating GA visualizations...")
        self.viz_hub.plot_ga_evolution(self.ga.history)
        self.viz_hub.plot_membership_functions(self.fis, save_name="optimized_membership_functions.png")
        self.viz_hub.plot_fuzzy_surface_3d(self.fis)
        
        self.console.print(f"✅ [green]Genetic optimization complete! Best fitness: {self.ga.best_fitness:.4f}")
    
    def visual_analysis_hub(self):
        """Comprehensive visual analysis."""
        self.console.print("\n[bold yellow]╔══════════════════════════════════════════════╗")
        self.console.print("[bold yellow]║        📊 VISUAL ANALYSIS HUB               ║")
        self.console.print("[bold yellow]╚══════════════════════════════════════════════╝\n")
        
        options = [
            "CNN Activation Heatmaps",
            "Fuzzy Membership Functions",
            "3D Fuzzy Decision Surface",
            "GA Evolution Progress",
            "Test Results Grid",
            "Dataset Overview",
            "All Visualizations",
            "← Back to Main Menu"
        ]
        
        choice = questionary.select(
            "Select visualization:",
            choices=options,
            style=self.custom_style
        ).ask()
        
        if choice == "← Back to Main Menu":
            return
        
        # Ensure models are loaded
        if not self.cnn_model or not self.fis:
            self.console.print("[red]⚠️  Models not initialized. Please initialize system first.")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("[cyan]Generating visualizations...", total=1)
            
            if choice == "CNN Activation Heatmaps" or choice == "All Visualizations":
                # Get a sample image
                sample_batch = next(iter(self.test_loader))
                image, label, severity, path = sample_batch
                image = image[0:1].to(self.device)
                
                # Get feature maps
                feature_maps = self.cnn_model.get_feature_maps(image)
                self.viz_hub.plot_activation_heatmap(image[0], feature_maps)
            
            if choice == "Fuzzy Membership Functions" or choice == "All Visualizations":
                self.viz_hub.plot_membership_functions(self.fis)
            
            if choice == "3D Fuzzy Decision Surface" or choice == "All Visualizations":
                self.viz_hub.plot_fuzzy_surface_3d(self.fis)
            
            if choice == "GA Evolution Progress" or choice == "All Visualizations":
                if self.ga and self.ga.history['best_fitness']:
                    self.viz_hub.plot_ga_evolution(self.ga.history)
                else:
                    self.console.print("[yellow]⚠️  GA not run yet. Skipping GA visualization.")
            
            if choice == "Test Results Grid" or choice == "All Visualizations":
                self._generate_results_grid()
            
            if choice == "Dataset Overview" or choice == "All Visualizations":
                self.viz_hub.plot_dataset_overview(self.dataset_metadata)
            
            progress.advance(task)
        
        self.console.print(f"\n✅ [green]Visualizations saved to: {self.viz_dir}")
        self.console.print("[dim]You can open the HTML files in a browser for interactive 3D plots.")
    
    def _generate_results_grid(self):
        """Generate results grid with predictions."""
        self.cnn_model.eval()
        
        images_list = []
        predictions_list = []
        
        # Get samples from test set
        sample_batch = next(iter(self.test_loader))
        images, labels, severities, paths = sample_batch
        
        images = images.to(self.device)
        
        with torch.no_grad():
            class_logits, defect_probs = self.cnn_model(images)
        
        # Make FIS predictions
        for i in range(min(16, len(images))):
            img = images[i]
            defect_prob = defect_probs[i].item()
            mat_frag = np.random.uniform(0.2, 0.8)  # Simulated sensor data
            
            fis_result = self.fis.predict(defect_prob, mat_frag)
            
            images_list.append(img)
            predictions_list.append(fis_result)
        
        self.viz_hub.plot_results_grid(images_list, predictions_list)
    
    def save_load_models(self):
        """Save or load model checkpoints."""
        action = questionary.select(
            "Choose action:",
            choices=["Save Models", "Load Models", "← Back"],
            style=self.custom_style
        ).ask()
        
        if action == "← Back":
            return
        
        if action == "Save Models":
            # Save CNN
            if self.cnn_trainer:
                cnn_path = self.models_dir / 'cnn_checkpoint.pth'
                self.cnn_trainer.save_checkpoint(str(cnn_path))
                self.console.print(f"✅ [green]Saved CNN to {cnn_path}")
            
            # Save FIS
            if self.fis:
                fis_path = self.models_dir / 'fis_params.pkl'
                self.fis.save(str(fis_path))
                self.console.print(f"✅ [green]Saved FIS to {fis_path}")
            
            self.console.print("✅ [green]All models saved successfully!")
        
        elif action == "Load Models":
            # Load CNN
            cnn_path = self.models_dir / 'cnn_checkpoint.pth'
            if cnn_path.exists() and self.cnn_trainer:
                self.cnn_trainer.load_checkpoint(str(cnn_path))
                self.console.print(f"✅ [green]Loaded CNN from {cnn_path}")
            
            # Load FIS
            fis_path = self.models_dir / 'fis_params.pkl'
            if fis_path.exists() and self.fis:
                self.fis.load(str(fis_path))
                self.console.print(f"✅ [green]Loaded FIS from {fis_path}")
            
            self.console.print("✅ [green]All models loaded successfully!")
    
    def main_menu(self):
        """Display main interactive menu."""
        while True:
            self.show_banner()
            self.show_status_dashboard()
            
            menu_options = [
                "🔧 Initialize System / Load Data",
                "🧠 Train CNN Model",
                "🧬 Run Genetic Optimization",
                "📊 Visual Analysis Hub",
                "💾 Save/Load Models",
                "🚪 Exit"
            ]
            
            choice = questionary.select(
                "Select operation:",
                choices=menu_options,
                style=self.custom_style
            ).ask()
            
            if choice == "🔧 Initialize System / Load Data":
                self.initialize_system()
                input("\nPress Enter to continue...")
            
            elif choice == "🧠 Train CNN Model":
                if not self.cnn_model:
                    self.console.print("[red]⚠️  Please initialize system first!")
                    input("\nPress Enter to continue...")
                else:
                    self.train_cnn()
                    input("\nPress Enter to continue...")
            
            elif choice == "🧬 Run Genetic Optimization":
                if not self.fis:
                    self.console.print("[red]⚠️  Please initialize system first!")
                    input("\nPress Enter to continue...")
                else:
                    self.run_genetic_optimization()
                    input("\nPress Enter to continue...")
            
            elif choice == "📊 Visual Analysis Hub":
                if not self.viz_hub:
                    self.console.print("[red]⚠️  Please initialize system first!")
                    input("\nPress Enter to continue...")
                else:
                    self.visual_analysis_hub()
                    input("\nPress Enter to continue...")
            
            elif choice == "💾 Save/Load Models":
                self.save_load_models()
                input("\nPress Enter to continue...")
            
            elif choice == "🚪 Exit":
                self.console.print("\n[bold cyan]═══════════════════════════════════════")
                self.console.print("[bold yellow]   Thank you for using CyberCore-QC!")
                self.console.print("[bold magenta]      Stay cyber. Stay secure.")
                self.console.print("[bold cyan]═══════════════════════════════════════\n")
                break


def main():
    """Main entry point."""
    # Create system instance
    workspace = Path(__file__).parent
    system = CyberCoreQC(workspace)
    
    try:
        # Run main menu
        system.main_menu()
    except KeyboardInterrupt:
        system.console.print("\n\n[red]⚠️  Operation interrupted by user.")
    except Exception as e:
        system.console.print(f"\n[red]❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

