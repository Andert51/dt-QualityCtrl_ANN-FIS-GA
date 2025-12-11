"""
CyberCore-QC: Hybrid Intelligent Quality Control System
========================================================
Main orchestrator combining CNN, FIS, and GA for industrial quality control.

Author: CyberCore AI Lab
Version: 1.0.0
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent Tcl threading errors
import torch
import numpy as np
import pandas as pd
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
from cnn_model import DefectCNN, DefectDataset, get_data_transforms
from fuzzy_system import FuzzyQualityController
from genetic_algorithm import GeneticAlgorithm
from visualizations import VisualizationHub
from kaggle_loader import KaggleDatasetLoader

# Import enhanced components
from enhanced_trainer import EnhancedCNNTrainer
from enhanced_ga import EnhancedGeneticOptimizer
from enhanced_visualizations import AnimatedVisualizer
from validation import SystemValidator, DataValidator, ConfigValidator
from logger import initialize_logging, get_logger


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
        
        # Force UTF-8 encoding for Windows compatibility with Unicode characters
        import io
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        
        self.console = Console(force_terminal=True, legacy_windows=False)
        
        # Initialize logging
        initialize_logging(self.workspace_dir / 'logs')
        self.logger = get_logger('CyberCoreQC')
        
        # Directory structure
        self.input_dir = self.workspace_dir / 'input' / 'dataset'
        self.output_dir = self.workspace_dir / 'output'
        self.models_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.viz_dir = self.results_dir / 'visualizations'
        
        # Create directories
        for d in [self.input_dir, self.models_dir, self.results_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Validators
        self.system_validator = SystemValidator(self.console)
        self.data_validator = DataValidator()
        self.config_validator = ConfigValidator()
        
        # Animated visualizer
        self.animator = AnimatedVisualizer(self.viz_dir)
        
        # System components
        self.cnn_model: Optional[DefectCNN] = None
        self.cnn_trainer: Optional[EnhancedCNNTrainer] = None
        self.fis: Optional[FuzzyQualityController] = None
        self.ga: Optional[EnhancedGeneticOptimizer] = None
        self.viz_hub: Optional[VisualizationHub] = None
        
        # Data
        self.dataset_metadata: Optional[Dict] = None
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None
        
        # Device - será configurado por usuario
        self.device = None
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            self.logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("GPU no disponible, usando CPU")
        
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
        
        # Device info
        if self.device is None:
            device_str = "Not Configured"
        elif self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
            device_str = f"GPU ({gpu_name})"
        else:
            device_str = "CPU"
        
        info = Text(
            f"Device: {device_str} | PyTorch {torch.__version__} | Scikit-Fuzzy Ready",
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
    
    def initialize_system(self, use_kaggle: bool = False, kaggle_dataset: str = 'neu'):
        """
        Initialize or load system components.
        
        Args:
            use_kaggle: Whether to try loading Kaggle dataset
            kaggle_dataset: Which Kaggle dataset to use ('neu', 'casting')
        """
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
            
            dataset_loaded = False
            
            # Try loading Kaggle dataset if requested
            if use_kaggle:
                try:
                    progress.update(task1, description=f"[cyan]Loading Kaggle dataset ({kaggle_dataset})...")
                    kaggle_loader = KaggleDatasetLoader(self.workspace_dir)
                    self.dataset_metadata = kaggle_loader.load_dataset(kaggle_dataset)
                    
                    if self.dataset_metadata:
                        dataset_loaded = True
                        self.console.print(f" [green]Loaded Kaggle dataset: {kaggle_dataset}")
                    else:
                        self.console.print(f" [yellow]Failed to load Kaggle dataset, falling back to synthetic")
                except Exception as e:
                    self.console.print(f" [yellow]Kaggle load error: {e}, using synthetic data")
            
            # Try existing metadata
            if not dataset_loaded and metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.dataset_metadata = json.load(f)
                dataset_loaded = True
                self.console.print(" [green]Loaded existing dataset metadata")
            
            # Generate synthetic dataset as fallback
            if not dataset_loaded:
                progress.update(task1, description="[yellow]Generating synthetic dataset...")
                generator = SyntheticDefectGenerator(img_size=(224, 224))
                self.dataset_metadata = generator.generate_dataset(
                    self.input_dir,
                    samples_per_class=100,
                    split_ratios=(0.7, 0.15, 0.15)
                )
                self.console.print(" [green]Generated synthetic dataset")
            
            progress.advance(task1)
            
            # Load data into PyTorch
            progress.update(task1, description="[cyan]Creating data loaders...")
            
            # Seleccionar dispositivo
            if self.device is None:
                self._select_device()
            
            self._create_data_loaders()
            progress.advance(task1)
            
            # Initialize CNN
            progress.update(task1, description="[cyan]Initializing CNN...")
            self.cnn_model = DefectCNN(num_classes=6, use_pretrained=True)
            self.cnn_trainer = EnhancedCNNTrainer(
                self.cnn_model, 
                device=self.device,
                console=self.console,
                enable_validation=True,
                enable_logging=True
            )
            progress.advance(task1)
            
            # Initialize FIS
            progress.update(task1, description="[cyan]Initializing Fuzzy System...")
            self.fis = FuzzyQualityController()
            progress.advance(task1)
        
        self.console.print("\n [bold green]System Initialized Successfully!")
        
        # Show dataset overview visualization
        if self.viz_hub and self.dataset_metadata:
            self.viz_hub.plot_dataset_overview(self.dataset_metadata)
    
    def _select_device(self):
        """Seleccionar dispositivo de cómputo (GPU/CPU)."""
        if not self.use_gpu:
            self.device = 'cpu'
            self.console.print("\n[yellow]  GPU no disponible, usando CPU[/yellow]")
            return
        
        # Mostrar información de GPU
        from gpu_optimizer import GPUOptimizer
        gpu_opt = GPUOptimizer()
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]🎮 CONFIGURACIÓN DE DISPOSITIVO[/bold cyan]")
        self.console.print("="*60)
        
        # Info de GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        self.console.print(f"\n[green] GPU Detectada:[/green] {gpu_name}")
        self.console.print(f"[cyan]   Memoria Total:[/cyan] {gpu_memory:.2f} GB")
        self.console.print(f"[cyan]   CUDA Version:[/cyan] {torch.version.cuda}")
        
        # Estimación de velocidad
        self.console.print("\n[bold yellow]📊 Estimación de Rendimiento:[/bold yellow]")
        self.console.print("[green]   GPU:[/green] ~10-50x más rápido que CPU")
        self.console.print("[green]   Batch size:[/green] 64-128 (vs 32 en CPU)")
        self.console.print("[green]   Mixed Precision:[/green] FP16 habilitado (2x boost)")
        
        # Selección
        choices = [
            "🚀 GPU (Recomendado - Mucho más rápido)",
            "🐌 CPU (Solo para debugging)"
        ]
        
        choice = questionary.select(
            "\nSelecciona dispositivo de cómputo:",
            choices=choices,
            style=self.custom_style
        ).ask()
        
        if "GPU" in choice:
            self.device = 'cuda'
            self.console.print("\n[bold green] Modo GPU activado - Entrenamiento acelerado[/bold green]")
            gpu_opt.print_gpu_info()
        else:
            self.device = 'cpu'
            self.console.print("\n[yellow]  Modo CPU seleccionado - Entrenamiento lento[/yellow]")
        
        self.logger.info(f"Device selected: {self.device}")
    
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
        
        # Create data loaders with GPU optimization
        from gpu_optimizer import GPUOptimizer
        gpu_opt = GPUOptimizer()
        gpu_config = gpu_opt.get_gpu_recommendations()
        
        num_workers = gpu_config['num_workers']
        pin_memory = gpu_config['pin_memory']
        batch_size = gpu_config['batch_size']
        
        self.console.print(f"[cyan]⚡ DataLoader config: batch_size={batch_size}, workers={num_workers}, pin_memory={pin_memory}[/cyan]")
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def train_cnn(self):
        """Train the CNN model."""
        self.console.print("\n[bold cyan]╔══════════════════════════════════════════════╗")
        self.console.print("[bold cyan]║        CNN TRAINING SEQUENCE INITIATED       ║")
        self.console.print("[bold cyan]╚══════════════════════════════════════════════╝\n")
        
        # Show GPU info if available
        if torch.cuda.is_available():
            from gpu_optimizer import GPUOptimizer
            GPUOptimizer.print_gpu_info()
        
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
        
        # Create animated training GIF
        self.console.print("\n[cyan]Creating animated training visualization...")
        self.animator.create_training_animation(history, 'training_animation.gif')
        
        # Create detailed metrics plot
        self.animator.create_detailed_metrics_plot(history, 'detailed_metrics.png')
        
        # Validation predictions for confusion matrix
        val_metrics = self.cnn_trainer.validate(self.val_loader)
        
        # Handle different metadata formats
        if 'classes' in self.dataset_metadata and isinstance(self.dataset_metadata['classes'], dict):
            class_names = list(self.dataset_metadata['classes'].values())
        elif 'class_names' in self.dataset_metadata:
            class_names = self.dataset_metadata['class_names']
        else:
            class_names = list(self.dataset_metadata['samples'].keys())
        
        self.viz_hub.plot_confusion_matrix(
            val_metrics['labels'],
            val_metrics['predictions'],
            class_names
        )
        
        self.console.print(" [green]CNN training complete and visualizations saved!")
    
    def run_genetic_optimization(self):
        """Run genetic algorithm to optimize FIS parameters."""
        self.console.print("\n[bold magenta]╔══════════════════════════════════════════════╗")
        self.console.print("[bold magenta]║        GENETIC OPTIMIZATION INITIATED        ║")
        self.console.print("[bold magenta]╚══════════════════════════════════════════════╝\n")
        
        # Ensure CNN is trained
        if not self.cnn_model:
            self.console.print("[red]  CNN not initialized. Please train CNN first.")
            return
        
        # Get validation predictions from CNN (use subset for faster GA)
        self.console.print("[cyan]Collecting CNN predictions for optimization...")
        self.cnn_model.eval()
        
        defect_probs = []
        true_labels = []
        
        # Use only first 100 samples for GA optimization (much faster)
        max_samples = 100
        sample_count = 0
        
        with torch.no_grad():
            for images, labels, severities, _ in self.val_loader:
                images = images.to(self.device)
                _, defect_logits = self.cnn_model(images)
                # Convert logits to probabilities using sigmoid
                defect_prob = torch.sigmoid(defect_logits)
                defect_probs.extend(defect_prob.cpu().numpy().flatten())
                true_labels.extend(labels.numpy())
                
                sample_count += len(labels)
                if sample_count >= max_samples:
                    # Trim to exact count
                    defect_probs = defect_probs[:max_samples]
                    true_labels = true_labels[:max_samples]
                    break
        
        self.console.print(f"[green]Using {len(defect_probs)} samples for optimization (faster convergence)")
        
        # Simulate material fragility (in real scenario, would come from sensors)
        material_fragilities = np.random.uniform(0.2, 0.8, len(defect_probs))
        
        # Define fitness function
        def fitness_function(params: np.ndarray) -> float:
            """
            Evaluate fitness of FIS parameters.
            Higher accuracy = higher fitness.
            """
            # Convert numpy array to chromosome dict
            chromosome = {
                'defect_low': (params[0], params[1], params[2]),
                'defect_medium': (params[3], params[4], params[5]),
                'defect_high': (params[6], params[7], params[8]),
                'frag_low': (params[9], params[10], params[11]),
                'frag_medium': (params[12], params[13], params[14]),
                'frag_high': (params[15], params[16], params[17]),
                'sev_low': (params[18], params[19], params[20]),
                'sev_medium': (params[21], params[22], params[23]),
                'sev_high': (params[24], params[25], params[26])
            }
            
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
                        expected_decision = 'PASS'
                    elif severity < 5:  # Low severity defects
                        expected_decision = 'INSPECT'
                    else:
                        expected_decision = 'REJECT'
                    
                    # Simple accuracy metric
                    if true_label == 0 and result['decision'] == 'PASS':
                        correct += 1
                    elif true_label > 0 and result['decision'] != 'PASS':
                        correct += 1
                except:
                    pass
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
        
        # Parameter bounds for GA (27 parameters total: 9 membership functions x 3 points each)
        bounds = [(0, 0.3)] * 3 + [(0.2, 0.6)] * 3 + [(0.5, 1.0)] * 3  # Defect probability
        bounds += [(0, 0.3)] * 3 + [(0.2, 0.6)] * 3 + [(0.5, 1.0)] * 3  # Material fragility
        bounds += [(0, 3)] * 3 + [(2, 7)] * 3 + [(6, 10)] * 3  # Severity output
        
        # Initialize and run enhanced GA with optimized parameters
        self.ga = EnhancedGeneticOptimizer(
            fitness_function=fitness_function,
            n_params=27,
            bounds=bounds,
            population_size=30,  # Reduced for faster execution (40% faster)
            mutation_rate=0.15,
            crossover_rate=0.8,
            elite_size=5,
            console=self.console,
            enable_logging=True,
            adaptive=True
        )
        
        results = self.ga.optimize(
            n_generations=30,  # Reduced from 50 for faster convergence
            target_fitness=0.95,
            patience=10,  # More aggressive early stopping
            show_animation=True
        )
        
        # Convert best solution back to chromosome dict
        best_params = results['best_solution']
        best_chromosome = {
            'defect_low': (best_params[0], best_params[1], best_params[2]),
            'defect_medium': (best_params[3], best_params[4], best_params[5]),
            'defect_high': (best_params[6], best_params[7], best_params[8]),
            'frag_low': (best_params[9], best_params[10], best_params[11]),
            'frag_medium': (best_params[12], best_params[13], best_params[14]),
            'frag_high': (best_params[15], best_params[16], best_params[17]),
            'sev_low': (best_params[18], best_params[19], best_params[20]),
            'sev_medium': (best_params[21], best_params[22], best_params[23]),
            'sev_high': (best_params[24], best_params[25], best_params[26])
        }
        
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
        
        # Create GA evolution animation
        self.console.print("\n[cyan]Creating animated GA evolution...")
        self.animator.create_ga_evolution_animation(
            generations=list(range(1, len(results['best_fitness_history']) + 1)),
            best_fitness=results['best_fitness_history'],
            avg_fitness=results['avg_fitness_history'],
            diversity=results['diversity_history']
        )
        
        self.console.print(f" [green]Genetic optimization complete! Best fitness: {results['best_fitness']:.4f}")
    
    def visual_analysis_hub(self):
        """Comprehensive visual analysis."""
        self.console.print("\n[bold yellow]╔══════════════════════════════════════════════╗")
        self.console.print("[bold yellow]║            VISUAL ANALYSIS HUB               ║")
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
            self.console.print("[red]  Models not initialized. Please initialize system first.")
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
                    self.console.print("[yellow]  GA not run yet. Skipping GA visualization.")
            
            if choice == "Test Results Grid" or choice == "All Visualizations":
                self._generate_results_grid()
            
            if choice == "Dataset Overview" or choice == "All Visualizations":
                self.viz_hub.plot_dataset_overview(self.dataset_metadata)
            
            progress.advance(task)
        
        self.console.print(f"\n [green]Visualizations saved to: {self.viz_dir}")
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
            class_logits, defect_logits = self.cnn_model(images)
            # Convert logits to probabilities using sigmoid
            defect_probs = torch.sigmoid(defect_logits)
        
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
                cnn_path = self.models_dir / 'best_cnn_model.pth'
                self.cnn_trainer.save_checkpoint(str(cnn_path))
                self.console.print(f" [green]Saved CNN to {cnn_path}")
            
            # Save FIS
            if self.fis:
                fis_path = self.models_dir / 'fis_params.pkl'
                self.fis.save(str(fis_path))
                self.console.print(f" [green]Saved FIS to {fis_path}")
            
            self.console.print(" [green]All models saved successfully!")
        
        elif action == "Load Models":
            # Load CNN
            cnn_path = self.models_dir / 'best_cnn_model.pth'
            if cnn_path.exists() and self.cnn_trainer:
                self.cnn_trainer.load_checkpoint(str(cnn_path))
                self.console.print(f" [green]Loaded CNN from {cnn_path}")
            
            # Load FIS
            fis_path = self.models_dir / 'fis_params.pkl'
            if fis_path.exists() and self.fis:
                self.fis.load(str(fis_path))
                self.console.print(f" [green]Loaded FIS from {fis_path}")
            
            self.console.print(" [green]All models loaded successfully!")
    
    def test_realtime_samples(self):
        """Test trained model with real-time sample images."""
        if not self.cnn_model or not self.fis:
            self.console.print("[red]  Please initialize and train the system first!")
            return
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan] Real-Time Quality Control Testing")
        self.console.print("="*60 + "\n")
        
        # Ask for test source
        test_source = questionary.select(
            "Select test image source:",
            choices=[
                "Test from current dataset (random samples)",
                "Test from custom image file",
                "Test batch from folder",
                "← Back"
            ],
            style=self.custom_style
        ).ask()
        
        if test_source == "← Back":
            return
        
        self.cnn_model.eval()
        transforms = get_data_transforms(img_size=224)['val']  # Use validation transforms (no augmentation)
        
        test_results = []
        
        if test_source == "Test from current dataset (random samples)":
            # Sample random images from test set
            num_samples = int(questionary.text(
                "How many samples to test?",
                default="5",
                style=self.custom_style
            ).ask())
            
            # Get random samples
            sample_indices = np.random.choice(len(self.test_loader.dataset), min(num_samples, len(self.test_loader.dataset)), replace=False)
            
            for idx in sample_indices:
                img_tensor, label, severity, img_path = self.test_loader.dataset[idx]
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    class_logits, defect_logits = self.cnn_model(img_tensor)
                    pred_class = torch.argmax(class_logits, dim=1).item()
                    # Convert logits to probability using sigmoid
                    defect_prob = torch.sigmoid(defect_logits).item()
                
                # FIS prediction
                material_fragility = np.random.uniform(0.3, 0.7)
                fis_result = self.fis.predict(defect_prob, material_fragility)
                
                test_results.append({
                    'image_path': img_path,
                    'true_class': self.dataset_metadata['class_names'][label],
                    'pred_class': self.dataset_metadata['class_names'][pred_class],
                    'defect_prob': defect_prob,
                    'severity_score': fis_result['severity_score'],
                    'decision': fis_result['decision'],
                    'correct': label == pred_class
                })
        
        elif test_source == "Test from custom image file":
            img_path = questionary.text(
                "Enter image path:",
                style=self.custom_style
            ).ask()
            
            if not Path(img_path).exists():
                self.console.print(f"[red] File not found: {img_path}")
                return
            
            # Load and process image
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                class_logits, defect_prob = self.cnn_model(img_tensor)
                pred_class = torch.argmax(class_logits, dim=1).item()
                defect_prob = defect_prob.item()
            
            # FIS prediction
            material_fragility = float(questionary.text(
                "Enter material fragility (0.0-1.0):",
                default="0.5",
                style=self.custom_style
            ).ask())
            
            fis_result = self.fis.predict(defect_prob, material_fragility)
            
            test_results.append({
                'image_path': img_path,
                'true_class': 'Unknown',
                'pred_class': self.dataset_metadata['class_names'][pred_class],
                'defect_prob': defect_prob,
                'severity_score': fis_result['severity_score'],
                'decision': fis_result['decision'],
                'correct': None
            })
        
        elif test_source == "Test batch from folder":
            folder_path = questionary.text(
                "Enter folder path containing images:",
                style=self.custom_style
            ).ask()
            
            folder = Path(folder_path)
            if not folder.exists():
                self.console.print(f"[red] Folder not found: {folder_path}")
                return
            
            # Find images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(folder.glob(ext))
            
            if not image_files:
                self.console.print(f"[red] No images found in folder")
                return
            
            from PIL import Image
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"[cyan]Testing {len(image_files)} images...", total=len(image_files))
                
                for img_path in image_files:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transforms(img).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            class_logits, defect_prob = self.cnn_model(img_tensor)
                            pred_class = torch.argmax(class_logits, dim=1).item()
                            defect_prob = defect_prob.item()
                        
                        material_fragility = np.random.uniform(0.3, 0.7)
                        fis_result = self.fis.predict(defect_prob, material_fragility)
                        
                        test_results.append({
                            'image_path': str(img_path),
                            'true_class': 'Unknown',
                            'pred_class': self.dataset_metadata['class_names'][pred_class],
                            'defect_prob': defect_prob,
                            'severity_score': fis_result['severity_score'],
                            'decision': fis_result['decision'],
                            'correct': None
                        })
                        
                        progress.advance(task)
                    except Exception as e:
                        self.console.print(f"[yellow]⚠️  Failed to process {img_path.name}: {e}")
        
        # Display results
        self._display_test_results(test_results)
        
        # Save results
        results_file = self.results_dir / f'realtime_test_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.console.print(f"\n💾 [green]Results saved to: {results_file}")
    
    def _display_test_results(self, results: List[Dict]):
        """Display test results in a formatted table."""
        table = Table(title="🧪 Test Results", border_style="cyan", box=box.ROUNDED)
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Image", style="cyan", no_wrap=False)
        table.add_column("True", style="yellow")
        table.add_column("Predicted", style="magenta")
        table.add_column("Defect Prob", style="red")
        table.add_column("Severity", style="orange1")
        table.add_column("Decision", style="bold")
        table.add_column("✓", style="green")
        
        for idx, result in enumerate(results, 1):
            img_name = Path(result['image_path']).name
            correct_mark = "✓" if result['correct'] else ("✗" if result['correct'] is False else "-")
            correct_color = "green" if result['correct'] else ("red" if result['correct'] is False else "dim")
            
            decision_color = "green" if result['decision'] == "PASS" else ("red" if result['decision'] == "REJECT" else "yellow")
            
            table.add_row(
                str(idx),
                img_name[:30],
                result['true_class'],
                result['pred_class'],
                f"{result['defect_prob']:.3f}",
                f"{result['severity_score']:.2f}",
                f"[{decision_color}]{result['decision']}[/{decision_color}]",
                f"[{correct_color}]{correct_mark}[/{correct_color}]"
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Statistics
        if any(r['correct'] is not None for r in results):
            accuracy = sum(1 for r in results if r['correct']) / sum(1 for r in results if r['correct'] is not None)
            self.console.print(f"\n📊 [bold cyan]Accuracy: {accuracy*100:.2f}%")
        
        pass_count = sum(1 for r in results if r['decision'] == 'PASS')
        reject_count = sum(1 for r in results if r['decision'] == 'REJECT')
        inspect_count = sum(1 for r in results if r['decision'] == 'INSPECT')
        
        self.console.print(f"✅ PASS: {pass_count} | ❌ REJECT: {reject_count} | ⚠️  INSPECT: {inspect_count}")
    
    def main_menu(self):
        """Display main interactive menu."""
        while True:
            self.show_banner()
            self.show_status_dashboard()
            
            menu_options = [
                "🔧 Initialize System / Load Data",
                "🧠 Train CNN Model",
                "🧬 Run Genetic Optimization",
                "🧪 Test Model (Real-time)",
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
                # Ask for dataset type
                dataset_choice = questionary.select(
                    "Select dataset source:",
                    choices=[
                        "🔬 Generate Synthetic Data",
                        "📦 Load Kaggle Dataset (NEU Surface Defects)",
                        "🏭 Load Kaggle Dataset (Casting Defects)",
                        "📁 Use Existing Dataset",
                        "← Back"
                    ],
                    style=self.custom_style
                ).ask()
                
                if dataset_choice == "← Back":
                    continue
                elif dataset_choice == "🔬 Generate Synthetic Data":
                    self.initialize_system(use_kaggle=False)
                elif dataset_choice == "📦 Load Kaggle Dataset (NEU Surface Defects)":
                    self.initialize_system(use_kaggle=True, kaggle_dataset='neu')
                elif dataset_choice == "🏭 Load Kaggle Dataset (Casting Defects)":
                    self.initialize_system(use_kaggle=True, kaggle_dataset='casting')
                elif dataset_choice == "📁 Use Existing Dataset":
                    self.initialize_system(use_kaggle=False)
                
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
            
            elif choice == "🧪 Test Model (Real-time)":
                if not self.cnn_model:
                    self.console.print("[red]⚠️  Please initialize and train system first!")
                    input("\nPress Enter to continue...")
                else:
                    self.test_realtime_samples()
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
                self.console.print("[bold yellow]   Thank you for using QualityCtrl dt!")
                self.console.print("[bold magenta]      Github: https://github.com/Andert51")
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

