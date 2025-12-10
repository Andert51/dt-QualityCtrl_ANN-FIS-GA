"""
CyberCore-QC: Quick Demo Script
================================
Demonstrates the hybrid system with a minimal example.
Faster than the full interactive system for testing.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_generator import SyntheticDefectGenerator
from cnn_model import DefectCNN, DefectCNNTrainer, DefectDataset, get_data_transforms
from fuzzy_system import FuzzyQualityController
from genetic_algorithm import GeneticAlgorithm
from visualizations import VisualizationHub

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import pyfiglet


def main():
    console = Console()
    
    # Banner
    banner = pyfiglet.figlet_format("CYBERCORE-QC", font="slant")
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print("[bold yellow]Quick Demo - Hybrid Intelligence in Action[/bold yellow]\n")
    
    # Setup paths
    workspace = Path(__file__).parent
    input_dir = workspace / 'input' / 'dataset'
    output_dir = workspace / 'output'
    viz_dir = output_dir / 'results' / 'visualizations'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[cyan]Device: {device.upper()}[/cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # 1. Generate mini dataset
        task = progress.add_task("[cyan]Generating mini dataset (120 images)...", total=5)
        generator = SyntheticDefectGenerator(img_size=(224, 224))
        metadata = generator.generate_dataset(
            input_dir,
            samples_per_class=20,  # Mini dataset
            split_ratios=(0.7, 0.15, 0.15)
        )
        progress.advance(task)
        
        # 2. Create data loaders
        progress.update(task, description="[cyan]Creating data loaders...")
        transforms = get_data_transforms(img_size=224)
        
        # Collect data
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        
        for class_name, samples in metadata['samples'].items():
            for sample in samples:
                if sample['split'] == 'train':
                    train_paths.append(sample['path'])
                    train_labels.append(sample['class_id'])
                elif sample['split'] == 'val':
                    val_paths.append(sample['path'])
                    val_labels.append(sample['class_id'])
        
        train_dataset = DefectDataset(train_paths, train_labels, transform=transforms['train'])
        val_dataset = DefectDataset(val_paths, val_labels, transform=transforms['val'])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        progress.advance(task)
        
        # 3. Quick CNN training
        progress.update(task, description="[cyan]Training CNN (5 epochs - quick demo)...")
        model = DefectCNN(num_classes=6, pretrained=True)
        trainer = DefectCNNTrainer(model, device=device)
        
        history = trainer.train(train_loader, val_loader, epochs=5, lr=0.001)
        progress.advance(task)
        
        # 4. Initialize and test FIS
        progress.update(task, description="[cyan]Testing Fuzzy Inference System...")
        fis = FuzzyQualityController()
        
        # Test predictions
        test_cases = [
            (0.1, 0.2),
            (0.5, 0.5),
            (0.9, 0.8)
        ]
        
        console.print("\n[bold yellow]Fuzzy System Test:[/bold yellow]")
        for defect_prob, mat_frag in test_cases:
            result = fis.predict(defect_prob, mat_frag)
            color = result['color']
            console.print(
                f"  Defect: {defect_prob:.1f}, Fragility: {mat_frag:.1f} "
                f"→ [{color}]{result['decision']}[/{color}] (Severity: {result['severity_score']:.2f})"
            )
        
        progress.advance(task)
        
        # 5. Quick GA optimization
        progress.update(task, description="[cyan]Running GA optimization (20 generations)...")
        
        # Get CNN predictions
        model.eval()
        defect_probs = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device)
                _, defect_prob = model(images)
                defect_probs.extend(defect_prob.cpu().numpy().flatten())
                true_labels.extend(labels.numpy())
        
        # Define simple fitness
        def fitness_function(chromosome):
            temp_fis = FuzzyQualityController()
            temp_fis.update_from_triangular_params(chromosome)
            
            correct = 0
            total = len(defect_probs)
            
            for defect_prob, true_label in zip(defect_probs, true_labels):
                try:
                    result = temp_fis.predict(float(defect_prob), 0.5)
                    if true_label == 0 and result['decision'] == 'Accept':
                        correct += 1
                    elif true_label > 0 and result['decision'] != 'Accept':
                        correct += 1
                except:
                    pass
            
            return correct / total if total > 0 else 0.0
        
        ga = GeneticAlgorithm(population_size=20, generations=20)
        best_chromosome = ga.evolve(fitness_function, verbose=False)
        fis.update_from_triangular_params(best_chromosome)
        
        progress.advance(task)
        
        # 6. Generate visualizations
        progress.update(task, description="[cyan]Creating visualizations...")
        viz_hub = VisualizationHub(viz_dir)
        
        viz_hub.plot_training_curves(history)
        viz_hub.plot_membership_functions(fis)
        viz_hub.plot_ga_evolution(ga.history)
        viz_hub.plot_dataset_overview(metadata)
        
        progress.advance(task)
    
    # Final summary
    console.print("\n[bold green]╔════════════════════════════════════════════╗[/bold green]")
    console.print("[bold green]║     ✅ DEMO COMPLETE - SYSTEM VERIFIED     ║[/bold green]")
    console.print("[bold green]╚════════════════════════════════════════════╝[/bold green]\n")
    
    console.print("[bold cyan]Results:[/bold cyan]")
    console.print(f"  📊 Dataset: {metadata['total_samples']} images generated")
    console.print(f"  🧠 CNN: {history['val_acc'][-1]:.2f}% validation accuracy")
    console.print(f"  🧬 GA: {ga.best_fitness:.4f} fitness achieved")
    console.print(f"  📁 Visualizations: {viz_dir}")
    
    console.print("\n[bold yellow]Next Steps:[/bold yellow]")
    console.print("  1. Run 'python main.py' for full interactive experience")
    console.print("  2. Train with more epochs (15-20) for better accuracy")
    console.print("  3. Check visualizations in output/results/visualizations/")
    
    console.print("\n[dim]Demo completed in ~2-5 minutes (depends on hardware)[/dim]")


if __name__ == "__main__":
    main()
