"""
Quick test script to verify enhanced components
Tests imports and basic functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("🔧 Testing Enhanced Components...\n")

# Test 1: UI Components
print("1️⃣ Testing UI Components...")
try:
    from ui_components import CyberpunkUI, TrainingProgressDisplay, GeneticAlgorithmDisplay
    from rich.console import Console
    
    console = Console()
    ui = CyberpunkUI(console)
    
    # Test panel creation
    panel = ui.create_title_panel("TEST PANEL", "Verification", style="success")
    console.print(panel)
    print("   ✅ UI Components OK\n")
except Exception as e:
    print(f"   ❌ UI Components FAILED: {e}\n")

# Test 2: Validation
print("2️⃣ Testing Validation...")
try:
    from validation import SystemValidator, DataValidator, ConfigValidator
    
    sys_val = SystemValidator(console)
    data_val = DataValidator()
    config_val = ConfigValidator()
    
    # Test environment validation
    result = sys_val.validate_environment()
    print(f"   Environment: {result.message}")
    print("   ✅ Validation OK\n")
except Exception as e:
    print(f"   ❌ Validation FAILED: {e}\n")

# Test 3: Logger
print("3️⃣ Testing Logger...")
try:
    from logger import initialize_logging, get_logger
    
    initialize_logging(Path('./test_logs'))
    logger = get_logger('TestLogger')
    
    logger.info("Test log message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    
    print("   ✅ Logger OK\n")
except Exception as e:
    print(f"   ❌ Logger FAILED: {e}\n")

# Test 4: Enhanced Trainer
print("4️⃣ Testing Enhanced Trainer...")
try:
    from enhanced_trainer import EnhancedCNNTrainer
    import torch
    import torch.nn as nn
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x), torch.rand(x.size(0), 1)
    
    model = DummyModel()
    trainer = EnhancedCNNTrainer(
        model, 
        device='cpu',
        console=console,
        enable_validation=True,
        enable_logging=True
    )
    
    print("   ✅ Enhanced Trainer OK\n")
except Exception as e:
    print(f"   ❌ Enhanced Trainer FAILED: {e}\n")

# Test 5: Enhanced GA
print("5️⃣ Testing Enhanced GA...")
try:
    from enhanced_ga import EnhancedGeneticOptimizer
    import numpy as np
    
    # Dummy fitness function
    def dummy_fitness(params):
        return -np.sum((params - 0.5)**2)
    
    ga = EnhancedGeneticOptimizer(
        fitness_function=dummy_fitness,
        n_params=5,
        bounds=[(0, 1)] * 5,
        population_size=10,
        console=console,
        enable_logging=True
    )
    
    print("   ✅ Enhanced GA OK\n")
except Exception as e:
    print(f"   ❌ Enhanced GA FAILED: {e}\n")

# Test 6: Enhanced Visualizations
print("6️⃣ Testing Enhanced Visualizations...")
try:
    from enhanced_visualizations import AnimatedVisualizer
    
    viz = AnimatedVisualizer(output_dir=Path('./test_viz'))
    
    # Dummy history
    history = {
        'loss': [1.0, 0.8, 0.6, 0.4],
        'accuracy': [0.5, 0.6, 0.7, 0.8],
        'val_loss': [1.1, 0.9, 0.7, 0.5],
        'val_accuracy': [0.4, 0.5, 0.6, 0.7],
        'defect_loss': [0.5, 0.4, 0.3, 0.2],
        'val_defect_loss': [0.6, 0.5, 0.4, 0.3],
        'learning_rates': [0.001, 0.001, 0.0005, 0.0005]
    }
    
    print("   ✅ Enhanced Visualizations OK\n")
except Exception as e:
    print(f"   ❌ Enhanced Visualizations FAILED: {e}\n")

print("\n✨ Component Testing Complete!")
print("\nTo run the full system: python main.py")
