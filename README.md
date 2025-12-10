# CyberCore-QC: Hybrid Intelligent Quality Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

**A Cyberpunk-Themed Industrial AI System**  
*Combining CNN, Fuzzy Logic, and Genetic Algorithms for Quality Control*

</div>

---

## Overview

**CyberCore-QC** is a cutting-edge hybrid intelligent system designed for industrial quality control. It chains three powerful AI mechanisms:

1. **🧠 CNN (Convolutional Neural Network)**: Visual defect detection using ResNet18 backbone
2. **🎛️ FIS (Fuzzy Inference System)**: Decision-making based on defect probability and material properties
3. **🧬 GA (Genetic Algorithm)**: Automated optimization of fuzzy membership functions

### Key Features

- **100% Self-Contained**: Includes synthetic data generator - runs without external datasets
- **Cyberpunk TUI**: Rich terminal interface with neon aesthetics
- **Advanced Visualizations**: 30+ high-quality plots using Matplotlib, Seaborn, and Plotly
- **End-to-End Pipeline**: From raw images to optimized decisions
- **Model Persistence**: Save/load trained models and configurations
- **Production Ready**: Robust error handling and logging

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Images                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │   CNN (ResNet18)     │
         │  Feature Extraction  │
         │  Defect Probability  │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────────────┐
         │  Fuzzy Inference System      │
         │  • Defect Prob (Input)       │
         │  • Material Fragility (Input)│
         │  • Severity Score (Output)   │
         └───────────┬──────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Decision Output     │
         │  • Accept (Green)    │
         │  • Rework (Yellow)   │
         │  • Reject (Red)      │
         └──────────────────────┘
                     
         ┌──────────────────────┐
         │  Genetic Algorithm   │
         │  Optimizes FIS MFs   │
         │  (Background Process)│
         └──────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Quick Start

1. **Clone or navigate to the project directory**

```bash
cd PF_CtrlCalidad_ANNFISGA
```

2. **Create and activate virtual environment**

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```powershell
pip install -r requirements.txt
```

4. **Run the system**

```powershell
python main.py
```

---

## Usage

### Interactive Menu

The system provides a cyberpunk-themed interactive terminal interface:

```
╔══════════════════════════════════════════════╗
║           CYBERCORE-QC MAIN MENU             ║
╚══════════════════════════════════════════════╝

1. Initialize System / Load Data
2. Train CNN Model
3. Run Genetic Optimization
4. Visual Analysis Hub
5. Save/Load Models
6. Exit
```

### Typical Workflow

1. **Initialize System**: Loads or generates synthetic dataset (600 images by default)
2. **Train CNN**: Trains the defect detection model (15-20 epochs recommended)
3. **Run GA Optimization**: Evolves fuzzy membership functions (50+ generations)
4. **Visual Analysis**: Generate comprehensive visualizations
5. **Save Models**: Persist trained models for future use

---

## Visualizations

CyberCore-QC generates **15+ types of visualizations**:

### CNN Visualizations
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrix**: Classification performance heatmap
- **Network Topology**: Graph visualization of CNN architecture
- **Activation Heatmaps**: Layer-wise feature activations (CAM-style)
- **Weight Visualizations**: First-layer convolutional filters

### FIS Visualizations
- **Membership Functions**: Fuzzy set triangular/trapezoidal plots
- **3D Decision Surface**: Interactive Plotly surface (defect prob × fragility → severity)
- **Decision Boundaries**: 2D contour maps

### GA Visualizations
- **Evolution Progress**: Fitness improvement over generations
- **Population Diversity**: Genetic diversity tracking
- **Best Solution**: Optimized membership function comparison

### System-Wide
- **Dataset Overview**: Class distribution and split ratios
- **Results Grid**: 4×4 sample predictions with color-coded decisions
- **Performance Metrics**: Accuracy, precision, recall tables

All visualizations are saved to `output/results/visualizations/`.

---

## Technical Details

### CNN Architecture

```python
Input: 224×224×3 RGB images
├─ ResNet18 Backbone (Pretrained on ImageNet)
│  ├─ Conv Layers (64 → 128 → 256 → 512 filters)
│  └─ Global Average Pooling
├─ Classification Head
│  ├─ FC(512 → 256) + ReLU + Dropout(0.3)
│  ├─ FC(256 → 128) + ReLU + Dropout(0.2)
│  └─ FC(128 → 6) [6 defect classes]
└─ Defect Probability Head
   ├─ FC(512 → 64) + ReLU + Dropout(0.2)
   └─ FC(64 → 1) + Sigmoid [Binary: defect/no-defect]
```

### Fuzzy Inference System

**Inputs:**
- **Defect Probability** (0.0 - 1.0): From CNN
  - Fuzzy Sets: Low, Medium, High
- **Material Fragility** (0.0 - 1.0): Simulated sensor data
  - Fuzzy Sets: Low, Medium, High

**Output:**
- **Severity Score** (0.0 - 10.0): Quality decision metric
  - Fuzzy Sets: Accept (0-3), Rework (3-7), Reject (7-10)

**Rules:** 9 fuzzy rules (3×3 combinations)

### Genetic Algorithm

- **Chromosome**: 18 real values (6 membership functions × 3 parameters [a, b, c])
- **Population Size**: 40-50 individuals
- **Selection**: Tournament selection (size=3)
- **Crossover**: Single-point + blending (rate=0.8)
- **Mutation**: Gaussian noise (rate=0.2)
- **Fitness Function**: Classification accuracy on validation set

---

## Project Structure

```
PF_CtrlCalidad_ANNFISGA/
├── main.py                      # Main orchestrator with TUI
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── data_generator.py        # Synthetic defect image generator
│   ├── cnn_model.py            # CNN architecture and trainer
│   ├── fuzzy_system.py         # Fuzzy inference system
│   ├── genetic_algorithm.py    # GA optimizer
│   └── visualizations.py       # Visualization hub
│
├── input/
│   └── dataset/                # Auto-generated synthetic images
│       ├── train/
│       ├── val/
│       ├── test/
│       └── metadata.json
│
├── output/
│   ├── models/                 # Saved model checkpoints
│   │   ├── best_cnn_model.pth
│   │   └── optimized_fis.pkl
│   └── results/
│       └── visualizations/     # All generated plots
│
├── config/                     # Configuration files (optional)
├── docs/                       # Additional documentation
└── test/                       # Unit tests (future)
```

---

## Synthetic Dataset

The system includes a **robust synthetic data generator** that creates realistic industrial surface defect images:

### Defect Classes (6 Types)

1. **Normal**: Clean surface (no defects)
2. **Scratches**: Linear surface damage
3. **Inclusion**: Foreign particles/spots
4. **Patches**: Irregular surface areas
5. **Pitted Surface**: Small holes/pitting
6. **Rolled Scale**: Oxide scale patterns

### Dataset Specs

- **Default Size**: 600 images (100 per class)
- **Image Size**: 224×224 pixels
- **Split Ratio**: 70% train / 15% val / 15% test
- **Format**: PNG with metadata JSON

---

## Configuration

### CNN Training Parameters

```python
epochs = 15-20          # Training epochs
batch_size = 16         # Batch size
learning_rate = 0.001   # Adam optimizer LR
```

### GA Parameters

```python
population_size = 40-50
generations = 50-100
crossover_rate = 0.8
mutation_rate = 0.2
elite_size = 5
```

---

## Performance

### Expected Metrics (on synthetic data)

- **CNN Accuracy**: 85-95% (validation set)
- **FIS Accuracy**: 70-80% (before GA)
- **Optimized FIS**: 80-90% (after GA)
- **Training Time**: ~5-10 min (15 epochs, GPU)
- **GA Optimization**: ~10-15 min (50 generations)

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in main.py
batch_size = 8  # Instead of 16
```

**2. Slow Training (No GPU)**
```python
# System auto-detects CPU/GPU
# Training will work but slower on CPU
```

**3. Visualization Window Not Showing**
```python
# Plots are saved to disk automatically
# Check: output/results/visualizations/
```

---

## Dependencies

### Core Libraries

- `torch` (2.0+): Deep learning framework
- `scikit-fuzzy` (0.4.2+): Fuzzy logic system
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`, `plotly`: Visualizations
- `rich`, `pyfiglet`: Terminal UI
- `questionary`: Interactive menus

See `requirements.txt` for complete list.

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Real dataset integration (NEU-DET, etc.)
- [ ] Additional defect types
- [ ] Advanced CNN architectures (EfficientNet, Vision Transformers)
- [ ] Multi-objective GA optimization
- [ ] Web dashboard (FastAPI + React)
- [ ] Model interpretability (SHAP, LIME)

---

## License

MIT License - feel free to use in academic or commercial projects.

---

## 👨‍💻 Author

**QualityCtrl AI Lab**  
*Andres Torres Ceja*

---

## 🌟 Acknowledgments

- PyTorch team for the excellent deep learning framework
- scikit-fuzzy maintainers for fuzzy logic implementation
- Rich library for beautiful terminal interfaces

---

<div align="center">

*Built with 💜 for the future of industrial AI*

</div>
