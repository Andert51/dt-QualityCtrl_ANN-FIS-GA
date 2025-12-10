# 🚀 CyberCore-QC Quick Start Guide

## Prerequisites Check

Before running, ensure you have:
- ✅ Python 3.8+ installed
- ✅ pip package manager
- ✅ (Optional) NVIDIA GPU with CUDA for faster training

## Installation Steps

### 1. Navigate to Project Directory

```powershell
cd C:\Users\andre\Core\Eye_ofthe_Universe\Universs\Soft_Computing\Projects\PF_CtrlCalidad_ANNFISGA
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This will install:
- PyTorch (with CPU/CUDA support)
- scikit-fuzzy
- Rich, Questionary (for TUI)
- Matplotlib, Seaborn, Plotly (for visualizations)
- And more...

Installation may take 5-10 minutes depending on your internet connection.

---

## Running the System

### Launch Main Application

```powershell
python main.py
```

You should see the cyberpunk-themed ASCII banner:

```
   ______  ______  ____  __________  ______  ____  ______       ____  ______
  / ____/ / ____/ / __ )/ ____/ __ \/ ____/ / __ \/ ____/      / __ \/ ____/
 / /     / /     / __  / __/ / /_/ / /     / / / / /     _____/ / / / /     
/ /___  / /___  / /_/ / /___/ _, _/ /___  / /_/ / /___  /____/ /_/ / /___   
\____/  \____/  /_____/_____/_/ |_|\____/  \____/\____/       \___\_\____/   
```

---

## First-Time Workflow

### Step 1: Initialize System

1. Select: `🔧 Initialize System / Load Data`
2. Wait for synthetic dataset generation (~30 seconds)
3. System creates 600 industrial defect images

**Expected Output:**
```
🏭 Generating Synthetic Industrial Defect Dataset...
  ⚙️  Generating Normal samples...
  ⚙️  Generating Scratches samples...
  ⚙️  Generating Inclusion samples...
  ⚙️  Generating Patches samples...
  ⚙️  Generating Pitted_Surface samples...
  ⚙️  Generating Rolled_Scale samples...
✅ Generated 600 synthetic images!
```

### Step 2: Train CNN

1. Select: `🧠 Train CNN Model`
2. Enter epochs (recommended: **15**)
3. Watch training progress with live metrics
4. Training takes ~5-10 minutes (GPU) or 15-30 minutes (CPU)

**Expected Accuracy:** 85-95% on validation set

### Step 3: Run Genetic Optimization

1. Select: `🧬 Run Genetic Optimization`
2. GA evolves fuzzy membership functions (50 generations)
3. Takes ~10-15 minutes
4. Improves FIS decision accuracy by 10-15%

**Expected Fitness Improvement:** 0.70 → 0.85+

### Step 4: Visual Analysis

1. Select: `📊 Visual Analysis Hub`
2. Choose visualization type or "All Visualizations"
3. Check `output/results/visualizations/` folder

**Generated Files:**
- `training_curves.png` - CNN learning curves
- `confusion_matrix.png` - Classification heatmap
- `network_topology.png` - CNN architecture graph
- `activation_heatmap.png` - Feature map visualizations
- `optimized_membership_functions.png` - Fuzzy sets
- `fuzzy_surface_3d.html` - Interactive 3D plot (open in browser!)
- `ga_evolution.png` - Genetic algorithm progress
- `results_grid.png` - Sample predictions
- `dataset_overview.png` - Dataset statistics

### Step 5: Save Models

1. Select: `💾 Save/Load Models`
2. Choose "Save Models"
3. Models saved to `output/models/`

---

## Testing Individual Components

### Test Synthetic Data Generator

```powershell
python src/data_generator.py
```

### Test CNN Architecture

```powershell
python src/cnn_model.py
```

### Test Fuzzy System

```powershell
python src/fuzzy_system.py
```

**Expected Output:**
```
Fuzzy Inference System Test Cases:
======================================================================

Low defect, low fragility
  Defect Prob: 0.10, Material Fragility: 0.20
  → Severity: 1.25
  → Decision: Accept (green)

Medium defect, medium fragility
  Defect Prob: 0.50, Material Fragility: 0.50
  → Severity: 5.00
  → Decision: Rework (yellow)

High defect, high fragility
  Defect Prob: 0.90, Material Fragility: 0.80
  → Severity: 9.50
  → Decision: Reject (red)
```

### Test Genetic Algorithm

```powershell
python src/genetic_algorithm.py
```

---

## Troubleshooting

### Issue 1: "No module named 'torch'"

**Solution:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "scikit-fuzzy not found"

**Solution:**
```powershell
pip install scikit-fuzzy
```

### Issue 3: PowerShell execution policy error

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 4: CUDA not available (training is slow)

**Check CUDA:**
```python
import torch
print(torch.cuda.is_available())  # Should be True if CUDA works
```

**Solution:** System will automatically use CPU if CUDA unavailable. Training works but slower.

### Issue 5: Out of memory error

**Solution:** Edit `main.py` and reduce batch size:
```python
# Line ~220 in main.py
self.train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8,  # Changed from 16
    shuffle=True, num_workers=0
)
```

---

## Performance Expectations

### Training Time (15 epochs)

- **GPU (NVIDIA RTX 3060+):** 5-8 minutes
- **CPU (Intel i7 8th gen+):** 15-30 minutes

### Genetic Algorithm (50 generations)

- **Any hardware:** 10-15 minutes

### Total First Run Time

- **GPU:** ~20-25 minutes
- **CPU:** ~30-45 minutes

---

## Next Steps

After successful setup:

1. **Experiment with parameters:**
   - Try different epoch counts
   - Adjust GA population size
   - Modify fuzzy membership functions

2. **Integrate real data:**
   - Replace synthetic images with real defect photos
   - Update dataset loader in `data_generator.py`

3. **Extend functionality:**
   - Add more defect classes
   - Implement ensemble models
   - Create web dashboard

---

## Getting Help

**Check logs in console output**

**Verify file structure:**
```powershell
tree /F
```

**Expected structure:**
```
├── main.py
├── requirements.txt
├── src/
│   ├── data_generator.py
│   ├── cnn_model.py
│   ├── fuzzy_system.py
│   ├── genetic_algorithm.py
│   └── visualizations.py
├── input/
├── output/
└── README.md
```

---

## Success Indicators

✅ **System initialized successfully**
- Message: "System Initialized Successfully!"
- Dataset folder created with 600 images

✅ **CNN trained successfully**
- Message: "Training Complete! Best Validation Accuracy: XX%"
- Accuracy > 80%

✅ **GA optimized successfully**
- Message: "Genetic optimization complete! Best fitness: 0.XX"
- Fitness > 0.75

✅ **Visualizations created**
- Message: "Saved X to visualizations/"
- Files appear in output/results/visualizations/

---

## Quick Commands Reference

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run
python main.py

# Test components
python src/data_generator.py
python src/cnn_model.py
python src/fuzzy_system.py
python src/genetic_algorithm.py

# Deactivate venv
deactivate
```

---

**Ready to go! Launch `python main.py` and select option 1 to begin! 🚀**
