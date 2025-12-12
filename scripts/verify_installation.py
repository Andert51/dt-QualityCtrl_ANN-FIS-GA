"""
CyberCore-QC: Installation Verification Script
===============================================
Verifies that all dependencies are installed correctly.
Run this after 'pip install -r requirements.txt'
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    
    print("=" * 70)
    print("CyberCore-QC Installation Verification")
    print("=" * 70)
    print()
    
    packages = {
        'Core ML': [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('sklearn', 'scikit-learn'),
            ('skfuzzy', 'scikit-fuzzy'),
        ],
        'Data & Computation': [
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('scipy', 'SciPy'),
        ],
        'Visualization': [
            ('matplotlib', 'Matplotlib'),
            ('seaborn', 'Seaborn'),
            ('plotly', 'Plotly'),
            ('networkx', 'NetworkX'),
        ],
        'Image Processing': [
            ('PIL', 'Pillow'),
            ('imageio', 'ImageIO'),
        ],
        'Terminal UI': [
            ('rich', 'Rich'),
            ('pyfiglet', 'PyFiglet'),
            ('questionary', 'Questionary'),
            ('colorama', 'Colorama'),
            ('tqdm', 'TQDM'),
        ],
        'Utilities': [
            ('yaml', 'PyYAML'),
            ('joblib', 'Joblib'),
        ]
    }
    
    all_ok = True
    
    for category, package_list in packages.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for module_name, display_name in package_list:
            try:
                __import__(module_name)
                version = ""
                
                # Try to get version
                try:
                    mod = sys.modules[module_name]
                    if hasattr(mod, '__version__'):
                        version = f" (v{mod.__version__})"
                except:
                    pass
                
                print(f"  ✅ {display_name:20s} {version}")
            except ImportError as e:
                print(f"  ❌ {display_name:20s} - NOT FOUND")
                all_ok = False
    
    print("\n" + "=" * 70)
    
    # Additional checks
    print("\nAdditional Checks:")
    print("-" * 70)
    
    # CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA not available (CPU mode only)")
    except:
        pass
    
    # File structure
    workspace = Path(__file__).parent
    required_dirs = ['src', 'input', 'output', 'config', 'docs']
    
    print("\nDirectory Structure:")
    print("-" * 70)
    for dir_name in required_dirs:
        dir_path = workspace / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - MISSING")
            all_ok = False
    
    # Required files
    required_files = ['main.py', 'demo.py', 'requirements.txt', 'README.md']
    
    print("\nRequired Files:")
    print("-" * 70)
    for file_name in required_files:
        file_path = workspace / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - MISSING")
            all_ok = False
    
    # Source modules
    src_modules = [
        'data_generator.py',
        'cnn_model.py',
        'fuzzy_system.py',
        'genetic_algorithm.py',
        'visualizations.py',
        '__init__.py'
    ]
    
    print("\nSource Modules:")
    print("-" * 70)
    for module_name in src_modules:
        module_path = workspace / 'src' / module_name
        if module_path.exists():
            print(f"  ✅ src/{module_name}")
        else:
            print(f"  ❌ src/{module_name} - MISSING")
            all_ok = False
    
    print("\n" + "=" * 70)
    
    if all_ok:
        print("\n✅ SUCCESS: All dependencies installed correctly!")
        print("\nNext Steps:")
        print("  1. Run 'python demo.py' for a quick test (2-5 min)")
        print("  2. Run 'python main.py' for the full interactive system")
        print("  3. See docs/QUICKSTART.md for detailed instructions")
    else:
        print("\n❌ INSTALLATION INCOMPLETE")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 70)
    print()
    
    return all_ok


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
