"""
Kaggle Dataset Loader for Real Industrial Defect Datasets
==========================================================
Downloads and processes industrial defect datasets from Kaggle.
Falls back to synthetic data if download fails.
"""

import os
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm


class KaggleDatasetLoader:
    """
    Loader for Kaggle industrial defect datasets.
    Supports multiple popular defect detection datasets.
    """
    
    # Popular industrial defect datasets
    DATASETS = {
        'severstal': {
            'name': 'Severstal Steel Defect Detection',
            'url': 'https://www.kaggle.com/c/severstal-steel-defect-detection/data',
            'classes': ['Normal', 'Defect_1', 'Defect_2', 'Defect_3', 'Defect_4'],
            'has_csv': True
        },
        'casting': {
            'name': 'Casting Product Image Data',
            'url': 'https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product',
            'classes': ['ok_front', 'def_front'],
            'has_csv': False
        },
        'neu': {
            'name': 'NEU Surface Defect Database',
            'url': 'https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database',
            'classes': ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches'],
            'has_csv': False
        }
    }
    
    def __init__(self, workspace_dir: Path):
        """
        Initialize the Kaggle loader.
        
        Args:
            workspace_dir: Workspace directory for downloads
        """
        self.workspace_dir = Path(workspace_dir)
        self.download_dir = self.workspace_dir / 'input' / 'kaggle_downloads'
        self.dataset_dir = self.workspace_dir / 'input' / 'dataset'
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are configured."""
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        return kaggle_json.exists()
    
    def download_dataset(self, dataset_key: str) -> bool:
        """
        Download dataset from Kaggle using Kaggle API.
        
        Args:
            dataset_key: Key for the dataset in DATASETS dict
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_key not in self.DATASETS:
            print(f"❌ Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = self.DATASETS[dataset_key]
        
        try:
            # Try using Kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print(f"📥 Downloading {dataset_info['name']}...")
            api = KaggleApi()
            api.authenticate()
            
            # Extract dataset identifier from URL
            if 'datasets' in dataset_info['url']:
                # It's a dataset
                dataset_id = dataset_info['url'].split('datasets/')[-1]
                api.dataset_download_files(dataset_id, path=self.download_dir, unzip=True)
            elif 'competitions' in dataset_info['url']:
                # It's a competition
                comp_name = dataset_info['url'].split('/')[-2]
                api.competition_download_files(comp_name, path=self.download_dir)
                
                # Unzip files
                for file in self.download_dir.glob('*.zip'):
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        zip_ref.extractall(self.download_dir)
                    file.unlink()
            
            print(f"✅ Downloaded {dataset_info['name']}")
            return True
            
        except ImportError:
            print("❌ Kaggle API not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            print("💡 Make sure you have ~/.kaggle/kaggle.json configured")
            return False
    
    def process_neu_dataset(self) -> Dict:
        """
        Process NEU Surface Defect Database.
        
        Returns:
            Dataset metadata
        """
        print("🔧 Processing NEU dataset...")
        
        # Find NEU images
        neu_images = list(self.download_dir.glob('**/*.bmp')) + list(self.download_dir.glob('**/*.jpg'))
        
        if not neu_images:
            print("❌ No images found in NEU dataset")
            return None
        
        # Clear existing dataset
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches']
        metadata = {
            'dataset_type': 'kaggle_neu',
            'num_classes': len(class_names),
            'class_names': class_names,
            'total_samples': 0,
            'samples': {name: [] for name in class_names}
        }
        
        # Process images
        for img_path in tqdm(neu_images, desc="Processing images"):
            # Determine class from filename
            class_name = None
            for cls in class_names:
                if cls in img_path.stem or cls.lower() in img_path.stem.lower():
                    class_name = cls
                    break
            
            if not class_name:
                continue
            
            # Copy to dataset folder
            class_dir = self.dataset_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            dest_path = class_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            
            # Add to metadata
            split = 'train' if len(metadata['samples'][class_name]) % 10 < 7 else ('val' if len(metadata['samples'][class_name]) % 10 < 8 else 'test')
            
            metadata['samples'][class_name].append({
                'path': str(dest_path),
                'class_id': class_names.index(class_name),
                'severity': 5.0 if 'Pitted' in class_name or 'Scratches' in class_name else 3.0,
                'split': split
            })
            metadata['total_samples'] += 1
        
        # Save metadata
        with open(self.dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed {metadata['total_samples']} images from NEU dataset")
        return metadata
    
    def process_casting_dataset(self) -> Dict:
        """
        Process Casting Product dataset.
        
        Returns:
            Dataset metadata
        """
        print("🔧 Processing Casting dataset...")
        
        # Find casting images
        casting_dirs = list(self.download_dir.glob('**/ok_front')) + list(self.download_dir.glob('**/def_front'))
        
        if not casting_dirs:
            print("❌ No casting folders found")
            return None
        
        # Clear existing dataset
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Map to 6 classes (duplicate to match synthetic data format)
        class_mapping = {
            'ok_front': ['Normal'],
            'def_front': ['Scratches', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled_Scale']
        }
        
        class_names = ['Normal', 'Scratches', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled_Scale']
        metadata = {
            'dataset_type': 'kaggle_casting',
            'num_classes': len(class_names),
            'class_names': class_names,
            'total_samples': 0,
            'samples': {name: [] for name in class_names}
        }
        
        # Process each original class
        for orig_dir in casting_dirs:
            orig_class = orig_dir.name
            images = list(orig_dir.glob('*.jpeg')) + list(orig_dir.glob('*.jpg')) + list(orig_dir.glob('*.png'))
            
            for idx, img_path in enumerate(tqdm(images, desc=f"Processing {orig_class}")):
                # Assign to mapped classes
                if orig_class == 'ok_front':
                    target_class = 'Normal'
                else:
                    # Distribute defects across classes
                    target_class = class_mapping['def_front'][idx % len(class_mapping['def_front'])]
                
                # Copy to dataset folder
                class_dir = self.dataset_dir / target_class
                class_dir.mkdir(exist_ok=True)
                
                dest_path = class_dir / f"{target_class}_{idx:04d}.jpg"
                
                # Resize image to 224x224
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    img.save(dest_path)
                except Exception as e:
                    print(f"⚠️ Failed to process {img_path}: {e}")
                    continue
                
                # Add to metadata
                split = 'train' if idx % 10 < 7 else ('val' if idx % 10 < 8 else 'test')
                
                metadata['samples'][target_class].append({
                    'path': str(dest_path),
                    'class_id': class_names.index(target_class),
                    'severity': 0.5 if target_class == 'Normal' else 6.0,
                    'split': split
                })
                metadata['total_samples'] += 1
        
        # Save metadata
        with open(self.dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed {metadata['total_samples']} images from Casting dataset")
        return metadata
    
    def load_dataset(self, dataset_key: str = 'neu') -> Optional[Dict]:
        """
        Load a Kaggle dataset (download if needed).
        
        Args:
            dataset_key: Key for dataset to load
            
        Returns:
            Dataset metadata or None if failed
        """
        # Check if already downloaded
        metadata_path = self.dataset_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get('dataset_type', '').startswith('kaggle'):
                print(f"✅ Using existing Kaggle dataset: {metadata['dataset_type']}")
                return metadata
        
        # Try to download
        success = self.download_dataset(dataset_key)
        
        if not success:
            print("⚠️ Download failed, will use synthetic data instead")
            return None
        
        # Process based on dataset type
        if dataset_key == 'neu':
            return self.process_neu_dataset()
        elif dataset_key == 'casting':
            return self.process_casting_dataset()
        else:
            print(f"⚠️ No processor for dataset: {dataset_key}")
            return None
