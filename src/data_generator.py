"""
CyberCore-QC: Synthetic Industrial Defect Dataset Generator
============================================================
Generates realistic industrial surface defect images with various defect types.
Fallback system for when external datasets are unavailable.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path
from typing import Tuple, List
import json


class SyntheticDefectGenerator:
    """
    Generates synthetic industrial surface images with realistic defects.
    
    Defect Types:
    - 0: Normal (No defect)
    - 1: Scratches (Linear defects)
    - 2: Inclusion (Foreign particles/spots)
    - 3: Patches (Surface irregularities)
    - 4: Pitted Surface (Small holes/pitting)
    - 5: Rolled-in Scale (Oxide scale defects)
    """
    
    DEFECT_CLASSES = {
        0: "Normal",
        1: "Scratches",
        2: "Inclusion",
        3: "Patches",
        4: "Pitted_Surface",
        5: "Rolled_Scale"
    }
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the synthetic data generator.
        
        Args:
            img_size: Tuple of (width, height) for generated images
        """
        self.img_size = img_size
        self.rng = np.random.default_rng(seed=42)
        
    def _create_base_surface(self) -> Image.Image:
        """Create a base industrial surface texture."""
        # Create realistic metal/industrial surface with noise
        base_color = self.rng.integers(100, 180)
        noise = self.rng.normal(0, 15, (self.img_size[1], self.img_size[0]))
        surface = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        # Add subtle grain pattern
        grain = self.rng.normal(0, 5, (self.img_size[1], self.img_size[0]))
        surface = np.clip(surface + grain, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(surface, mode='L').convert('RGB')
        
        # Add slight texture blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def _add_scratches(self, img: Image.Image, severity: float) -> Image.Image:
        """Add linear scratch defects."""
        draw = ImageDraw.Draw(img)
        num_scratches = int(severity * 10) + 1
        
        for _ in range(num_scratches):
            # Random scratch parameters
            x1 = self.rng.integers(0, self.img_size[0])
            y1 = self.rng.integers(0, self.img_size[1])
            
            # Scratch length and angle
            length = self.rng.integers(30, 150)
            angle = self.rng.uniform(0, 2 * np.pi)
            
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Draw scratch with varying darkness
            darkness = self.rng.integers(20, 80)
            width = int(severity * 3) + 1
            draw.line([(x1, y1), (x2, y2)], fill=(darkness, darkness, darkness), width=width)
        
        return img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    def _add_inclusion(self, img: Image.Image, severity: float) -> Image.Image:
        """Add foreign particle/inclusion defects."""
        draw = ImageDraw.Draw(img)
        num_inclusions = int(severity * 8) + 1
        
        for _ in range(num_inclusions):
            x = self.rng.integers(10, self.img_size[0] - 10)
            y = self.rng.integers(10, self.img_size[1] - 10)
            radius = int(severity * 15) + 3
            
            # Dark or light inclusions
            if self.rng.random() > 0.5:
                color_val = self.rng.integers(30, 70)
            else:
                color_val = self.rng.integers(180, 230)
            
            color = (color_val, color_val, color_val)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    
    def _add_patches(self, img: Image.Image, severity: float) -> Image.Image:
        """Add patch/irregular surface defects."""
        img_array = np.array(img)
        num_patches = int(severity * 5) + 1
        
        for _ in range(num_patches):
            center_x = self.rng.integers(30, self.img_size[0] - 30)
            center_y = self.rng.integers(30, self.img_size[1] - 30)
            patch_size = int(severity * 40) + 20
            
            # Create irregular patch using gaussian
            y, x = np.ogrid[-patch_size:patch_size, -patch_size:patch_size]
            mask = x*x + y*y <= patch_size*patch_size
            
            # Apply patch with color variation
            color_offset = self.rng.integers(-50, 50)
            for i in range(3):
                patch_region = img_array[
                    max(0, center_y - patch_size):min(self.img_size[1], center_y + patch_size),
                    max(0, center_x - patch_size):min(self.img_size[0], center_x + patch_size),
                    i
                ]
                
                mask_region = mask[:patch_region.shape[0], :patch_region.shape[1]]
                patch_region[mask_region] = np.clip(
                    patch_region[mask_region] + color_offset, 0, 255
                )
        
        return Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=2.0))
    
    def _add_pitting(self, img: Image.Image, severity: float) -> Image.Image:
        """Add pitted surface defects (small holes)."""
        draw = ImageDraw.Draw(img)
        num_pits = int(severity * 20) + 5
        
        for _ in range(num_pits):
            x = self.rng.integers(5, self.img_size[0] - 5)
            y = self.rng.integers(5, self.img_size[1] - 5)
            pit_radius = int(severity * 5) + 1
            
            # Dark pits
            darkness = self.rng.integers(10, 50)
            draw.ellipse(
                [x - pit_radius, y - pit_radius, x + pit_radius, y + pit_radius],
                fill=(darkness, darkness, darkness)
            )
        
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def _add_rolled_scale(self, img: Image.Image, severity: float) -> Image.Image:
        """Add rolled-in scale defects (oxide patterns)."""
        img_array = np.array(img)
        num_scales = int(severity * 6) + 1
        
        for _ in range(num_scales):
            x = self.rng.integers(20, self.img_size[0] - 20)
            y = self.rng.integers(20, self.img_size[1] - 20)
            
            # Create scale pattern
            scale_width = int(severity * 60) + 20
            scale_height = int(severity * 15) + 5
            
            # Add dark irregular pattern
            for dx in range(-scale_width // 2, scale_width // 2):
                for dy in range(-scale_height // 2, scale_height // 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < self.img_size[0] and 0 <= py < self.img_size[1]:
                        if self.rng.random() > 0.3:  # Irregular pattern
                            darkening = self.rng.integers(30, 70)
                            img_array[py, px] = np.clip(img_array[py, px] - darkening, 0, 255)
        
        return Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=1.5))
    
    def generate_image(self, defect_class: int, severity: float = None) -> Tuple[Image.Image, float]:
        """
        Generate a single defect image.
        
        Args:
            defect_class: Defect type (0-5)
            severity: Defect severity (0.0-1.0), auto-generated if None
            
        Returns:
            Tuple of (PIL Image, actual severity used)
        """
        if severity is None:
            severity = self.rng.uniform(0.3, 1.0) if defect_class > 0 else 0.0
        
        # Create base surface
        img = self._create_base_surface()
        
        # Add defect based on class
        if defect_class == 1:
            img = self._add_scratches(img, severity)
        elif defect_class == 2:
            img = self._add_inclusion(img, severity)
        elif defect_class == 3:
            img = self._add_patches(img, severity)
        elif defect_class == 4:
            img = self._add_pitting(img, severity)
        elif defect_class == 5:
            img = self._add_rolled_scale(img, severity)
        # defect_class == 0 is normal, no defect added
        
        return img, severity
    
    def generate_dataset(
        self,
        output_dir: Path,
        samples_per_class: int = 200,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> dict:
        """
        Generate a complete synthetic dataset.
        
        Args:
            output_dir: Directory to save dataset
            samples_per_class: Number of samples per defect class
            split_ratios: (train, val, test) split ratios
            
        Returns:
            Dictionary with dataset statistics and paths
        """
        output_dir = Path(output_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for class_id, class_name in self.DEFECT_CLASSES.items():
                (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'classes': self.DEFECT_CLASSES,
            'samples': {},
            'image_size': self.img_size,
            'total_samples': 0
        }
        
        train_ratio, val_ratio, test_ratio = split_ratios
        
        print("🏭 Generating Synthetic Industrial Defect Dataset...")
        
        for class_id, class_name in self.DEFECT_CLASSES.items():
            print(f"  ⚙️  Generating {class_name} samples...")
            
            class_metadata = []
            
            for i in range(samples_per_class):
                # Determine split
                rand_val = self.rng.random()
                if rand_val < train_ratio:
                    split = 'train'
                elif rand_val < train_ratio + val_ratio:
                    split = 'val'
                else:
                    split = 'test'
                
                # Generate image
                img, severity = self.generate_image(class_id)
                
                # Save image
                filename = f"{class_name}_{i:04d}.png"
                filepath = output_dir / split / class_name / filename
                img.save(filepath)
                
                # Record metadata
                class_metadata.append({
                    'filename': filename,
                    'split': split,
                    'class_id': class_id,
                    'class_name': class_name,
                    'severity': float(severity),
                    'path': str(filepath)
                })
            
            metadata['samples'][class_name] = class_metadata
            metadata['total_samples'] += samples_per_class
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated {metadata['total_samples']} synthetic images!")
        print(f"📁 Dataset saved to: {output_dir}")
        
        return metadata


if __name__ == "__main__":
    # Test synthetic data generation
    generator = SyntheticDefectGenerator()
    
    # Generate small test dataset
    test_dir = Path("../output/synthetic_dataset")
    metadata = generator.generate_dataset(test_dir, samples_per_class=50)
    
    print(f"\nDataset Statistics:")
    print(f"Total Samples: {metadata['total_samples']}")
    print(f"Classes: {len(metadata['classes'])}")
