"""
CyberCore-QC: CNN Defect Detection Module
==========================================
PyTorch-based Convolutional Neural Network for visual defect detection.
Uses ResNet backbone with custom classifier for defect probability estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from tqdm import tqdm


class DefectDataset(Dataset):
    """Custom PyTorch Dataset for defect images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, severity_scores: Optional[List[float]] = None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of class labels
            transform: Torchvision transforms to apply
            severity_scores: Optional severity scores for each image
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.severity_scores = severity_scores if severity_scores else [0.0] * len(labels)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        severity = self.severity_scores[idx]
        
        return image, label, severity, img_path


class DefectCNN(nn.Module):
    """
    Convolutional Neural Network for Defect Detection.
    
    Uses ResNet18 backbone with modified classifier head.
    Outputs both classification logits and defect probability.
    """
    
    def __init__(self, num_classes: int = 6, use_pretrained: bool = True):
        """
        Initialize the CNN.
        
        Args:
            num_classes: Number of defect classes
            use_pretrained: Whether to use pretrained ResNet weights
        """
        super(DefectCNN, self).__init__()
        
        # Load ResNet18 backbone with proper weights parameter
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace classifier with custom head
        self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Defect probability head (binary: defect or no defect)
        self.defect_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            Tuple of (class_logits, defect_probability)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Defect probability (0 = no defect, 1 = defect present)
        defect_prob = self.defect_head(features)
        
        return class_logits, defect_prob
    
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps from different layers
        """
        feature_maps = {}
        
        # Hook into different layers
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(self.backbone.layer1.register_forward_hook(hook_fn('layer1')))
        hooks.append(self.backbone.layer2.register_forward_hook(hook_fn('layer2')))
        hooks.append(self.backbone.layer3.register_forward_hook(hook_fn('layer3')))
        hooks.append(self.backbone.layer4.register_forward_hook(hook_fn('layer4')))
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return feature_maps


class DefectCNNTrainer:
    """Trainer class for DefectCNN."""
    
    def __init__(self, model: DefectCNN, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: DefectCNN model
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.defect_criterion = nn.BCELoss()
        
        # Metrics storage
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_defect_loss': [],
            'val_defect_loss': []
        }
        
        # Best model state storage
        self.best_model_state = None
        
    def train_epoch(self, train_loader: DataLoader, optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        running_class_loss = 0.0
        running_defect_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for images, labels, severities, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Binary defect labels (0 = no defect, 1 = defect)
            defect_labels = (labels > 0).float().unsqueeze(1).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            class_logits, defect_prob = self.model(images)
            
            # Calculate losses
            class_loss = self.class_criterion(class_logits, labels)
            defect_loss = self.defect_criterion(defect_prob, defect_labels)
            
            # Combined loss (weighted)
            loss = class_loss + 0.5 * defect_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_class_loss += class_loss.item()
            running_defect_loss += defect_loss.item()
            
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_defect_loss = running_defect_loss / len(train_loader)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'class_loss': running_class_loss / len(train_loader),
            'defect_loss': epoch_defect_loss
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        running_class_loss = 0.0
        running_defect_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_defect_probs = []
        
        with torch.no_grad():
            for images, labels, severities, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                defect_labels = (labels > 0).float().unsqueeze(1).to(self.device)
                
                # Forward pass
                class_logits, defect_prob = self.model(images)
                
                # Calculate losses
                class_loss = self.class_criterion(class_logits, labels)
                defect_loss = self.defect_criterion(defect_prob, defect_labels)
                loss = class_loss + 0.5 * defect_loss
                
                running_loss += loss.item()
                running_class_loss += class_loss.item()
                running_defect_loss += defect_loss.item()
                
                _, predicted = torch.max(class_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_defect_probs.extend(defect_prob.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        epoch_defect_loss = running_defect_loss / len(val_loader)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'class_loss': running_class_loss / len(val_loader),
            'defect_loss': epoch_defect_loss,
            'predictions': all_preds,
            'labels': all_labels,
            'defect_probs': all_defect_probs
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 20, lr: float = 0.001) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_acc = 0.0
        
        print(f"\n🚀 Starting Training on {self.device.upper()}...")
        print(f"{'='*60}")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_defect_loss'].append(train_metrics['defect_loss'])
            self.history['val_defect_loss'].append(val_metrics['defect_loss'])
            
            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])
            
            # Save best model state in memory
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                # Keep best model state in memory
                self.best_model_state = {
                    'model_state_dict': self.model.state_dict().copy(),
                    'val_acc': best_val_acc
                }
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            print(f"{'='*60}")
        
        print(f"\n✅ Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint (uses best model if available)."""
        # Use best model state if available, otherwise current state
        if hasattr(self, 'best_model_state') and self.best_model_state:
            torch.save({
                'model_state_dict': self.best_model_state['model_state_dict'],
                'history': self.history
            }, filepath)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history
            }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


def get_data_transforms(img_size: int = 224):
    """
    Get data augmentation transforms.
    
    Args:
        img_size: Target image size
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms


if __name__ == "__main__":
    # Test CNN architecture
    model = DefectCNN(num_classes=6)
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    class_out, defect_out = model(dummy_input)
    print(f"\nClass Output Shape: {class_out.shape}")
    print(f"Defect Probability Shape: {defect_out.shape}")
