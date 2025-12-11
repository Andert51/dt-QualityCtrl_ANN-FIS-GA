"""
Enhanced CNN Trainer with Advanced UI, Validation, and Logging
Professional training loop with cyberpunk aesthetics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from rich.layout import Layout

from ui_components import CyberpunkUI, TrainingProgressDisplay, create_advanced_progress
from validation import DataValidator, ValidationResult
from logger import get_logger
from gpu_optimizer import GPUOptimizer


class EnhancedCNNTrainer:
    """
    Enhanced CNN trainer with professional UI and robust error handling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        console: Optional[Console] = None,
        enable_validation: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cuda' or 'cpu')
            console: Rich console for output
            enable_validation: Enable data validation
            enable_logging: Enable logging
        """
        # GPU Optimization
        self.gpu_optimizer = GPUOptimizer()
        self.device = self.gpu_optimizer.get_optimal_device(prefer_gpu=True if device == 'cuda' else False)
        
        # Optimize model for GPU
        self.model = self.gpu_optimizer.optimize_model_for_gpu(model, self.device)
        
        # Mixed precision training
        self.scaler, self.use_amp = self.gpu_optimizer.enable_mixed_precision() if self.device == 'cuda' else (None, False)
        
        self.console = console or Console()
        self.ui = CyberpunkUI(self.console)
        self.train_display = TrainingProgressDisplay(self.console)
        
        self.enable_validation = enable_validation
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            self.logger = get_logger('CNNTrainer')
            if self.device == 'cuda':
                self.logger.info(f"GPU Training enabled with mixed precision: {self.use_amp}")
        
        # Loss functions
        # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability and mixed precision support
        self.class_criterion = nn.CrossEntropyLoss()
        self.defect_criterion = nn.BCEWithLogitsLoss()
        
        # Metrics storage
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_defect_loss': [],
            'val_defect_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model state
        self.best_model_state = None
        self.best_val_acc = 0.0
        
        # Callbacks
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add training callback."""
        self.callbacks.append(callback)
    
    def _validate_batch(self, images: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """Validate batch data if enabled."""
        if not self.enable_validation:
            return
        
        result = DataValidator.validate_batch(images, labels, num_classes)
        if not result.passed:
            if self.enable_logging:
                self.logger.error(f"Batch validation failed: {result.message}")
            raise ValueError(f"Invalid batch: {result.message}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with advanced UI.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Use GPU tensors for accumulation to avoid sync overhead
        running_loss = torch.tensor(0.0, device=self.device)
        running_class_loss = torch.tensor(0.0, device=self.device)
        running_defect_loss = torch.tensor(0.0, device=self.device)
        correct = 0
        total = 0
        
        # Create advanced progress bar
        with create_advanced_progress() as progress:
            task = progress.add_task(
                f"[cyan]Epoch {epoch}/{total_epochs}[/cyan]",
                total=len(train_loader)
            )
            
            for batch_idx, (images, labels, severities, _) in enumerate(train_loader):
                try:
                    # Move to device (non-blocking for async transfer)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    defect_labels = (labels > 0).float().unsqueeze(1).to(self.device, non_blocking=True)
                    
                    # Validate batch
                    if self.enable_validation and batch_idx % 10 == 0:
                        self._validate_batch(images, labels, 6)
                    
                    # Forward pass with mixed precision
                    optimizer.zero_grad()
                    
                    if self.use_amp:
                        # Mixed precision forward pass (FP16)
                        with torch.amp.autocast('cuda'):
                            class_logits, defect_prob = self.model(images)
                            
                            # Calculate losses
                            class_loss = self.class_criterion(class_logits, labels)
                            defect_loss = self.defect_criterion(defect_prob, defect_labels)
                            loss = class_loss + 0.5 * defect_loss
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Optimizer step with scaler
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Standard FP32 training
                        class_logits, defect_prob = self.model(images)
                        
                        # Calculate losses
                        class_loss = self.class_criterion(class_logits, labels)
                        defect_loss = self.defect_criterion(defect_prob, defect_labels)
                        loss = class_loss + 0.5 * defect_loss
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                    
                    # Update metrics - accumulate tensors to avoid GPU sync
                    running_loss += loss.detach()
                    running_class_loss += class_loss.detach()
                    running_defect_loss += defect_loss.detach()
                    
                    _, predicted = torch.max(class_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress (sync every 10 batches to reduce overhead)
                    if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                        current_acc = 100 * correct / total
                        progress.update(
                            task,
                            advance=min(10, len(train_loader) - (batch_idx - 10)),
                            description=f"[cyan]Epoch {epoch}/{total_epochs}[/cyan] | Loss: {running_loss.item()/(batch_idx+1):.4f} | Acc: {current_acc:.2f}%"
                        )
                    else:
                        progress.update(task, advance=1)
                    
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    raise
        
        # Calculate epoch metrics (convert tensors to scalars at the end)
        metrics = {
            'loss': running_loss.item() / len(train_loader),
            'class_loss': running_class_loss.item() / len(train_loader),
            'defect_loss': running_defect_loss.item() / len(train_loader),
            'accuracy': 100 * correct / total
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model with progress tracking.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Use GPU tensors for accumulation to avoid sync overhead
        running_loss = torch.tensor(0.0, device=self.device)
        running_class_loss = torch.tensor(0.0, device=self.device)
        running_defect_loss = torch.tensor(0.0, device=self.device)
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with create_advanced_progress() as progress:
                task = progress.add_task(
                    "[magenta]Validating...[/magenta]",
                    total=len(val_loader)
                )
                
                for images, labels, severities, _ in val_loader:
                    # Non-blocking transfer for GPU
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    defect_labels = (labels > 0).float().unsqueeze(1).to(self.device, non_blocking=True)
                    
                    # Forward pass
                    class_logits, defect_prob = self.model(images)
                    
                    # Calculate losses
                    class_loss = self.class_criterion(class_logits, labels)
                    defect_loss = self.defect_criterion(defect_prob, defect_labels)
                    loss = class_loss + 0.5 * defect_loss
                    
                    # Update metrics - accumulate tensors to avoid GPU sync
                    running_loss += loss.detach()
                    running_class_loss += class_loss.detach()
                    running_defect_loss += defect_loss.detach()
                    
                    _, predicted = torch.max(class_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    progress.update(task, advance=1)
        
        metrics = {
            'loss': running_loss.item() / len(val_loader),
            'class_loss': running_class_loss.item() / len(val_loader),
            'defect_loss': running_defect_loss.item() / len(val_loader),
            'accuracy': 100 * correct / total,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 0.001,
        patience: int = 10
    ) -> Dict:
        """
        Full training loop with enhanced UI and monitoring.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            patience: Patience for learning rate scheduler
            
        Returns:
            Training history
        """
        # Log training start
        if self.enable_logging:
            config = {
                'epochs': epochs,
                'learning_rate': lr,
                'batch_size': train_loader.batch_size,
                'device': self.device,
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset)
            }
            self.logger.info(f"Training started: {epochs} epochs, LR={lr}, device={self.device}")
        
        # Show training header
        header = self.ui.create_title_panel(
            "CNN TRAINING INITIATED",
            f"Epochs: {epochs} | LR: {lr} | Device: {self.device}",
            style="primary"
        )
        self.console.print(header)
        
        # Setup optimizer and scheduler
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience
        )
        
        best_val_acc = 0.0
        training_start = time.time()
        
        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                
                # Train epoch
                train_metrics = self.train_epoch(train_loader, optimizer, epoch, epochs)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                epoch_time = time.time() - epoch_start
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['train_defect_loss'].append(train_metrics['defect_loss'])
                self.history['val_defect_loss'].append(val_metrics['defect_loss'])
                self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                self.history['epoch_times'].append(epoch_time)
                
                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])
                
                # Save best model in memory
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.best_val_acc = best_val_acc
                    self.best_model_state = {
                        'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                        'val_acc': best_val_acc,
                        'epoch': epoch
                    }
                
                # Display epoch summary
                self._display_epoch_summary(
                    epoch, epochs, train_metrics, val_metrics, best_val_acc, epoch_time
                )
                
                # Log to file
                if self.enable_logging:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% - "
                        f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%"
                    )
                
                # Callbacks
                for callback in self.callbacks:
                    callback(epoch, train_metrics, val_metrics)
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
            if self.enable_logging:
                self.logger.warning("Training interrupted by user")
        
        except Exception as e:
            self.console.print(f"\n[red]❌ Training failed: {str(e)}[/red]")
            if self.enable_logging:
                self.logger.error(f"Training failed", exc_info=True)
            raise
        
        finally:
            training_duration = time.time() - training_start
            
            # Show completion
            completion_panel = self.ui.create_title_panel(
                "TRAINING COMPLETE",
                f"Best Accuracy: {best_val_acc:.2f}% | Duration: {training_duration/60:.1f}min",
                style="success"
            )
            self.console.print("\n", completion_panel)
            
            if self.enable_logging:
                self.logger.info(f"Training completed - Best Acc: {best_val_acc:.2f}%, Duration: {training_duration/60:.1f}min")
        
        return self.history
    
    def _display_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict,
        best_val_acc: float,
        epoch_time: float
    ):
        """Display beautiful epoch summary."""
        
        # Create metrics table
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            title=f"[bold yellow]Epoch {epoch}/{total_epochs}[/bold yellow]"
        )
        
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Train", justify="right", style="green", width=12)
        table.add_column("Validation", justify="right", style="magenta", width=12)
        table.add_column("Status", justify="center", width=10)
        
        # Accuracy
        acc_status = "✓" if val_metrics['accuracy'] >= best_val_acc else "●"
        table.add_row(
            "Accuracy",
            f"{train_metrics['accuracy']:.2f}%",
            f"{val_metrics['accuracy']:.2f}%",
            f"[green]{acc_status}[/green]" if val_metrics['accuracy'] >= best_val_acc else acc_status
        )
        
        # Loss
        table.add_row(
            "Total Loss",
            f"{train_metrics['loss']:.4f}",
            f"{val_metrics['loss']:.4f}",
            "●"
        )
        
        # Class Loss
        table.add_row(
            "Class Loss",
            f"{train_metrics['class_loss']:.4f}",
            f"{val_metrics['class_loss']:.4f}",
            "●"
        )
        
        # Defect Loss
        table.add_row(
            "Defect Loss",
            f"{train_metrics['defect_loss']:.4f}",
            f"{val_metrics['defect_loss']:.4f}",
            "●"
        )
        
        # Time
        table.add_row(
            "Epoch Time",
            f"{epoch_time:.1f}s",
            "",
            ""
        )
        
        # Best accuracy footer
        table.caption = f"[bold green]Best Val Accuracy: {best_val_acc:.2f}%[/bold green]"
        
        self.console.print("\n", table)
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint (uses best model if available)."""
        if self.best_model_state:
            torch.save({
                'model_state_dict': self.best_model_state['model_state_dict'],
                'history': self.history,
                'best_val_acc': self.best_model_state['val_acc'],
                'epoch': self.best_model_state['epoch']
            }, filepath)
            
            if self.enable_logging:
                self.logger.info(f"Saved best model (acc: {self.best_model_state['val_acc']:.2f}%) to {filepath}")
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history
            }, filepath)
            
            if self.enable_logging:
                self.logger.info(f"Saved current model to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if self.enable_logging:
            acc = checkpoint.get('best_val_acc', 'N/A')
            self.logger.info(f"Loaded model from {filepath} (accuracy: {acc})")
        
        self.console.print(f"[green]✓[/green] Loaded model from {filepath}")


# Export
__all__ = ['EnhancedCNNTrainer']
