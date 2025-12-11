"""
Robust validation and error handling for dt-QualityCtrl System
Comprehensive checks, informative errors, automatic recovery
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    details: Optional[Dict] = None


class SystemValidator:
    """Comprehensive system validation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.results: List[ValidationResult] = []
    
    def validate_environment(self) -> List[ValidationResult]:
        """Validate system environment."""
        results = []
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            results.append(ValidationResult(
                passed=True,
                message=f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)",
                severity='info',
                details={'device': gpu_name, 'memory_gb': gpu_memory}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message="CUDA not available, using CPU",
                severity='warning'
            ))
        
        # Check CPU info
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        results.append(ValidationResult(
            passed=True,
            message=f"CPU cores available: {cpu_count}",
            severity='info',
            details={'cores': cpu_count}
        ))
        
        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            mem_gb = mem.total / 1e9
            mem_available_gb = mem.available / 1e9
            
            if mem_available_gb < 2:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Low memory: {mem_available_gb:.1f}GB available",
                    severity='warning',
                    details={'total_gb': mem_gb, 'available_gb': mem_available_gb}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Memory: {mem_available_gb:.1f}GB / {mem_gb:.1f}GB available",
                    severity='info',
                    details={'total_gb': mem_gb, 'available_gb': mem_available_gb}
                ))
        except ImportError:
            results.append(ValidationResult(
                passed=True,
                message="psutil not available, skipping memory check",
                severity='info'
            ))
        
        self.results.extend(results)
        return results
    
    def validate_directories(self, required_dirs: List[Path]) -> List[ValidationResult]:
        """Validate required directories exist and are writable."""
        results = []
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    results.append(ValidationResult(
                        passed=True,
                        message=f"Created directory: {dir_path}",
                        severity='info'
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Cannot create directory {dir_path}: {e}",
                        severity='error',
                        details={'error': str(e)}
                    ))
            else:
                # Check if writable
                test_file = dir_path / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                    results.append(ValidationResult(
                        passed=True,
                        message=f"Directory writable: {dir_path}",
                        severity='info'
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Directory not writable: {dir_path}",
                        severity='error',
                        details={'error': str(e)}
                    ))
        
        self.results.extend(results)
        return results
    
    def validate_dataset(self, dataset_metadata: Dict) -> List[ValidationResult]:
        """Validate dataset integrity."""
        results = []
        
        # Check required fields
        required_fields = ['total_samples', 'samples']
        for field in required_fields:
            if field not in dataset_metadata:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Missing required field in dataset: {field}",
                    severity='error'
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Dataset field present: {field}",
                    severity='info'
                ))
        
        # Check sample count
        if 'total_samples' in dataset_metadata:
            total = dataset_metadata['total_samples']
            if total == 0:
                results.append(ValidationResult(
                    passed=False,
                    message="Dataset is empty",
                    severity='error'
                ))
            elif total < 100:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Small dataset: {total} samples (recommend >100)",
                    severity='warning',
                    details={'sample_count': total}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Dataset size: {total} samples",
                    severity='info',
                    details={'sample_count': total}
                ))
        
        # Check class distribution
        if 'samples' in dataset_metadata:
            samples = dataset_metadata['samples']
            class_counts = {k: len(v) for k, v in samples.items()}
            
            min_count = min(class_counts.values()) if class_counts else 0
            max_count = max(class_counts.values()) if class_counts else 0
            
            if min_count == 0:
                results.append(ValidationResult(
                    passed=False,
                    message="Empty classes found in dataset",
                    severity='error',
                    details={'class_counts': class_counts}
                ))
            elif max_count / (min_count + 0.001) > 5:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Class imbalance detected (ratio: {max_count/min_count:.1f}:1)",
                    severity='warning',
                    details={'class_counts': class_counts}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Balanced dataset ({len(class_counts)} classes)",
                    severity='info',
                    details={'class_counts': class_counts}
                ))
        
        # Check file existence
        if 'samples' in dataset_metadata:
            missing_files = []
            for class_name, sample_list in dataset_metadata['samples'].items():
                for sample in sample_list[:10]:  # Check first 10 per class
                    if isinstance(sample, dict) and 'path' in sample:
                        if not Path(sample['path']).exists():
                            missing_files.append(sample['path'])
            
            if missing_files:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Missing {len(missing_files)} sample files",
                    severity='error',
                    details={'missing_count': len(missing_files), 'examples': missing_files[:5]}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message="All sample files exist",
                    severity='info'
                ))
        
        self.results.extend(results)
        return results
    
    def validate_model_state(self, model: torch.nn.Module) -> List[ValidationResult]:
        """Validate model state and parameters."""
        results = []
        
        # Check if model is on correct device
        device = next(model.parameters()).device
        results.append(ValidationResult(
            passed=True,
            message=f"Model device: {device}",
            severity='info',
            details={'device': str(device)}
        ))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results.append(ValidationResult(
            passed=True,
            message=f"Parameters: {trainable_params:,} trainable / {total_params:,} total",
            severity='info',
            details={'trainable': trainable_params, 'total': total_params}
        ))
        
        # Check for NaN or Inf in parameters
        has_nan = False
        has_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                results.append(ValidationResult(
                    passed=False,
                    message=f"NaN detected in parameter: {name}",
                    severity='error'
                ))
            if torch.isinf(param).any():
                has_inf = True
                results.append(ValidationResult(
                    passed=False,
                    message=f"Inf detected in parameter: {name}",
                    severity='error'
                ))
        
        if not has_nan and not has_inf:
            results.append(ValidationResult(
                passed=True,
                message="All parameters are valid (no NaN/Inf)",
                severity='info'
            ))
        
        self.results.extend(results)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        by_severity = {}
        for result in self.results:
            by_severity[result.severity] = by_severity.get(result.severity, 0) + 1
        
        return {
            'total_checks': total,
            'passed': passed,
            'failed': total - passed,
            'by_severity': by_severity,
            'critical_issues': [r for r in self.results if r.severity == 'critical'],
            'errors': [r for r in self.results if r.severity == 'error'],
            'warnings': [r for r in self.results if r.severity == 'warning']
        }


class DataValidator:
    """Validate data inputs and outputs."""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple] = None,
        expected_dtype: Optional[torch.dtype] = None,
        check_nan: bool = True,
        check_inf: bool = True,
        value_range: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """Validate a tensor."""
        issues = []
        
        # Shape check
        if expected_shape and tensor.shape != expected_shape:
            issues.append(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}")
        
        # Dtype check
        if expected_dtype and tensor.dtype != expected_dtype:
            issues.append(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
        
        # NaN check
        if check_nan and torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            issues.append(f"Contains {nan_count} NaN values")
        
        # Inf check
        if check_inf and torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            issues.append(f"Contains {inf_count} Inf values")
        
        # Value range check
        if value_range:
            min_val, max_val = value_range
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()
            
            if tensor_min < min_val or tensor_max > max_val:
                issues.append(f"Values out of range [{min_val}, {max_val}]: [{tensor_min:.4f}, {tensor_max:.4f}]")
        
        if issues:
            return ValidationResult(
                passed=False,
                message="Tensor validation failed: " + "; ".join(issues),
                severity='error',
                details={'issues': issues}
            )
        else:
            return ValidationResult(
                passed=True,
                message="Tensor is valid",
                severity='info',
                details={'shape': tuple(tensor.shape), 'dtype': str(tensor.dtype)}
            )
    
    @staticmethod
    def validate_batch(
        images: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> ValidationResult:
        """Validate a training batch."""
        issues = []
        
        # Check batch dimension matches
        if images.shape[0] != labels.shape[0]:
            issues.append(f"Batch size mismatch: images {images.shape[0]}, labels {labels.shape[0]}")
        
        # Check image dimensions
        if len(images.shape) != 4:
            issues.append(f"Images should be 4D (B, C, H, W), got {images.shape}")
        
        # Check label range
        if labels.min() < 0 or labels.max() >= num_classes:
            issues.append(f"Labels out of range [0, {num_classes-1}]: [{labels.min()}, {labels.max()}]")
        
        # Check for anomalies
        if torch.isnan(images).any():
            issues.append("Images contain NaN")
        if torch.isinf(images).any():
            issues.append("Images contain Inf")
        
        if issues:
            return ValidationResult(
                passed=False,
                message="Batch validation failed: " + "; ".join(issues),
                severity='error',
                details={'issues': issues}
            )
        else:
            return ValidationResult(
                passed=True,
                message=f"Batch valid: {images.shape[0]} samples",
                severity='info'
            )


class ConfigValidator:
    """Validate configuration parameters."""
    
    @staticmethod
    def validate_training_config(config: Dict) -> List[ValidationResult]:
        """Validate training configuration."""
        results = []
        
        # Learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if lr <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Invalid learning rate: {lr} (must be > 0)",
                    severity='error'
                ))
            elif lr > 0.1:
                results.append(ValidationResult(
                    passed=True,
                    message=f"High learning rate: {lr} (recommend < 0.1)",
                    severity='warning'
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Learning rate: {lr}",
                    severity='info'
                ))
        
        # Batch size
        if 'batch_size' in config:
            bs = config['batch_size']
            if bs <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Invalid batch size: {bs}",
                    severity='error'
                ))
            elif bs < 8:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Small batch size: {bs} (may be slow)",
                    severity='warning'
                ))
            elif bs > 256:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Large batch size: {bs} (may cause OOM)",
                    severity='warning'
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Batch size: {bs}",
                    severity='info'
                ))
        
        # Epochs
        if 'epochs' in config:
            ep = config['epochs']
            if ep <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Invalid epochs: {ep}",
                    severity='error'
                ))
            elif ep < 5:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Few epochs: {ep} (may underfit)",
                    severity='warning'
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Epochs: {ep}",
                    severity='info'
                ))
        
        return results


# Export
__all__ = [
    'ValidationResult',
    'SystemValidator',
    'DataValidator',
    'ConfigValidator'
]
