"""
GPU Optimization Utilities
Maximize GPU utilization for training and inference
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings


class GPUOptimizer:
    """
    Utilities for optimizing GPU usage and performance.
    """
    
    @staticmethod
    def get_optimal_device(prefer_gpu: bool = True) -> str:
        """
        Get the best available device with GPU optimization.
        
        Args:
            prefer_gpu: Prefer GPU if available
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if prefer_gpu and torch.cuda.is_available():
            # Clear cache for fresh start
            torch.cuda.empty_cache()
            
            # Get GPU info
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ GPU Detected: {device_name}")
            print(f"   Total Memory: {total_memory:.2f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            return 'cuda'
        else:
            if prefer_gpu:
                print("⚠️  GPU not available, using CPU")
            return 'cpu'
    
    @staticmethod
    def optimize_model_for_gpu(model: nn.Module, device: str) -> nn.Module:
        """
        Optimize model for GPU execution.
        
        Args:
            model: PyTorch model
            device: Target device
            
        Returns:
            Optimized model
        """
        model = model.to(device)
        
        if device == 'cuda':
            # Enable cudnn benchmarking for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 for Ampere GPUs (RTX 30xx, A100, etc.)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ TF32 enabled for faster training")
        
        return model
    
    @staticmethod
    def get_optimal_batch_size(device: str, default_cpu: int = 32, default_gpu: int = 64) -> int:
        """
        Get optimal batch size based on device.
        
        Args:
            device: Device string
            default_cpu: Default batch size for CPU
            default_gpu: Default batch size for GPU
            
        Returns:
            Optimal batch size
        """
        if device == 'cuda':
            # Check available memory
            free_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory_gb = free_memory / 1e9
            
            # Adjust batch size based on memory (conservative for stability)
            if free_memory_gb >= 10:
                batch_size = 128  # RTX 3080/3090
            elif free_memory_gb >= 6:
                batch_size = 64   # RTX 2060/2070/3060
            else:
                batch_size = 32   # Small GPU
            
            print(f"✅ Optimal GPU batch size: {batch_size}")
            return batch_size
        else:
            return default_cpu
    
    @staticmethod
    def enable_mixed_precision() -> tuple:
        """
        Enable mixed precision training (FP16) for faster GPU training.
        
        Returns:
            (scaler, enabled) tuple
        """
        if torch.cuda.is_available():
            try:
                scaler = torch.amp.GradScaler('cuda')
                print("✅ Mixed Precision (FP16) enabled - 2x faster training!")
                return scaler, True
            except Exception as e:
                print(f"⚠️  Mixed precision not available: {e}")
                return None, False
        return None, False
    
    @staticmethod
    def optimize_dataloader_workers(device: str) -> int:
        """
        Get optimal number of DataLoader workers.
        
        Args:
            device: Device string
            
        Returns:
            Optimal number of workers
        """
        import multiprocessing
        import platform
        
        # Windows has high overhead for multiprocessing, use 0 for best GPU performance
        if platform.system() == 'Windows':
            workers = 0
            print("✅ DataLoader workers: 0 (optimal for Windows + GPU - eliminates multiprocessing overhead)")
        elif device == 'cuda':
            # Linux/Mac can benefit from workers
            num_cpus = multiprocessing.cpu_count()
            workers = min(4, num_cpus)
            print(f"✅ DataLoader workers: {workers} (async GPU loading)")
        else:
            # CPU training
            num_cpus = multiprocessing.cpu_count()
            workers = min(2, num_cpus // 2)
        
        return workers
    
    @staticmethod
    def monitor_gpu_usage() -> Optional[Dict[str, float]]:
        """
        Monitor current GPU usage.
        
        Returns:
            Dictionary with GPU stats or None
        """
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100
        }
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
    
    @staticmethod
    def optimize_for_inference(model: nn.Module, device: str) -> nn.Module:
        """
        Optimize model for faster inference.
        
        Args:
            model: PyTorch model
            device: Device string
            
        Returns:
            Optimized model
        """
        model.eval()
        
        if device == 'cuda':
            # Use torch.jit for graph optimization
            try:
                # Create dummy input
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                
                # Trace model
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                print("✅ Model optimized with TorchScript JIT")
                return traced_model
            except Exception as e:
                print(f"⚠️  JIT optimization failed: {e}")
                return model
        
        return model
    
    @staticmethod
    def get_gpu_recommendations() -> Dict[str, Any]:
        """
        Get GPU optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'use_gpu': False,
            'mixed_precision': False,
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': False,
            'non_blocking': False
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            memory_gb = gpu_props.total_memory / 1e9
            
            recommendations['use_gpu'] = True
            recommendations['pin_memory'] = True  # Faster host-to-device transfer
            recommendations['non_blocking'] = True  # Async GPU operations
            
            # Get optimized num_workers (0 for Windows to avoid multiprocessing overhead)
            recommendations['num_workers'] = GPUOptimizer.optimize_dataloader_workers('cuda')
            
            # Memory-based batch size recommendations (conservative for stability)
            if memory_gb >= 10:
                recommendations['batch_size'] = 128  # RTX 3080/3090
                recommendations['mixed_precision'] = True
            elif memory_gb >= 6:
                recommendations['batch_size'] = 64  # RTX 2060/2070/3060
                recommendations['mixed_precision'] = True
            else:
                recommendations['batch_size'] = 32  # Small GPU
            
            # Compute capability checks
            compute_capability = gpu_props.major + gpu_props.minor / 10
            if compute_capability >= 7.0:  # Volta or newer
                recommendations['mixed_precision'] = True
        
        return recommendations
    
    @staticmethod
    def print_gpu_info():
        """Print detailed GPU information."""
        if not torch.cuda.is_available():
            print("❌ No GPU available")
            return
        
        print("\n" + "="*60)
        print("🎮 GPU CONFIGURATION")
        print("="*60)
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
            if i == 0:  # Current device
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Free: {(props.total_memory / 1e9) - allocated:.2f} GB")
        
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print("="*60 + "\n")


# Convenience functions
def setup_gpu_training(model: nn.Module, prefer_gpu: bool = True) -> tuple:
    """
    Complete GPU setup for training.
    
    Args:
        model: PyTorch model
        prefer_gpu: Prefer GPU if available
        
    Returns:
        (model, device, scaler, config) tuple
    """
    optimizer = GPUOptimizer()
    
    # Get device
    device = optimizer.get_optimal_device(prefer_gpu)
    
    # Optimize model
    model = optimizer.optimize_model_for_gpu(model, device)
    
    # Mixed precision
    scaler, use_amp = optimizer.enable_mixed_precision() if device == 'cuda' else (None, False)
    
    # Get recommendations
    config = optimizer.get_gpu_recommendations()
    
    # Print info
    if device == 'cuda':
        optimizer.print_gpu_info()
    
    return model, device, scaler, config


# Export
__all__ = [
    'GPUOptimizer',
    'setup_gpu_training'
]
