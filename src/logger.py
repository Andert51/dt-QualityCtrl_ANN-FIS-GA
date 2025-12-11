"""
Professional logging system for dt-QualityCtrl
Comprehensive tracking, rotating files, structured outputs
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
import json

try:
    import torch
except ImportError:
    torch = None


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Colorize level name
        record.levelname = f"{log_color}{record.levelname:8s}{reset}"
        
        # Format message
        formatted = super().format(record)
        
        return formatted


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        
        return json.dumps(log_data)


class SystemLogger:
    """
    Centralized logging system for the application.
    Handles console, file, and JSON logging.
    """
    
    def __init__(
        self,
        name: str = 'dt-QualityCtrl',
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize logging system.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            max_bytes: Max size per log file
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = log_dir or Path('output/logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Rotating file handler for general logs
        general_log = self.log_dir / 'system.log'
        file_handler = RotatingFileHandler(
            general_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        json_log = self.log_dir / 'system.jsonl'
        json_handler = RotatingFileHandler(
            json_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(file_level)
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
        
        # Error-specific handler
        error_log = self.log_dir / 'errors.log'
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        self.logger.info(f"Logging system initialized: {name}")
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self.logger
    
    def log_training_start(self, config: dict):
        """Log training session start."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING SESSION STARTED")
        self.logger.info("=" * 80)
        
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_training_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ):
        """Log epoch metrics."""
        self.logger.info(
            f"Epoch {epoch}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )
    
    def log_training_end(self, best_acc: float, duration: float):
        """Log training session end."""
        self.logger.info("=" * 80)
        self.logger.info(f"TRAINING COMPLETE | Best Accuracy: {best_acc:.2f}% | Duration: {duration:.1f}s")
        self.logger.info("=" * 80)
    
    def log_ga_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float
    ):
        """Log genetic algorithm generation."""
        self.logger.info(
            f"Generation {generation} | "
            f"Best: {best_fitness:.4f} | "
            f"Avg: {avg_fitness:.4f} | "
            f"Diversity: {diversity:.2%}"
        )
    
    def log_system_event(self, event: str, details: Optional[dict] = None):
        """Log system event with optional structured data."""
        extra = {'extra_data': details} if details else {}
        self.logger.info(event, extra=extra)
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """Log error with full traceback."""
        if context:
            self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(f"Error: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str):
        """Log warning."""
        self.logger.warning(message)
    
    def log_debug(self, message: str, data: Optional[dict] = None):
        """Log debug message with optional data."""
        if data:
            extra = {'extra_data': data}
            self.logger.debug(message, extra=extra)
        else:
            self.logger.debug(message)


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, log_dir: Path):
        """Initialize performance logger."""
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / 'performance.jsonl'
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        metrics: Optional[dict] = None
    ):
        """Log operation performance."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'success': success
        }
        
        if metrics:
            entry['metrics'] = metrics
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_resource_usage(self):
        """Log current resource usage."""
        try:
            import psutil
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / 1e9
            }
            
            # GPU if available
            if torch and torch.cuda.is_available():
                entry['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                entry['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            
            with open(self.log_dir / 'resources.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        
        except ImportError:
            pass
    
    def log_ga_generation(self, generation: int, best_fitness: float, avg_fitness: float, diversity: float):
        """Log genetic algorithm generation metrics."""
        entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity
        }
        
        with open(self.log_dir / 'ga_evolution.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


class ExperimentLogger:
    """Logger for ML experiments."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """Initialize experiment logger."""
        self.log_dir = log_dir / 'experiments'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.exp_dir = self.log_dir / self.experiment_id
        self.exp_dir.mkdir(exist_ok=True)
        
        self.config_file = self.exp_dir / 'config.json'
        self.metrics_file = self.exp_dir / 'metrics.jsonl'
        self.results_file = self.exp_dir / 'results.json'
    
    def save_config(self, config: dict):
        """Save experiment configuration."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def log_metric(self, step: int, metrics: dict):
        """Log metrics for a step."""
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def save_results(self, results: dict):
        """Save final experiment results."""
        results['experiment_id'] = self.experiment_id
        results['timestamp'] = datetime.now().isoformat()
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    def get_experiment_dir(self) -> Path:
        """Get experiment directory."""
        return self.exp_dir


# Global logger instance
_global_logger: Optional[SystemLogger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create global logger."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SystemLogger()
    
    if name:
        return logging.getLogger(name)
    else:
        return _global_logger.get_logger()


def initialize_logging(log_dir: Optional[Path] = None) -> SystemLogger:
    """Initialize global logging system."""
    global _global_logger
    
    _global_logger = SystemLogger(log_dir=log_dir)
    return _global_logger


# Export
__all__ = [
    'SystemLogger',
    'PerformanceLogger',
    'ExperimentLogger',
    'get_logger',
    'initialize_logging'
]
