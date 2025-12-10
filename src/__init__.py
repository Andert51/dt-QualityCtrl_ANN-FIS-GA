"""
CyberCore-QC Source Package
============================
Hybrid Intelligent Quality Control System

Modules:
- data_generator: Synthetic industrial defect image generator
- cnn_model: CNN defect detection (ResNet18-based)
- fuzzy_system: Fuzzy inference system for severity assessment
- genetic_algorithm: GA optimizer for FIS parameters
- visualizations: Comprehensive visualization suite
"""

__version__ = "1.0.0"
__author__ = "CyberCore AI Lab"
__all__ = [
    "SyntheticDefectGenerator",
    "DefectCNN",
    "DefectCNNTrainer",
    "FuzzyQualityController",
    "GeneticAlgorithm",
    "VisualizationHub"
]

from .data_generator import SyntheticDefectGenerator
from .cnn_model import DefectCNN, DefectCNNTrainer
from .fuzzy_system import FuzzyQualityController
from .genetic_algorithm import GeneticAlgorithm
from .visualizations import VisualizationHub
