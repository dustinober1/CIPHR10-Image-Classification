"""
Configuration management for CIFAR-10 classification project.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "./data"
    batch_size: int = 64
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True
    
    # Data augmentation settings
    use_augmentation: bool = True
    random_horizontal_flip: float = 0.5
    random_rotation: int = 15
    color_jitter: Dict[str, float] = field(default_factory=lambda: {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    })


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_name: str = "improved"  # 'improved' or 'simple'
    num_classes: int = 10
    dropout_rate: float = 0.3
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer settings
    optimizer: str = "adam"  # 'adam', 'sgd', 'adamw'
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 50,  # for cosine annealing
        'step_size': 20,  # for step scheduler
        'gamma': 0.1,  # for step scheduler
        'patience': 5,  # for plateau scheduler
        'factor': 0.5  # for plateau scheduler
    })
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5  # Save every N epochs
    save_best_only: bool = True


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "./logs"
    
    # Output directories
    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    plot_dir: str = "./plots"


@dataclass
class ExperimentConfig:
    """Main configuration class that combines all sub-configs."""
    name: str = "cifar10_experiment"
    description: str = "CIFAR-10 image classification experiment"
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set up device
        if self.system.device == "auto":
            if torch.cuda.is_available():
                self.system.device = "cuda"
            elif torch.backends.mps.is_available():
                self.system.device = "mps"
            else:
                self.system.device = "cpu"
        
        # Create directories
        self._create_directories()
        
        # Set random seeds
        self._set_seeds()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.system.log_dir,
            self.system.output_dir,
            self.system.checkpoint_dir,
            self.system.plot_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.system.seed)
        torch.cuda.manual_seed_all(self.system.seed)
        
        if self.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self.system.benchmark
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__
        }
    
    def save(self, filepath: str):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = cls(
            name=config_dict.get('name', 'loaded_experiment'),
            description=config_dict.get('description', 'Loaded from file')
        )
        
        # Update sub-configs
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'system' in config_dict:
            for key, value in config_dict['system'].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        return config


# Predefined configurations for different scenarios
def get_quick_config() -> ExperimentConfig:
    """Get configuration for quick testing."""
    config = ExperimentConfig(
        name="quick_test",
        description="Quick test configuration with minimal epochs"
    )
    config.training.epochs = 5
    config.data.batch_size = 32
    config.training.early_stopping_patience = 3
    
    return config


def get_high_performance_config() -> ExperimentConfig:
    """Get configuration optimized for high performance."""
    config = ExperimentConfig(
        name="high_performance",
        description="High performance configuration with optimized settings"
    )
    config.model.model_name = "improved"
    config.model.dropout_rate = 0.2
    config.training.epochs = 100
    config.training.learning_rate = 0.001
    config.training.optimizer = "adamw"
    config.training.scheduler_type = "cosine"
    config.data.batch_size = 128
    
    return config


def get_cpu_config() -> ExperimentConfig:
    """Get configuration optimized for CPU training."""
    config = ExperimentConfig(
        name="cpu_training",
        description="Configuration optimized for CPU training"
    )
    config.system.device = "cpu"
    config.data.batch_size = 32
    config.data.num_workers = 2
    config.data.pin_memory = False
    config.training.epochs = 20
    
    return config


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()