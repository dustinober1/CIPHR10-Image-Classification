#!/usr/bin/env python3
"""
Training script for CIFAR-10 classification model.
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import CIFAR10DataModule
from src.models.cifar10_cnn import get_model
from src.utils.trainer import ModelTrainer
from src.utils.visualization import plot_training_history
from configs.config import ExperimentConfig, get_default_config


def setup_logging(config: ExperimentConfig):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if requested
    if config.system.log_to_file:
        log_file = Path(config.system.log_dir) / f"{config.name}_train.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def get_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer: optim.Optimizer, config: ExperimentConfig):
    """Create learning rate scheduler based on configuration."""
    if not config.training.use_scheduler:
        return None
    
    scheduler_type = config.training.scheduler_type.lower()
    params = config.training.scheduler_params
    
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', config.training.epochs)
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=params.get('step_size', 20),
            gamma=params.get('gamma', 0.1)
        )
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params.get('patience', 5),
            factor=params.get('factor', 0.5),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classification model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['simple', 'improved'], help='Model architecture')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda/mps)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.name:
        config.name = args.name
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.model:
        config.model.model_name = args.model
    if args.device:
        config.system.device = args.device
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Using device: {config.system.device}")
    
    # Save configuration
    config_path = Path(config.system.output_dir) / f"{config.name}_config.json"
    config.save(str(config_path))
    
    # Setup device
    device = torch.device(config.system.device)
    
    # Create data module
    logger.info("Setting up data loaders...")
    data_module = CIFAR10DataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        val_split=config.data.val_split,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Create model
    logger.info(f"Creating model: {config.model.model_name}")
    model = get_model(
        model_name=config.model.model_name,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        **config.model.model_params
    )
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        save_best=config.training.save_best_only,
        early_stopping_patience=config.training.early_stopping_patience if config.training.early_stopping else None
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    
    # Save final checkpoint
    if config.training.save_checkpoints:
        checkpoint_path = Path(config.system.checkpoint_dir) / f"{config.name}_final.pth"
        trainer.save_checkpoint(
            filepath=str(checkpoint_path),
            epoch=config.training.epochs,
            config=config.to_dict(),
            test_results=test_results
        )
    
    # Plot training history
    plot_path = Path(config.system.plot_dir) / f"{config.name}_training_history.png"
    plot_training_history(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        val_accuracies=history['val_accuracies'],
        learning_rates=history.get('learning_rates'),
        save_path=str(plot_path)
    )
    
    # Print final results
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    logger.info(f"Final test accuracy: {test_results['test_accuracy']:.2f}%")
    logger.info(f"Final test loss: {test_results['test_loss']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()