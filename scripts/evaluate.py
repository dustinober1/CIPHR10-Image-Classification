#!/usr/bin/env python3
"""
Evaluation script for CIFAR-10 classification model.
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import CIFAR10DataModule, CIFAR10_CLASSES
from src.models.cifar10_cnn import get_model
from src.utils.trainer import ModelTrainer
from src.utils.visualization import (
    plot_confusion_matrix, plot_class_accuracies, 
    visualize_predictions, create_results_summary
)
from configs.config import ExperimentConfig, get_default_config


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: ExperimentConfig
):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for evaluation
        config: Experiment configuration
    """
    # Create trainer for evaluation
    trainer = ModelTrainer(
        model=model,
        device=device,
        criterion=nn.CrossEntropyLoss()
    )
    
    # Basic evaluation
    test_results = trainer.evaluate(test_loader)
    
    # Get predictions for detailed analysis
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Plot confusion matrix
    confusion_path = Path(config.system.plot_dir) / f"{config.name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=all_targets,
        y_pred=all_predictions,
        class_names=CIFAR10_CLASSES,
        title=f"Confusion Matrix - {config.name}",
        save_path=str(confusion_path)
    )
    
    # Plot class accuracies
    class_acc_path = Path(config.system.plot_dir) / f"{config.name}_class_accuracies.png"
    plot_class_accuracies(
        class_accuracies=test_results['class_accuracies'],
        class_names=CIFAR10_CLASSES,
        title=f"Per-Class Accuracy - {config.name}",
        save_path=str(class_acc_path)
    )
    
    # Visualize predictions
    visualize_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        num_samples=8,
        class_names=CIFAR10_CLASSES,
        title=f"Model Predictions - {config.name}"
    )
    
    # Create results summary
    summary_path = Path(config.system.output_dir) / f"{config.name}_results_summary.csv"
    summary_df = create_results_summary(
        test_results=test_results,
        class_names=CIFAR10_CLASSES,
        save_path=str(summary_path)
    )
    
    print("\nResults Summary:")
    print("=" * 50)
    print(summary_df.to_string(index=False))
    print("=" * 50)
    
    return test_results, all_predictions, all_targets


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 classification model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config file (optional)')
    parser.add_argument('--name', type=str,
                       help='Experiment name for output files')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str,
                       help='Device to use (cpu/cuda/mps)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.name:
        config.name = args.name
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.system.device = args.device
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    # Setup device
    device = torch.device(config.system.device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model configuration from checkpoint if available
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if 'model' in saved_config:
            config.model.model_name = saved_config['model'].get('model_name', 'improved')
            config.model.num_classes = saved_config['model'].get('num_classes', 10)
            config.model.dropout_rate = saved_config['model'].get('dropout_rate', 0.3)
    
    # Create model
    print(f"Creating model: {config.model.model_name}")
    model = get_model(
        model_name=config.model.model_name,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        **config.model.model_params
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded successfully on {device}")
    
    # Create data module
    print("Setting up data loaders...")
    data_module = CIFAR10DataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        val_split=config.data.val_split,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    _, _, test_loader = data_module.get_dataloaders()
    
    # Evaluate model
    print("Starting evaluation...")
    test_results, predictions, targets = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config
    )
    
    # Print additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Total test samples: {len(targets)}")
    print(f"Correctly classified: {np.sum(predictions == targets)}")
    print(f"Misclassified: {np.sum(predictions != targets)}")
    
    # Per-class statistics
    print(f"\nPer-class accuracy:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        acc = test_results['class_accuracies'][i]
        print(f"  {class_name}: {acc:.2f}%")
    
    # Find best and worst performing classes
    class_accs = [test_results['class_accuracies'][i] for i in range(10)]
    best_class_idx = np.argmax(class_accs)
    worst_class_idx = np.argmin(class_accs)
    
    print(f"\nBest performing class: {CIFAR10_CLASSES[best_class_idx]} ({class_accs[best_class_idx]:.2f}%)")
    print(f"Worst performing class: {CIFAR10_CLASSES[worst_class_idx]} ({class_accs[worst_class_idx]:.2f}%)")
    
    print(f"\nEvaluation completed. Results saved to: {config.system.output_dir}")


if __name__ == "__main__":
    main()