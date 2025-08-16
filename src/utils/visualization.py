"""
Visualization utilities for CIFAR-10 classification project.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def denormalize(tensor: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]
    if std is None:
        std = [0.2023, 0.1994, 0.2010]
    
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return torch.clamp(tensor, 0, 1)


def show_sample_images(
    dataloader: torch.utils.data.DataLoader,
    num_images: int = 8,
    title: str = "Sample Images",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Display sample images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_images: Number of images to display
        title: Title for the plot
        figsize: Figure size
    """
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select subset of images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Denormalize images
    images = denormalize(images)
    
    # Create plot
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(f'{CIFAR10_CLASSES[labels[i]]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    learning_rates: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training history including losses, accuracy, and learning rate.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        val_accuracies: Validation accuracies per epoch
        learning_rates: Learning rates per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    if learning_rates:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss comparison
    ax3.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    ax3.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    ax3.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot learning rate if provided
    if learning_rates:
        ax4.plot(epochs, learning_rates, 'orange', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Title for the plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_class_accuracies(
    class_accuracies: Dict[int, float],
    class_names: List[str] = None,
    title: str = "Per-Class Accuracy",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot per-class accuracy as a bar chart.
    
    Args:
        class_accuracies: Dictionary mapping class indices to accuracies
        class_names: Names of classes
        title: Title for the plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    # Prepare data
    classes = [class_names[i] for i in sorted(class_accuracies.keys())]
    accuracies = [class_accuracies[i] for i in sorted(class_accuracies.keys())]
    
    # Create bar plot
    plt.figure(figsize=figsize)
    bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for mean accuracy
    mean_acc = np.mean(accuracies)
    plt.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_acc:.1f}%')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracies plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 8,
    class_names: List[str] = None,
    title: str = "Model Predictions",
    figsize: Tuple[int, int] = (16, 8)
):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device for inference
        num_samples: Number of samples to visualize
        class_names: Names of classes
        title: Title for the plot
        figsize: Figure size
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    model.eval()
    
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select subset
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        predictions = predictions.cpu()
        confidences = confidences.cpu()
    
    # Denormalize images for visualization
    images_denorm = denormalize(images)
    
    # Create plot
    fig, axes = plt.subplots(2, num_samples//2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        image = images_denorm[i].permute(1, 2, 0).numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        confidence = confidences[i].item()
        
        axes[i].imshow(image)
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].set_title(
            f'True: {class_names[true_label]}\\n'
            f'Pred: {class_names[pred_label]} ({confidence:.2f})',
            color=color,
            fontsize=10,
            fontweight='bold'
        )
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_results_summary(
    test_results: Dict,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a summary table of test results.
    
    Args:
        test_results: Dictionary with test results
        class_names: Names of classes
        save_path: Path to save the summary
        
    Returns:
        DataFrame with results summary
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    # Create summary DataFrame
    summary_data = []
    
    # Overall results
    summary_data.append({
        'Metric': 'Overall Test Accuracy',
        'Value': f"{test_results['test_accuracy']:.2f}%"
    })
    
    summary_data.append({
        'Metric': 'Overall Test Loss',
        'Value': f"{test_results['test_loss']:.4f}"
    })
    
    # Per-class accuracies
    class_accs = test_results.get('class_accuracies', {})
    for class_idx, acc in class_accs.items():
        summary_data.append({
            'Metric': f'{class_names[class_idx]} Accuracy',
            'Value': f"{acc:.2f}%"
        })
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results summary saved to {save_path}")
    
    return df