"""
Training utilities for CIFAR-10 classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import time
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict


class ModelTrainer:
    """
    Training class for CIFAR-10 classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to run training on
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            logger: Logger for training progress
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.logger = logger or self._setup_logger()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger if none provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save the best model state
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Load best model if saved
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model state")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                # Overall accuracy
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Per-class accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1
        
        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / total
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for class_idx in range(10):  # CIFAR-10 has 10 classes
            if class_total[class_idx] > 0:
                class_accuracies[class_idx] = 100.0 * class_correct[class_idx] / class_total[class_idx]
            else:
                class_accuracies[class_idx] = 0.0
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'class_accuracies': class_accuracies
        }
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        self.logger.info(f"  Accuracy: {test_acc:.2f}%")
        
        return results
    
    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            **kwargs
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded from {filepath} (epoch {epoch})")
        
        return checkpoint