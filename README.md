# CIFAR-10 Image Classification

A modern PyTorch implementation for CIFAR-10 image classification with improved architecture, comprehensive training pipeline, and extensive evaluation tools.

![CIFAR-10 Classes](https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png)

## Overview

This project implements a convolutional neural network (CNN) for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### Key Features

- **Modern PyTorch Implementation**: Clean, modular code structure following best practices
- **Multiple Model Architectures**: Simple and improved CNN variants
- **Comprehensive Training Pipeline**: Configurable training with various optimizers and schedulers
- **Advanced Data Augmentation**: Robust preprocessing and augmentation techniques
- **Extensive Evaluation Tools**: Confusion matrices, per-class accuracy, visualization utilities
- **Configuration Management**: Flexible configuration system for experiments
- **Reproducible Results**: Seed management and deterministic training
- **Command-Line Interface**: Easy-to-use training and evaluation scripts

## Performance

- **Target Accuracy**: 70%+ (original project achieved 72.2%)
- **Improved Model**: Optimized architecture with better regularization
- **Training Time**: Efficient training with GPU support

### Benchmark Comparisons

| Method | Accuracy | Reference |
|--------|----------|-----------|
| Deep Belief Networks | 78.9% | [Krizhevsky, 2010](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf) |
| Maxout Networks | 90.6% | [Goodfellow et al., 2013](https://arxiv.org/pdf/1302.4389.pdf) |
| Wide Residual Networks | 96.0% | [Zagoruyko et al., 2016](https://arxiv.org/pdf/1605.07146.pdf) |
| GPipe | 99.0% | [Huang et al., 2018](https://arxiv.org/pdf/1811.06965.pdf) |
| **This Implementation** | **72.2%** | Achievable with simple CNN |

## Project Structure

```
CIFAR10-Image-Classification/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── cifar10_cnn.py          # CNN model architectures
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # Data loading and preprocessing
│   └── utils/
│       ├── __init__.py
│       ├── trainer.py              # Training utilities
│       └── visualization.py       # Plotting and visualization
├── configs/
│   └── config.py                   # Configuration management
├── scripts/
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation script
├── notebooks/
│   └── Project_2_CIFAR_10_Image_Classifier-2.ipynb  # Original notebook
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU training)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd CIFAR10-Image-Classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## Quick Start

### Training a Model

1. **Basic training** with default configuration:
   ```bash
   python scripts/train.py --name my_experiment
   ```

2. **Custom training** with specific parameters:
   ```bash
   python scripts/train.py \\
       --name high_performance \\
       --model improved \\
       --epochs 100 \\
       --batch-size 128 \\
       --lr 0.001
   ```

3. **Training with configuration file**:
   ```bash
   python scripts/train.py --config configs/my_config.json
   ```

### Evaluating a Model

1. **Evaluate a trained model**:
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/my_experiment_final.pth
   ```

2. **Detailed evaluation with visualizations**:
   ```bash
   python scripts/evaluate.py \\
       --checkpoint checkpoints/my_experiment_final.pth \\
       --name evaluation_results
   ```

## Usage

### Configuration

The project uses a comprehensive configuration system. You can:

1. **Use predefined configurations**:
   ```python
   from configs.config import get_high_performance_config, get_quick_config
   
   config = get_high_performance_config()
   ```

2. **Create custom configurations**:
   ```python
   from configs.config import ExperimentConfig
   
   config = ExperimentConfig(
       name="custom_experiment",
       description="My custom configuration"
   )
   config.training.epochs = 50
   config.model.dropout_rate = 0.2
   ```

3. **Load/save configurations**:
   ```python
   # Save
   config.save("my_config.json")
   
   # Load
   config = ExperimentConfig.load("my_config.json")
   ```

### Data Loading

```python
from src.data.dataset import CIFAR10DataModule

# Create data module
data_module = CIFAR10DataModule(
    data_dir="./data",
    batch_size=64,
    val_split=0.1
)

# Get data loaders
train_loader, val_loader, test_loader = data_module.get_dataloaders()
```

### Model Training

```python
from src.models.cifar10_cnn import get_model
from src.utils.trainer import ModelTrainer

# Create model
model = get_model("improved", num_classes=10, dropout_rate=0.3)

# Create trainer
trainer = ModelTrainer(model=model, device=device)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)
```

### Visualization

```python
from src.utils.visualization import plot_training_history, show_sample_images

# Plot training progress
plot_training_history(
    train_losses=history['train_losses'],
    val_losses=history['val_losses'],
    val_accuracies=history['val_accuracies']
)

# Show sample images
show_sample_images(train_loader, num_images=8)
```

## Model Architectures

### Simple CNN
- 3 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected classifier

### Improved CNN (Recommended)
- 6 convolutional layers with residual-like connections
- Advanced regularization techniques
- Adaptive global average pooling
- Optimized for better performance

## Configuration Options

### Data Configuration
- `batch_size`: Training batch size (default: 64)
- `val_split`: Validation split ratio (default: 0.1)
- `num_workers`: Data loading workers (default: 4)
- Data augmentation settings (rotation, flipping, color jitter)

### Training Configuration
- `epochs`: Number of training epochs (default: 50)
- `learning_rate`: Initial learning rate (default: 0.001)
- `optimizer`: Optimizer choice (adam, sgd, adamw)
- `scheduler`: Learning rate scheduler (cosine, step, plateau)
- Early stopping and checkpointing options

### Model Configuration
- `model_name`: Architecture choice (simple, improved)
- `dropout_rate`: Dropout probability (default: 0.3)
- `num_classes`: Number of output classes (default: 10)

## Results and Outputs

The training process generates several outputs:

- **Checkpoints**: Saved model states in `checkpoints/`
- **Logs**: Training logs in `logs/`
- **Plots**: Training curves and evaluation plots in `plots/`
- **Results**: Evaluation summaries in `results/`

### Example Outputs

1. **Training History Plot**: Loss and accuracy curves over epochs
2. **Confusion Matrix**: Detailed classification performance
3. **Per-Class Accuracy**: Individual class performance metrics
4. **Sample Predictions**: Visualized model predictions

## Reproducibility

To ensure reproducible results:

1. **Set random seeds** in configuration
2. **Use deterministic algorithms** when possible
3. **Pin data loader workers** for consistent data ordering
4. **Save complete experiment configurations**

## Performance Tips

### For Better Accuracy
- Use the improved model architecture
- Increase training epochs (50-100)
- Enable data augmentation
- Use learning rate scheduling
- Consider ensemble methods

### For Faster Training
- Increase batch size (if memory allows)
- Use multiple data loading workers
- Enable mixed precision training
- Use GPU acceleration

### For Debugging
- Start with quick configuration (5 epochs)
- Use smaller batch sizes
- Enable verbose logging
- Visualize sample batches

## Contributing

This project was created as part of a Udacity Machine Learning Nanodegree. Contributions for improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Add tests if applicable
5. Submit a pull request

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical plotting
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **Udacity**: For the original project structure and learning materials
- **CIFAR-10 Dataset**: Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **PyTorch Team**: For the excellent deep learning framework
- **Research Community**: For the benchmark methods and architectures referenced

## Future Improvements

Potential enhancements for this project:

1. **Advanced Architectures**: ResNet, DenseNet, EfficientNet implementations
2. **Transfer Learning**: Pre-trained model fine-tuning
3. **Hyperparameter Optimization**: Automated hyperparameter search
4. **Model Compression**: Quantization and pruning techniques
5. **Deployment**: Model serving and inference optimization
6. **Web Interface**: Streamlit or Gradio app for interactive classification

---

**Note**: This project demonstrates modern machine learning engineering practices and can serve as a template for similar image classification tasks.