# Knowledge Distillation with PyTorch on CIFAR-10

This repository contains an implementation of various knowledge distillation techniques using PyTorch on the CIFAR-10 dataset. Knowledge distillation is a model compression technique where a smaller student model learns to mimic a larger, more powerful teacher model.

Motivation come from https://github.com/geohot/ai-notebooks/blob/master/cifar10_distillation.ipynb

## 📚 Background

Knowledge distillation, introduced by [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531), enables the transfer of knowledge from a large, computationally expensive model to a smaller, more efficient one. This is particularly useful for deployment on resource-constrained devices.

In this project, we implement and compare three approaches:

1. **Standard Training**: Training a small model directly on the dataset
2. **Logits-based Distillation**: Training a small model to mimic the output probabilities of a larger model
3. **Feature-based Distillation**: Training a small model to mimic both the outputs and intermediate representations of a larger model

## 🏗️ Project Structure

```
.
├── distil_pytorch/             # Main package
│   ├── models/                 # CNN model implementations
│   ├── utils/                  # Utility functions for training and distillation
│   └── data/                   # Data loading utilities
├── distilation.ipynb           # Main notebook with experimental results
├── setup.py                    # Package setup file
├── accuracy_comparison.png     # Results visualization
├── distillation_comparison.png # Learning curves comparison
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- matplotlib 3.4+
- numpy 1.20+

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/knowledge-distillation-pytorch.git
   cd knowledge-distillation-pytorch
   ```

2. Install the package in development mode:
   ```
   pip install -e .
   ```

3. Run the Jupyter notebook:
   ```
   jupyter notebook distilation.ipynb
   ```

## 🔍 Implementation Details

### Models

- **Teacher**: CNN with scale factor 32 (~4.3M parameters)
- **Student**: CNN with scale factor 16 (~1.1M parameters)

Both models use a similar architecture with convolutional layers, batch normalization, max pooling, and dropout.

### Distillation Methods

1. **Standard Training**:
   - Uses standard cross-entropy loss with ground truth labels

2. **Logits-based Distillation**:
   - Combines hard loss (cross-entropy with true labels) and soft loss (KL divergence between softened logits)
   - Uses temperature parameter T to control softening
   - Loss = α * hard_loss + (1-α) * soft_loss

3. **Feature-based Distillation**:
   - Extends logits distillation by also matching intermediate feature maps
   - Loss = α * hard_loss + β * feature_loss + (1-α-β) * soft_loss

## 📊 Results

Our experiments show significant findings:

| Model/Method | Parameters | Accuracy |
|--------------|------------|----------|
| Teacher | ~4.3M | 82.44% |
| Student (Standard) | ~1.1M | 75.83% |
| Student (Logits) | ~1.1M | 66.54% |
| Student (Feature) | ~1.1M | 77.34% |

Key observations:
- Feature-based distillation achieved the best results, retaining 94% of the teacher's accuracy with only 25% of the parameters
- Surprisingly, logits-based distillation performed worse than standard training
- Feature distillation outperformed logits distillation by a substantial margin (10.80%)

## 🔬 Analysis

1. **Feature distillation is superior**: Intermediate representations contain critical information that output probabilities alone don't capture.

2. **Architecture matters**: The relatively small gap between feature distillation and direct training (1.51%) indicates the student architecture is already well-designed.

3. **Efficiency-performance trade-off**: The feature-distilled student offers an excellent balance between model size and performance.

4. **Distillation isn't always beneficial**: Knowledge transfer techniques must be carefully implemented and evaluated.

## 📖 References

- Hinton, G., Vinyals, O., Dean, J.: [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531) (2015)
- PyTorch Tutorial: [Knowledge Distillation](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- Romero, A., et al.: [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) (2015)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 