from distil_pytorch.models.cnn import CIFAR10CNN, create_teacher_model, create_student_model
from distil_pytorch.utils.distillation import (
    DistillationLoss,
    FeatureDistillationLoss,
    train_teacher,
    generate_teacher_predictions,
    train_student_with_distillation as train_student_with_teacher,
    train_student_with_teacher_model,
    train_student_with_feature_distillation,
    train_student_standard as train_model_directly,
    train_student_standard
)
from distil_pytorch.utils.data import load_cifar10, visualize_samples, count_parameters
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Tuple

# Add evaluation function
def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: str = 'cuda') -> float:
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        Accuracy of the model on the test dataset (between 0 and 1)
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total

__version__ = '0.1.0'
