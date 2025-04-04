import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, List, Any, Optional, Union


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss: combines cross-entropy on ground truth labels 
    and KL divergence between student and teacher predictions.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0) -> None:
        """
        Initialize the distillation loss.
        
        Args:
            alpha: Weight for hard labels loss. (1-alpha) will be used for soft labels.
                   Default increased to 0.5 for better stability.
            temperature: Temperature for softening probability distributions.
                        Higher values produce softer distributions.
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, student_outputs: torch.Tensor, 
                teacher_outputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distillation loss.
        
        Args:
            student_outputs: Logits from the student model
            teacher_outputs: Logits from the teacher model
            targets: Ground truth hard labels
            
        Returns:
            Weighted loss combining hard and soft targets
        """
        # Hard targets loss with standard cross-entropy
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Soft targets loss with KL divergence
        # Apply temperature scaling
        student_temp = student_outputs / self.temperature
        teacher_temp = teacher_outputs / self.temperature
        
        # Convert to log probabilities and probabilities
        soft_student = F.log_softmax(student_temp, dim=1)
        soft_teacher = F.softmax(teacher_temp, dim=1)
        
        # KL divergence loss
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        distillation_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return distillation_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based Knowledge Distillation Loss: combines cross-entropy on ground truth labels,
    KL divergence between logits, and L2 loss between feature maps.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.4, temperature: float = 3.0) -> None:
        """
        Initialize the feature distillation loss.
        
        Args:
            alpha: Weight for hard labels loss. Increased to 0.5 for stability.
            beta: Weight for feature matching loss. Reduced to 0.4 to balance.
            temperature: Temperature for softening probability distributions.
        """
        super(FeatureDistillationLoss, self).__init__()
        self.alpha = alpha  # weight for hard labels (ground truth)
        self.beta = beta    # weight for feature matching
        self.temperature = temperature
    
    def forward(self, student_outputs: torch.Tensor, 
                teacher_outputs: torch.Tensor, 
                student_features: torch.Tensor, 
                teacher_features: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined distillation loss.
        
        Args:
            student_outputs: Logits from the student model
            teacher_outputs: Logits from the teacher model
            student_features: Features from the student model
            teacher_features: Features from the teacher model
            targets: Ground truth hard labels
            
        Returns:
            Combined loss of hard labels, soft labels, and feature matching
        """
        # Check feature dimensions match
        if student_features.shape != teacher_features.shape:
            raise ValueError(f"Feature shapes don't match: student {student_features.shape} vs teacher {teacher_features.shape}")
            
        # Hard targets loss with standard cross-entropy
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Soft targets loss with KL divergence
        # Apply temperature scaling
        student_temp = student_outputs / self.temperature
        teacher_temp = teacher_outputs / self.temperature
        
        # Convert to log probabilities and probabilities
        soft_student = F.log_softmax(student_temp, dim=1)
        soft_teacher = F.softmax(teacher_temp, dim=1)
        
        # KL divergence loss
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature matching loss (L2 distance)
        feature_loss = F.mse_loss(student_features, teacher_features)
        
        # Combine all losses
        total_loss = (self.alpha * hard_loss + 
                     (1 - self.alpha - self.beta) * soft_loss + 
                     self.beta * feature_loss)
        
        return total_loss


def train_teacher(model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 epochs: int = 25, 
                 device: str = 'cuda', 
                 lr: float = 0.001) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the teacher model using standard cross-entropy loss.
    
    Args:
        model: Teacher model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        
    Returns:
        Tuple containing:
            - Trained teacher model
            - Training history dictionary with keys: 
              'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    
    return model, history


def generate_teacher_predictions(teacher_model: nn.Module, 
                               data_loader: DataLoader, 
                               device: str = 'cuda') -> torch.Tensor:
    """
    Generate soft targets from the teacher model.
    
    Args:
        teacher_model: Trained teacher model
        data_loader: DataLoader for the dataset
        device: Device to use for inference
        
    Returns:
        Tensor of teacher predictions for all samples in the data_loader
    """
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    teacher_preds = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = teacher_model(inputs)
            teacher_preds.append(outputs.cpu())
    
    return torch.cat(teacher_preds, dim=0)


def train_student_with_distillation(student_model: nn.Module,
                                  teacher_outputs: torch.Tensor,
                                  train_loader: DataLoader,
                                  val_loader: DataLoader, 
                                  epochs: int = 10,
                                  alpha: float = 0.5,
                                  temperature: float = 3.0,
                                  device: str = 'cuda',
                                  lr: float = 0.001) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the student model using knowledge distillation.
    
    Args:
        student_model: Student model to train
        teacher_outputs: Pre-computed teacher predictions
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        alpha: Weight for hard labels (increased to 0.5 for stability)
        temperature: Temperature for softening distributions
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        
    Returns:
        Tuple containing:
            - Trained student model
            - Training history dictionary with keys: 
              'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    student_model = student_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    distillation_criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    standard_criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Get all teacher outputs as a tensor
    all_teacher_outputs = teacher_outputs.to(device)
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get corresponding teacher outputs for this batch
            batch_size = inputs.size(0)
            idx_start = i * batch_size
            idx_end = min((i + 1) * batch_size, len(all_teacher_outputs))
            teacher_batch_outputs = all_teacher_outputs[idx_start:idx_end]
            
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            
            # Calculate distillation loss
            loss = distillation_criterion(student_outputs, teacher_batch_outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = student_model(inputs)
                loss = standard_criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    
    return student_model, history


def train_student_with_teacher_model(student_model: nn.Module,
                                   teacher_model: nn.Module,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader, 
                                   epochs: int = 10,
                                   alpha: float = 0.5,
                                   temperature: float = 3.0,
                                   device: str = 'cuda',
                                   lr: float = 0.001) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the student model directly using a teacher model in one step.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model to provide knowledge
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        alpha: Weight for hard labels
        temperature: Temperature for softening distributions
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        
    Returns:
        Tuple containing:
            - Trained student model
            - Training history dictionary with keys: 
              'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    print("Generating teacher predictions...")
    teacher_outputs = generate_teacher_predictions(teacher_model, train_loader, device)
    print(f"Generated predictions for {len(teacher_outputs)} samples")
    
    return train_student_with_distillation(
        student_model, 
        teacher_outputs, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        alpha=alpha, 
        temperature=temperature, 
        device=device,
        lr=lr
    )


def train_student_with_feature_distillation(student_model: nn.Module,
                                         teacher_model: nn.Module,
                                         train_loader: DataLoader,
                                         val_loader: DataLoader, 
                                         epochs: int = 10,
                                         alpha: float = 0.5,
                                         beta: float = 0.4,
                                         temperature: float = 3.0,
                                         device: str = 'cuda',
                                         lr: float = 0.001) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the student model using feature-based knowledge distillation.
    
    Args:
        student_model: Student model to train
        teacher_model: Pre-trained teacher model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        alpha: Weight for hard labels (ground truth), increased to 0.5
        beta: Weight for feature matching, reduced to 0.4
        temperature: Temperature for softening distributions
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        
    Returns:
        Tuple containing:
            - Trained student model
            - Training history dictionary with keys: 
              'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Teacher model in evaluation mode
    
    # Check feature dimensions match using a sample batch
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch[:1].to(device)  # Use first sample only
    with torch.no_grad():
        student_features = student_model.get_features(sample_batch)
        teacher_features = teacher_model.get_features(sample_batch)
        if student_features.shape != teacher_features.shape:
            print(f"Warning: Feature shapes don't match: student {student_features.shape} vs teacher {teacher_features.shape}")
            print("This may cause issues during feature distillation")
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    distillation_criterion = FeatureDistillationLoss(alpha=alpha, beta=beta, temperature=temperature)
    standard_criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Get student outputs and features
            student_outputs = student_model(inputs)
            student_features = student_model.get_features(inputs)
            
            # Get teacher outputs and features (no gradient tracking)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_features = teacher_model.get_features(inputs)
            
            # Calculate distillation loss
            loss = distillation_criterion(
                student_outputs, teacher_outputs, 
                student_features, teacher_features, 
                targets
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = student_model(inputs)
                loss = standard_criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    
    return student_model, history


def train_student_standard(student_model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader, 
                         epochs: int = 10,
                         device: str = 'cuda',
                         lr: float = 0.001) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the student model using standard cross-entropy loss without distillation.
    
    Args:
        student_model: Student model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        lr: Learning rate
        
    Returns:
        Tuple containing:
            - Trained student model
            - Training history dictionary with keys: 
              'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    student_model = student_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = student_model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    
    return student_model, history 