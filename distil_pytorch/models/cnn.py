import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    CNN model for CIFAR-10 image classification, configurable by scale parameter.
    Similar architecture to the original TensorFlow model from the notebook.
    
    Notes:
        - The forward method returns raw logits (not softmaxed)
        - The get_features method returns the penultimate layer's activations
    """
    def __init__(self, scale: int = 32, dropout_rate: float = 0.5) -> None:
        """
        Initialize the CNN model.
        
        Args:
            scale: Scale factor for the number of filters, controls model capacity.
                   Teacher typically uses higher scale than student.
            dropout_rate: Dropout probability for regularization.
        """
        super(CIFAR10CNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, scale, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(scale, scale, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block
        self.conv3 = nn.Conv2d(scale, 2*scale, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*scale, 2*scale, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2*scale*8*8, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
            
        Returns:
            Tensor of shape [batch_size, 10] with class logits (not softmaxed).
            For training with distillation, we need raw logits, not probabilities.
        """
        # First block
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool1(x)
        
        # Second block
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.pool2(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return raw logits instead of log_softmax for distillation
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the feature representation before the final classification layer.
        Can be used for feature-based distillation.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
            
        Returns:
            Feature tensor from the penultimate layer of shape [batch_size, 512]
        """
        # First block
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool1(x)
        
        # Second block
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.pool2(x)
        
        # Feature representation
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        
        return x


def create_teacher_model() -> CIFAR10CNN:
    """Create a teacher model with higher capacity"""
    return CIFAR10CNN(scale=32)


def create_student_model() -> CIFAR10CNN:
    """Create a student model with lower capacity"""
    return CIFAR10CNN(scale=16) 