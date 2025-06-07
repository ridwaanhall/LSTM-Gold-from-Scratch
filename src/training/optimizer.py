"""
Optimization algorithms for LSTM training
"""

import numpy as np
from typing import Dict, List, Optional

from ..utils.logger import get_logger


class AdamOptimizer:
    """
    Adam optimizer implementation for LSTM parameter updates.
    
    Adam combines the advantages of AdaGrad and RMSProp optimizers.
    It maintains moving averages of both gradients and squared gradients.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, logger=None):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small constant for numerical stability
            logger: Logger instance
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.logger = logger or get_logger()
        
        # Moving averages
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        
        # Time step
        self.t = 0
        
        self.logger.debug(f"Adam optimizer initialized: lr={learning_rate}, beta1={beta1}, beta2={beta2}")
    
    def update(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update parameters using Adam algorithm.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            
        Returns:
            Updated parameters
        """
        self.t += 1
        updated_params = {}
        
        for param_name, param_value in parameters.items():
            if param_name not in gradients:
                updated_params[param_name] = param_value
                continue
                
            grad = gradients[param_name]
            
            # Initialize moving averages if first time
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param_value)
                self.v[param_name] = np.zeros_like(param_value)
            
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[param_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[param_name] = param_value - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        return updated_params
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate
    
    def set_lr(self, learning_rate: float):
        """Set learning rate."""
        self.learning_rate = learning_rate
    
    def reset(self):
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.t = 0


class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer with momentum.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, logger=None):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            logger: Logger instance
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.logger = logger or get_logger()
        
        # Velocity for momentum
        self.velocity = {}
        
        self.logger.debug(f"SGD optimizer initialized: lr={learning_rate}, momentum={momentum}")
    
    def update(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update parameters using SGD with momentum.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            
        Returns:
            Updated parameters
        """
        updated_params = {}
        
        for param_name, param_value in parameters.items():
            if param_name not in gradients:
                updated_params[param_name] = param_value
                continue
                
            grad = gradients[param_name]
            
            # Initialize velocity if first time
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param_value)
            
            # Update velocity
            self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.learning_rate * grad
            
            # Update parameters
            updated_params[param_name] = param_value + self.velocity[param_name]
        
        return updated_params
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate
    
    def set_lr(self, learning_rate: float):
        """Set learning rate."""
        self.learning_rate = learning_rate
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = {}


class LearningRateScheduler:
    """
    Learning rate scheduling strategies.
    """
    
    def __init__(self, scheduler_type: str = 'exponential', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('exponential', 'step', 'cosine', 'plateau')
            **kwargs: Scheduler-specific parameters
        """
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        self.initial_lr = None
        self.current_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def step(self, optimizer, epoch: int, loss: Optional[float] = None) -> float:
        """
        Update learning rate based on scheduler.
        
        Args:
            optimizer: Optimizer instance
            epoch: Current epoch
            loss: Current validation loss (for plateau scheduler)
            
        Returns:
            New learning rate
        """
        if self.initial_lr is None:
            self.initial_lr = optimizer.get_lr()
        
        if self.scheduler_type == 'exponential':
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            new_lr = self.initial_lr * (decay_rate ** epoch)
            
        elif self.scheduler_type == 'step':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            new_lr = self.initial_lr * (gamma ** (epoch // step_size))
            
        elif self.scheduler_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            new_lr = eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
            
        elif self.scheduler_type == 'plateau':
            factor = self.kwargs.get('factor', 0.5)
            patience = self.kwargs.get('patience', 10)
            threshold = self.kwargs.get('threshold', 1e-4)
            
            if loss is None:
                return optimizer.get_lr()
            
            if loss < self.best_loss - threshold:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                new_lr = optimizer.get_lr() * factor
                self.patience_counter = 0
            else:
                new_lr = optimizer.get_lr()
                
        else:
            new_lr = optimizer.get_lr()
        
        optimizer.set_lr(new_lr)
        return new_lr


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model instance
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights:
                # Save current best weights (simplified - in practice would need deep copy)
                self.best_weights = model.get_layer_info()
                
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped_epoch = self.counter
            return True
        
        return False
    
    def restore_weights(self, model):
        """Restore best weights to model."""
        if self.best_weights is not None and self.restore_best_weights:
            # This is a simplified version - in practice would need proper weight restoration
            pass


class GradientClipper:
    """
    Gradient clipping utilities.
    """
    
    @staticmethod
    def clip_by_norm(gradients: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum allowed norm
            
        Returns:
            Clipped gradients
        """
        # Calculate global norm
        total_norm = 0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            clipped_gradients = {}
            for name, grad in gradients.items():
                clipped_gradients[name] = grad * clip_coef
            return clipped_gradients
        
        return gradients
    
    @staticmethod
    def clip_by_value(gradients: Dict[str, np.ndarray], min_value: float, max_value: float) -> Dict[str, np.ndarray]:
        """
        Clip gradients by value.
        
        Args:
            gradients: Dictionary of gradients
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Clipped gradients
        """
        clipped_gradients = {}
        for name, grad in gradients.items():
            clipped_gradients[name] = np.clip(grad, min_value, max_value)
        return clipped_gradients


def create_optimizer(optimizer_type: str, config, logger=None):
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: Type of optimizer ('adam', 'sgd')
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return AdamOptimizer(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            logger=logger
        )
    elif optimizer_type.lower() == 'sgd':
        return SGDOptimizer(
            learning_rate=config.learning_rate,
            momentum=config.beta1,  # Use beta1 as momentum
            logger=logger
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(scheduler_type: str, **kwargs):
    """
    Factory function to create learning rate schedulers.
    
    Args:
        scheduler_type: Type of scheduler
        **kwargs: Scheduler parameters
        
    Returns:
        Scheduler instance
    """
    return LearningRateScheduler(scheduler_type, **kwargs)
