"""
Training module for LSTM Gold Price Prediction
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List
import os

from ..models.lstm_network import LSTMNetwork
from .optimizer import create_optimizer, create_scheduler, EarlyStopping, GradientClipper
from ..utils.logger import get_logger


class LSTMTrainer:
    """
    Professional trainer for LSTM gold price prediction model.
    Handles training loop, validation, checkpointing, and monitoring.
    """
    
    def __init__(self, config, logger=None):
        """
        Initialize LSTM trainer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or get_logger()
        
        # Initialize model
        self.model = LSTMNetwork(config, logger)
        
        # Initialize optimizer
        self.optimizer = create_optimizer('adam', config, logger)
        
        # Initialize learning rate scheduler
        self.scheduler = create_scheduler('exponential', decay_rate=0.95)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=1e-6,
            restore_best_weights=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_time = 0
        
        self.logger.info("LSTM Trainer initialized successfully")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction of data for validation
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        n_samples = X.shape[0]
        n_train = int(n_samples * (1 - validation_split))
        
        # Split data (time series - no shuffling)
        X_train = X[:n_train]
        X_val = X[n_train:]
        y_train = y[:n_train]
        y_val = y[n_train:]
        
        self.logger.log_data_info(
            X.shape,
            len(X_train),
            len(X_val)
        )
        
        return X_train, X_val, y_train, y_val
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches for training.
        
        Args:
            X: Input data
            y: Target data
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            batches.append((X_batch, y_batch))
        
        return batches
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train for one epoch.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            
        Returns:
            Average training loss for the epoch
        """
        # Create batches
        batches = self.create_batches(X_train, y_train, self.config.batch_size, shuffle=True)
        
        epoch_losses = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            # Ensure correct shapes
            if y_batch.ndim == 1:
                y_batch = y_batch.reshape(-1, 1)
            
            # Forward pass
            y_pred = self.model.forward(X_batch, training=True)
            
            # Calculate loss
            loss = np.mean((y_pred - y_batch) ** 2)
            
            # Backward pass
            batch_size = X_batch.shape[0]
            dout = 2 * (y_pred - y_batch) / batch_size
            
            # Collect gradients from all layers
            all_gradients = {}
            current_grad = dout
            
            for layer_idx, layer in enumerate(reversed(self.model.layers)):
                current_grad = layer.backward(current_grad)
                
                # Collect gradients
                if hasattr(layer, 'lstm_cell') and hasattr(layer.lstm_cell, 'gradients'):
                    for grad_name, grad_value in layer.lstm_cell.gradients.items():
                        all_gradients[f'layer_{len(self.model.layers)-1-layer_idx}_{grad_name}'] = grad_value
                elif hasattr(layer, 'dW'):
                    all_gradients[f'layer_{len(self.model.layers)-1-layer_idx}_dW'] = layer.dW
                    all_gradients[f'layer_{len(self.model.layers)-1-layer_idx}_db'] = layer.db
            
            # Apply gradient clipping
            if self.config.gradient_clip > 0:
                all_gradients = GradientClipper.clip_by_norm(all_gradients, self.config.gradient_clip)
            
            # Update parameters using optimizer
            for layer_idx, layer in enumerate(self.model.layers):
                # Extract gradients for this layer
                layer_gradients = {}
                if hasattr(layer, 'lstm_cell'):
                    for grad_name in ['dWf', 'dbf', 'dWi', 'dbi', 'dWc', 'dbc', 'dWo', 'dbo']:
                        key = f'layer_{layer_idx}_{grad_name}'
                        if key in all_gradients:
                            layer_gradients[grad_name] = all_gradients[key]
                    
                    # Get current parameters
                    current_params = layer.lstm_cell.get_parameters()
                    
                    # Update parameters
                    updated_params = self.optimizer.update(current_params, layer_gradients)
                    layer.lstm_cell.set_parameters(updated_params)
                
                elif hasattr(layer, 'W'):
                    layer_gradients = {
                        'W': all_gradients.get(f'layer_{layer_idx}_dW', np.zeros_like(layer.W)),
                        'b': all_gradients.get(f'layer_{layer_idx}_db', np.zeros_like(layer.b))
                    }
                    
                    current_params = layer.get_parameters()
                    updated_params = self.optimizer.update(current_params, layer_gradients)
                    layer.set_parameters(updated_params)
            
            epoch_losses.append(loss)
        
        return np.mean(epoch_losses)
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Validate the model.
        
        Args:
            X_val: Validation input data
            y_val: Validation target data
            
        Returns:
            Validation loss
        """
        # Ensure correct shapes
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        # Forward pass (no training)
        y_pred = self.model.predict(X_val)
        
        # Calculate loss
        val_loss = np.mean((y_pred - y_val) ** 2)
        
        return val_loss
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Complete training loop.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        self.logger.log_training_start(
            self.config.epochs,
            self.config.batch_size,
            self.config.learning_rate
        )
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y, validation_split)
        
        # Reset model history
        self.model.reset_history()
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training step
                train_loss = self.train_epoch(X_train, y_train)
                
                # Validation step
                val_loss = self.validate(X_val, y_val) if len(X_val) > 0 else None
                
                # Update learning rate
                current_lr = self.scheduler.step(self.optimizer, epoch, val_loss)
                
                # Log progress
                self.logger.log_epoch(epoch + 1, train_loss, val_loss, current_lr)
                
                # Add to history
                self.model.add_history(train_loss, val_loss, current_lr)
                
                # Save best model
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.config.save_model:
                        model_path = os.path.join(self.config.models_dir, f'best_model_epoch_{epoch+1}.pkl')
                        self.model.save_model(model_path)
                        self.logger.log_model_save(model_path, epoch + 1)
                
                # Periodic saving
                if self.config.save_model and (epoch + 1) % self.config.save_frequency == 0:
                    model_path = os.path.join(self.config.models_dir, f'model_epoch_{epoch+1}.pkl')
                    self.model.save_model(model_path)
                    self.logger.log_model_save(model_path, epoch + 1)
                
                # Early stopping check
                if val_loss is not None and self.early_stopping(val_loss, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Log epoch time
                epoch_time = time.time() - epoch_start_time
                self.logger.debug(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
        except Exception as e:
            self.logger.log_error_with_traceback("Training failed", e)
            raise
        
        finally:
            # Final model save
            if self.config.save_model:
                final_model_path = os.path.join(self.config.models_dir, self.config.model_filename)
                self.model.save_model(final_model_path)
                self.logger.log_model_save(final_model_path, self.current_epoch + 1)
        
        self.training_time = time.time() - start_time
        self.logger.info(f"Training completed in {self.training_time:.2f}s")
        
        return self.model.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = self.model.evaluate(X_test, y_test)
        self.logger.log_metrics(metrics)
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        self.model.load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_training_summary(self) -> Dict:
        """
        Get training summary statistics.
        
        Returns:
            Training summary
        """
        history = self.model.history
        
        summary = {
            'total_epochs': len(history['train_loss']),
            'best_train_loss': min(history['train_loss']) if history['train_loss'] else None,
            'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'training_time': self.training_time,
            'epochs_per_second': len(history['train_loss']) / self.training_time if self.training_time > 0 else 0
        }
        
        return summary
    
    def __str__(self) -> str:
        """String representation."""
        return f"LSTMTrainer(model={self.model}, optimizer={type(self.optimizer).__name__})"
