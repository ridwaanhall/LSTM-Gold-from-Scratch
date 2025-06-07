"""
Complete LSTM Network implementation for gold price prediction
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional

from .lstm_layer import LSTMLayer, DenseLayer
from ..utils.logger import get_logger


class LSTMNetwork:
    """
    Multi-layer LSTM network for time series prediction.
    """
    
    def __init__(self, config, logger=None):
        """
        Initialize LSTM network.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or get_logger()
        
        # Network architecture
        self.layers = []
        self.compiled = False
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Build network
        self._build_network()
        
        self.logger.info("LSTM Network initialized successfully")
    
    def _build_network(self):
        """Build the LSTM network architecture."""
        input_size = self.config.input_size
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        output_size = self.config.output_size
        dropout_rate = self.config.dropout_rate
        
        # Build LSTM layers
        for i in range(num_layers):
            if i == 0:
                # First layer
                layer_input_size = input_size
            else:
                # Subsequent layers
                layer_input_size = hidden_size
            
            # Only last layer doesn't return sequences (unless we want all outputs)
            return_sequences = i < num_layers - 1
            
            lstm_layer = LSTMLayer(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                return_sequences=return_sequences,
                dropout_rate=dropout_rate if i < num_layers - 1 else 0,  # No dropout on last layer
                logger=self.logger
            )
            
            self.layers.append(lstm_layer)
            self.logger.debug(f"Added LSTM layer {i+1}: {lstm_layer}")
        
        # Add dense output layer
        dense_layer = DenseLayer(
            input_size=hidden_size,
            output_size=output_size,
            activation='linear',  # Linear for regression
            logger=self.logger
        )
        self.layers.append(dense_layer)
        
        self.logger.info(f"Network built with {len(self.layers)} layers")
        self.compiled = True
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input sequences (batch_size, sequence_length, input_size)
            training: Whether in training mode
            
        Returns:
            Network output
        """
        if not self.compiled:
            raise ValueError("Network must be compiled before forward pass")
        
        # Set training mode for all layers
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
        
        current_input = X
        
        # Forward through all layers
        for i, layer in enumerate(self.layers):
            current_input = layer.forward(current_input)
            self.logger.debug(f"Layer {i} output shape: {current_input.shape}")
        
        return current_input
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Backward pass through the network.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        batch_size = y_true.shape[0]
        
        # Calculate loss (Mean Squared Error)
        loss = np.mean((y_pred - y_true) ** 2)
        
        # Calculate gradient of loss w.r.t. predictions
        dout = 2 * (y_pred - y_true) / batch_size  # MSE gradient
        
        # Backward through layers in reverse order
        current_grad = dout
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        
        return loss
    
    def update_parameters(self, learning_rate: float):
        """
        Update all network parameters.
        
        Args:
            learning_rate: Learning rate
        """
        for layer in self.layers:
            if hasattr(layer, 'update_parameters'):
                if isinstance(layer, LSTMLayer):
                    layer.update_parameters(learning_rate, self.config.gradient_clip)
                else:
                    layer.update_parameters(learning_rate)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        return self.forward(X, training=False)
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> float:
        """
        Single training step.
        
        Args:
            X_batch: Input batch
            y_batch: Target batch
            learning_rate: Learning rate
            
        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(X_batch, training=True)
        
        # Backward pass
        loss = self.backward(y_batch, y_pred)
        
        # Update parameters
        self.update_parameters(learning_rate)
        
        return loss
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Input data
            y: True values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = np.mean((y_pred - y) ** 2)
        mae = np.mean(np.abs(y_pred - y))
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model parameters.
        
        Args:
            filepath: Path to save model
        """
        try:
            model_data = {
                'config': self.config.to_dict(),
                'layers': [],
                'history': self.history
            }
            
            # Save layer parameters
            for layer in self.layers:
                layer_data = {
                    'type': type(layer).__name__,
                    'parameters': layer.get_parameters()
                }
                if hasattr(layer, 'input_size'):
                    layer_data['input_size'] = layer.input_size
                if hasattr(layer, 'hidden_size'):
                    layer_data['hidden_size'] = layer.hidden_size
                if hasattr(layer, 'output_size'):
                    layer_data['output_size'] = layer.output_size
                if hasattr(layer, 'return_sequences'):
                    layer_data['return_sequences'] = layer.return_sequences
                if hasattr(layer, 'dropout_rate'):
                    layer_data['dropout_rate'] = layer.dropout_rate
                if hasattr(layer, 'activation'):
                    layer_data['activation'] = layer.activation
                
                model_data['layers'].append(layer_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load model parameters.
        
        Args:
            filepath: Path to load model from
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore history
            self.history = model_data.get('history', {'train_loss': [], 'val_loss': [], 'learning_rate': []})
            
            # Restore layer parameters
            for i, (layer, layer_data) in enumerate(zip(self.layers, model_data['layers'])):
                layer.set_parameters(layer_data['parameters'])
                self.logger.debug(f"Loaded parameters for layer {i}: {layer_data['type']}")
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_layer_info(self) -> List[Dict]:
        """
        Get information about all layers.
        
        Returns:
            List of layer information
        """
        info = []
        for i, layer in enumerate(self.layers):
            layer_info = {
                'index': i,
                'type': type(layer).__name__,
                'description': str(layer)
            }
            
            # Add layer-specific info
            if hasattr(layer, 'input_size'):
                layer_info['input_size'] = layer.input_size
            if hasattr(layer, 'hidden_size'):
                layer_info['hidden_size'] = layer.hidden_size
            if hasattr(layer, 'output_size'):
                layer_info['output_size'] = layer.output_size
            
            info.append(layer_info)
        
        return info
    
    def summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Model summary string
        """
        summary_str = "LSTM Network Summary\n"
        summary_str += "=" * 50 + "\n"
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = 0
            if hasattr(layer, 'get_parameters'):
                params = layer.get_parameters()
                for param_name, param_array in params.items():
                    layer_params += param_array.size
            
            total_params += layer_params
            
            summary_str += f"Layer {i}: {str(layer)}\n"
            summary_str += f"  Parameters: {layer_params:,}\n"
            summary_str += "-" * 50 + "\n"
        
        summary_str += f"Total Parameters: {total_params:,}\n"
        summary_str += "=" * 50
        
        return summary_str
    
    def reset_history(self):
        """Reset training history."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def add_history(self, train_loss: float, val_loss: Optional[float] = None, 
                   learning_rate: Optional[float] = None):
        """
        Add values to training history.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss (optional)
            learning_rate: Learning rate (optional)
        """
        self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
    
    def __str__(self) -> str:
        """String representation."""
        return f"LSTMNetwork(layers={len(self.layers)}, compiled={self.compiled})"
