"""
LSTM Layer implementation using LSTM cells
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from .lstm_cell import LSTMCell
from ..utils.logger import get_logger


class LSTMLayer:
    """
    LSTM Layer that manages a sequence of LSTM cells.
    Handles forward and backward propagation through time.
    """
    
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = True, 
                 dropout_rate: float = 0.0, logger=None):
        """
        Initialize LSTM layer.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            return_sequences: Whether to return full sequence or just last output
            dropout_rate: Dropout rate for regularization
            logger: Logger instance (optional)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.logger = logger or get_logger()
        
        # Initialize LSTM cell
        self.lstm_cell = LSTMCell(input_size, hidden_size, logger)
        
        # Training mode flag
        self.training = True
        
        # Cache for sequences
        self.sequence_cache = {}
        
        self.logger.debug(f"LSTM Layer initialized: input_size={input_size}, hidden_size={hidden_size}")
    
    def forward(self, X: np.ndarray, initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Forward pass through LSTM layer.
        
        Args:
            X: Input sequences (batch_size, sequence_length, input_size)
            initial_state: Optional initial (h0, c0) state
            
        Returns:
            Output sequences or final output based on return_sequences
        """
        batch_size, sequence_length, _ = X.shape
        
        # Initialize states
        if initial_state is None:
            h_prev = np.zeros((self.hidden_size, batch_size))
            c_prev = np.zeros((self.hidden_size, batch_size))
        else:
            h_prev, c_prev = initial_state
            h_prev = h_prev.T if h_prev.shape[0] == batch_size else h_prev
            c_prev = c_prev.T if c_prev.shape[0] == batch_size else c_prev
        
        # Store states for each time step
        hidden_states = []
        cell_states = []
        
        # Forward pass through time
        for t in range(sequence_length):
            # Get input at time t (input_size, batch_size)
            x_t = X[:, t, :].T
            
            # Forward through LSTM cell
            h_current, c_current = self.lstm_cell.forward(x_t, h_prev, c_prev)
            
            # Apply dropout if in training mode
            if self.training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, h_current.shape) / (1 - self.dropout_rate)
                h_current = h_current * dropout_mask
            
            # Store states
            hidden_states.append(h_current.T)  # Convert back to (batch_size, hidden_size)
            cell_states.append(c_current.T)
            
            # Update previous states
            h_prev = h_current
            c_prev = c_current
        
        # Cache for backward pass
        self.sequence_cache = {
            'hidden_states': hidden_states,
            'cell_states': cell_states,
            'X': X,
            'initial_state': initial_state
        }
        
        # Return sequences or just final output
        if self.return_sequences:
            return np.stack(hidden_states, axis=1)  # (batch_size, sequence_length, hidden_size)
        else:
            return hidden_states[-1]  # (batch_size, hidden_size)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through LSTM layer.
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        if not self.sequence_cache:
            raise ValueError("Must call forward() before backward()")
        
        # Get cached values
        hidden_states = self.sequence_cache['hidden_states']
        cell_states = self.sequence_cache['cell_states']
        X = self.sequence_cache['X']
        
        batch_size, sequence_length, _ = X.shape
        
        # Initialize gradients
        dX = np.zeros_like(X)
        
        # If not returning sequences, expand dout to full sequence
        if not self.return_sequences:
            dh_sequence = np.zeros((batch_size, sequence_length, self.hidden_size))
            dh_sequence[:, -1, :] = dout  # Only last time step has gradient
        else:
            dh_sequence = dout
        
        # Initialize gradients for LSTM states
        dh_next = np.zeros((self.hidden_size, batch_size))
        dc_next = np.zeros((self.hidden_size, batch_size))
        
        # Backward pass through time (reverse order)
        for t in reversed(range(sequence_length)):
            # Add gradient from current time step
            dh_current = dh_sequence[:, t, :].T + dh_next
            
            # Set LSTM cell cache for this time step
            self._set_cell_cache_for_timestep(t, X, hidden_states, cell_states)
            
            # Backward through LSTM cell
            dx_t, dh_prev, dc_prev = self.lstm_cell.backward(dh_current, dc_next)
            
            # Store input gradient
            dX[:, t, :] = dx_t.T
            
            # Update gradients for next iteration
            dh_next = dh_prev
            dc_next = dc_prev
        
        return dX
    
    def _set_cell_cache_for_timestep(self, t: int, X: np.ndarray, 
                                   hidden_states: List[np.ndarray], 
                                   cell_states: List[np.ndarray]):
        """
        Set LSTM cell cache for specific timestep during backward pass.
        
        Args:
            t: Current timestep
            X: Input sequences
            hidden_states: List of hidden states
            cell_states: List of cell states
        """
        batch_size = X.shape[0]
        
        # Current input
        x_t = X[:, t, :].T
        
        # Previous states
        if t > 0:
            h_prev = hidden_states[t-1].T
            c_prev = cell_states[t-1].T
        else:
            h_prev = np.zeros((self.hidden_size, batch_size))
            c_prev = np.zeros((self.hidden_size, batch_size))
        
        # Current states
        h_current = hidden_states[t].T
        c_current = cell_states[t].T
        
        # Reconstruct intermediate values (this is simplified - in practice, 
        # we'd need to store these during forward pass for exact reconstruction)
        concat_input = np.vstack([x_t, h_prev])
        
        # Approximate gate values (this is a simplification)
        f_gate = self.lstm_cell._sigmoid(np.dot(self.lstm_cell.Wf, concat_input) + self.lstm_cell.bf)
        i_gate = self.lstm_cell._sigmoid(np.dot(self.lstm_cell.Wi, concat_input) + self.lstm_cell.bi)
        c_tilde = self.lstm_cell._tanh(np.dot(self.lstm_cell.Wc, concat_input) + self.lstm_cell.bc)
        o_gate = self.lstm_cell._sigmoid(np.dot(self.lstm_cell.Wo, concat_input) + self.lstm_cell.bo)
        
        # Set cache
        self.lstm_cell.cache = {
            'x': x_t,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'concat_input': concat_input,
            'f_gate': f_gate,
            'i_gate': i_gate,
            'c_tilde': c_tilde,
            'c_current': c_current,
            'o_gate': o_gate,
            'h_current': h_current
        }
    
    def update_parameters(self, learning_rate: float, gradient_clip: Optional[float] = None):
        """
        Update LSTM cell parameters.
        
        Args:
            learning_rate: Learning rate
            gradient_clip: Gradient clipping threshold
        """
        self.lstm_cell.update_parameters(learning_rate, gradient_clip)
    
    def set_training(self, training: bool):
        """
        Set training mode.
        
        Args:
            training: Whether in training mode
        """
        self.training = training
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.lstm_cell.get_parameters()
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        Set layer parameters.
        
        Args:
            parameters: Dictionary of parameters
        """
        self.lstm_cell.set_parameters(parameters)
    
    def reset_gradients(self):
        """Reset gradients."""
        self.lstm_cell.reset_gradients()
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, ...]:
        """
        Get output shape given input shape.
        
        Args:
            input_shape: (batch_size, sequence_length, input_size)
            
        Returns:
            Output shape
        """
        batch_size, sequence_length, _ = input_shape
        
        if self.return_sequences:
            return (batch_size, sequence_length, self.hidden_size)
        else:
            return (batch_size, self.hidden_size)
    
    def __str__(self) -> str:
        """String representation."""
        return (f"LSTMLayer(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"return_sequences={self.return_sequences}, dropout_rate={self.dropout_rate})")


class DenseLayer:
    """
    Dense (fully connected) layer for LSTM output.
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'linear', logger=None):
        """
        Initialize Dense layer.
        
        Args:
            input_size: Input feature size
            output_size: Output size
            activation: Activation function ('linear', 'relu', 'sigmoid', 'tanh')
            logger: Logger instance
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.logger = logger or get_logger()
        
        # Initialize weights and bias
        self.W = np.random.normal(0, np.sqrt(2.0 / input_size), (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self.cache = {}
        
        self.logger.debug(f"Dense layer initialized: {input_size} -> {output_size}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through dense layer.
        
        Args:
            X: Input (batch_size, input_size)
            
        Returns:
            Output (batch_size, output_size)
        """
        # Ensure correct shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Linear transformation
        Z = np.dot(X, self.W.T) + self.b.T  # (batch_size, output_size)
        
        # Apply activation
        if self.activation == 'linear':
            A = Z
        elif self.activation == 'relu':
            A = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation == 'tanh':
            A = np.tanh(Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Cache for backward pass
        self.cache = {'X': X, 'Z': Z, 'A': A}
        
        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through dense layer.
        
        Args:
            dA: Gradient w.r.t. output
            
        Returns:
            Gradient w.r.t. input
        """
        X = self.cache['X']
        Z = self.cache['Z']
        
        # Gradient w.r.t. activation
        if self.activation == 'linear':
            dZ = dA
        elif self.activation == 'relu':
            dZ = dA * (Z > 0)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            dZ = dA * sigmoid * (1 - sigmoid)
        elif self.activation == 'tanh':
            tanh = np.tanh(Z)
            dZ = dA * (1 - tanh ** 2)
        
        # Gradients w.r.t. parameters
        self.dW = np.dot(dZ.T, X)  # (output_size, input_size)
        self.db = np.sum(dZ, axis=0, keepdims=True).T  # (output_size, 1)
        
        # Gradient w.r.t. input
        dX = np.dot(dZ, self.W)  # (batch_size, input_size)
        
        return dX
    
    def update_parameters(self, learning_rate: float):
        """
        Update parameters.
        
        Args:
            learning_rate: Learning rate
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters."""
        return {'W': self.W, 'b': self.b}
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set parameters."""
        self.W = parameters['W']
        self.b = parameters['b']
    
    def __str__(self) -> str:
        """String representation."""
        return f"DenseLayer(input_size={self.input_size}, output_size={self.output_size}, activation={self.activation})"
