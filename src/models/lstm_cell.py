"""
LSTM Cell implementation from scratch using NumPy
"""

import numpy as np
from typing import Tuple, Dict, Optional

from ..utils.logger import get_logger


class LSTMCell:
    """
    LSTM (Long Short-Term Memory) cell implementation from scratch.
    
    This class implements a single LSTM cell with forget gate, input gate,
    output gate, and cell state following the standard LSTM architecture.
    """
    
    def __init__(self, input_size: int, hidden_size: int, logger=None):
        """
        Initialize LSTM cell with random weights.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            logger: Logger instance (optional)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.logger = logger or get_logger()
        
        # Initialize weights and biases
        self._initialize_parameters()
        
        # Cache for gradients
        self.gradients = {}
        
        # Cache for forward pass (needed for backprop)
        self.cache = {}
    
    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization."""
        # Xavier initialization scale
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # Forget gate parameters
        self.Wf = np.random.normal(0, scale, (self.hidden_size, self.input_size + self.hidden_size))
        self.bf = np.zeros((self.hidden_size, 1))
        
        # Input gate parameters
        self.Wi = np.random.normal(0, scale, (self.hidden_size, self.input_size + self.hidden_size))
        self.bi = np.zeros((self.hidden_size, 1))
        
        # Candidate values parameters
        self.Wc = np.random.normal(0, scale, (self.hidden_size, self.input_size + self.hidden_size))
        self.bc = np.zeros((self.hidden_size, 1))
        
        # Output gate parameters
        self.Wo = np.random.normal(0, scale, (self.hidden_size, self.input_size + self.hidden_size))
        self.bo = np.zeros((self.hidden_size, 1))
        
        self.logger.debug(f"LSTM cell initialized with input_size={self.input_size}, hidden_size={self.hidden_size}")
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input at current time step (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            c_prev: Previous cell state (hidden_size, batch_size)
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        # Ensure inputs are 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)
        
        # Concatenate input and previous hidden state
        concat_input = np.vstack([x, h_prev])  # (input_size + hidden_size, batch_size)
        
        # Forget gate
        f_gate = self._sigmoid(np.dot(self.Wf, concat_input) + self.bf)
        
        # Input gate
        i_gate = self._sigmoid(np.dot(self.Wi, concat_input) + self.bi)
        
        # Candidate values
        c_tilde = self._tanh(np.dot(self.Wc, concat_input) + self.bc)
        
        # Update cell state
        c_current = f_gate * c_prev + i_gate * c_tilde
        
        # Output gate
        o_gate = self._sigmoid(np.dot(self.Wo, concat_input) + self.bo)
        
        # Update hidden state
        h_current = o_gate * self._tanh(c_current)
        
        # Cache values for backward pass
        self.cache = {
            'x': x,
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
        
        return h_current, c_current
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through LSTM cell.
        
        Args:
            dh_next: Gradient w.r.t. hidden state from next time step
            dc_next: Gradient w.r.t. cell state from next time step
            
        Returns:
            Tuple of (dx, dh_prev, dc_prev)
        """
        # Get cached values
        cache = self.cache
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        concat_input = cache['concat_input']
        f_gate = cache['f_gate']
        i_gate = cache['i_gate']
        c_tilde = cache['c_tilde']
        c_current = cache['c_current']
        o_gate = cache['o_gate']
        
        # Gradients w.r.t. output gate
        do_gate = dh_next * self._tanh(c_current)
        
        # Gradients w.r.t. cell state
        dc_current = dc_next + dh_next * o_gate * self._tanh_derivative(c_current)
        
        # Gradients w.r.t. forget gate
        df_gate = dc_current * c_prev
        
        # Gradients w.r.t. input gate
        di_gate = dc_current * c_tilde
        
        # Gradients w.r.t. candidate values
        dc_tilde = dc_current * i_gate
        
        # Gradients w.r.t. previous cell state
        dc_prev = dc_current * f_gate
        
        # Gradients w.r.t. gates before activation
        df_gate_raw = df_gate * self._sigmoid_derivative(f_gate)
        di_gate_raw = di_gate * self._sigmoid_derivative(i_gate)
        dc_tilde_raw = dc_tilde * self._tanh_derivative(c_tilde)
        do_gate_raw = do_gate * self._sigmoid_derivative(o_gate)
        
        # Gradients w.r.t. weights and biases
        self.gradients['dWf'] = np.dot(df_gate_raw, concat_input.T)
        self.gradients['dbf'] = np.sum(df_gate_raw, axis=1, keepdims=True)
        
        self.gradients['dWi'] = np.dot(di_gate_raw, concat_input.T)
        self.gradients['dbi'] = np.sum(di_gate_raw, axis=1, keepdims=True)
        
        self.gradients['dWc'] = np.dot(dc_tilde_raw, concat_input.T)
        self.gradients['dbc'] = np.sum(dc_tilde_raw, axis=1, keepdims=True)
        
        self.gradients['dWo'] = np.dot(do_gate_raw, concat_input.T)
        self.gradients['dbo'] = np.sum(do_gate_raw, axis=1, keepdims=True)
        
        # Gradients w.r.t. concatenated input
        dconcat_input = (np.dot(self.Wf.T, df_gate_raw) + 
                        np.dot(self.Wi.T, di_gate_raw) + 
                        np.dot(self.Wc.T, dc_tilde_raw) + 
                        np.dot(self.Wo.T, do_gate_raw))
        
        # Split gradients for input and previous hidden state
        dx = dconcat_input[:self.input_size]
        dh_prev = dconcat_input[self.input_size:]
        
        return dx, dh_prev, dc_prev
    
    def update_parameters(self, learning_rate: float, gradient_clip: Optional[float] = None):
        """
        Update parameters using gradients.
        
        Args:
            learning_rate: Learning rate for parameter updates
            gradient_clip: Gradient clipping threshold (optional)
        """
        # Apply gradient clipping if specified
        if gradient_clip is not None:
            self._clip_gradients(gradient_clip)
        
        # Update weights and biases
        self.Wf -= learning_rate * self.gradients['dWf']
        self.bf -= learning_rate * self.gradients['dbf']
        
        self.Wi -= learning_rate * self.gradients['dWi']
        self.bi -= learning_rate * self.gradients['dbi']
        
        self.Wc -= learning_rate * self.gradients['dWc']
        self.bc -= learning_rate * self.gradients['dbc']
        
        self.Wo -= learning_rate * self.gradients['dWo']
        self.bo -= learning_rate * self.gradients['dbo']
    
    def _clip_gradients(self, threshold: float):
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            threshold: Clipping threshold
        """
        for key in self.gradients:
            self.gradients[key] = np.clip(self.gradients[key], -threshold, threshold)
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get all parameters as a dictionary.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'Wf': self.Wf, 'bf': self.bf,
            'Wi': self.Wi, 'bi': self.bi,
            'Wc': self.Wc, 'bc': self.bc,
            'Wo': self.Wo, 'bo': self.bo
        }
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        Set parameters from dictionary.
        
        Args:
            parameters: Dictionary of parameters
        """
        self.Wf = parameters['Wf']
        self.bf = parameters['bf']
        self.Wi = parameters['Wi']
        self.bi = parameters['bi']
        self.Wc = parameters['Wc']
        self.bc = parameters['bc']
        self.Wo = parameters['Wo']
        self.bo = parameters['bo']
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid of input
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid function.
        
        Args:
            sigmoid_output: Output of sigmoid function
            
        Returns:
            Derivative of sigmoid
        """
        return sigmoid_output * (1 - sigmoid_output)
    
    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent activation function.
        
        Args:
            x: Input array
            
        Returns:
            Tanh of input
        """
        return np.tanh(x)
    
    @staticmethod
    def _tanh_derivative(tanh_output: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh function.
        
        Args:
            tanh_output: Output of tanh function
            
        Returns:
            Derivative of tanh
        """
        return 1 - tanh_output ** 2
    
    def reset_gradients(self):
        """Reset gradients to zero."""
        self.gradients = {}
    
    def __str__(self) -> str:
        """String representation of LSTM cell."""
        return f"LSTMCell(input_size={self.input_size}, hidden_size={self.hidden_size})"
