"""
Configuration management for LSTM Gold Price Prediction
"""

import os
from typing import Dict, Any


class Config:
    """
    Configuration class to manage all hyperparameters and settings
    for the LSTM gold price prediction model.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with default values and optional overrides.
        
        Args:
            **kwargs: Optional configuration overrides
        """
        # Data Configuration
        self.api_url = "https://sahabat.pegadaian.co.id/gold/prices/chart?interval=3650&isRequest=true"
        self.sequence_length = kwargs.get('sequence_length', 60)  # Number of days to look back
        self.prediction_days = kwargs.get('prediction_days', 30)  # Days to predict ahead
        self.train_test_split = kwargs.get('train_test_split', 0.8)  # 80% train, 20% test
        
        # Model Architecture
        self.input_size = kwargs.get('input_size', 4)  # hargaJual, hargaBeli, moving_avg, volatility
        self.hidden_size = kwargs.get('hidden_size', 128)  # LSTM hidden units
        self.num_layers = kwargs.get('num_layers', 2)  # Number of LSTM layers
        self.output_size = kwargs.get('output_size', 1)  # Predict single price
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)  # Dropout for regularization
        
        # Training Configuration
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 10)  # Early stopping patience
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)  # Gradient clipping threshold
        
        # Optimizer Configuration
        self.beta1 = kwargs.get('beta1', 0.9)  # Adam optimizer parameter
        self.beta2 = kwargs.get('beta2', 0.999)  # Adam optimizer parameter
        self.epsilon = kwargs.get('epsilon', 1e-8)  # Adam optimizer parameter
        
        # Data Preprocessing
        self.normalize_data = kwargs.get('normalize_data', True)
        self.add_technical_indicators = kwargs.get('add_technical_indicators', True)
        self.moving_average_window = kwargs.get('moving_average_window', 10)
        self.volatility_window = kwargs.get('volatility_window', 20)
        
        # File Paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, 'data')
        self.models_dir = os.path.join(self.project_root, 'models')
        self.logs_dir = os.path.join(self.project_root, 'logs')
        
        # Ensure directories exist
        self._create_directories()
        
        # Logging Configuration
        self.log_level = kwargs.get('log_level', 'INFO')
        self.log_file = os.path.join(self.logs_dir, 'lstm_gold_prediction.log')
        
        # Model Saving
        self.save_model = kwargs.get('save_model', True)
        self.model_filename = kwargs.get('model_filename', 'lstm_gold_model.pkl')
        self.save_frequency = kwargs.get('save_frequency', 10)  # Save every N epochs
        
        # Validation
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._validate_config()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "LSTM Gold Price Prediction Configuration:\n"
        config_str += "=" * 50 + "\n"
        
        sections = {
            "Data Configuration": [
                'sequence_length', 'prediction_days', 'train_test_split'
            ],
            "Model Architecture": [
                'input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout_rate'
            ],
            "Training Configuration": [
                'learning_rate', 'epochs', 'batch_size', 'patience', 'gradient_clip'
            ],
            "Optimizer Configuration": [
                'beta1', 'beta2', 'epsilon'
            ]
        }
        
        for section, params in sections.items():
            config_str += f"\n{section}:\n"
            config_str += "-" * len(section) + "\n"
            for param in params:
                if hasattr(self, param):
                    config_str += f"  {param}: {getattr(self, param)}\n"
        
        return config_str


# Default configuration instance
default_config = Config()
