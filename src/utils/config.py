"""
Configuration management for LSTM Gold Price Prediction
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Configuration manager class to handle YAML configuration files
    and provide default values for the LSTM gold price prediction model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
        
        self._validate_config()
        self._create_directories()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            # Data Configuration
            'data': {
                'days_to_fetch': 3650,
                'cache_dir': 'data/cache',
                'max_retries': 3,
                'api_timeout': 30,
                'api_url': "https://sahabat.pegadaian.co.id/gold/prices/chart?interval=3650&isRequest=true"
            },
            
            # Model Configuration
            'model': {
                'sequence_length': 60,
                'input_size': 10,  # Will be set automatically
                'hidden_sizes': [64, 32, 16],
                'output_size': 1,
                'dropout_rate': 0.2,
                'activation': 'tanh'
            },
            
            # Training Configuration
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'test_split': 0.1,
                'early_stopping_patience': 10,
                'min_delta': 0.0001,
                'checkpoint_dir': 'models/checkpoints',
                'save_best_only': True,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'gradient_clip_norm': 1.0,
                'lr_schedule': {
                    'type': 'step',
                    'step_size': 20,
                    'gamma': 0.8
                },
                'loss_function': 'mse'
            },
            
            # Preprocessing Configuration
            'preprocessing': {
                'normalize_features': True,
                'normalize_targets': True,
                'normalization_method': 'minmax',
                'technical_indicators': {
                    'sma_periods': [10, 20, 50],
                    'ema_periods': [12, 26],
                    'rsi_period': 14,
                    'macd_params': [12, 26, 9],
                    'bollinger_period': 20,
                    'bollinger_std': 2
                },
                'lag_features': [1, 2, 3, 5, 10],
                'price_changes': [1, 5, 10, 20],
                'volatility_window': 20,
                'remove_outliers': True,
                'outlier_threshold': 3.0,
                'min_data_points': 100
            },
            
            # Evaluation Configuration
            'evaluation': {
                'regression_metrics': ['mse', 'rmse', 'mae', 'mape', 'r2'],
                'trading_metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'],
                'time_series_metrics': ['directional_accuracy', 'uptrend_accuracy', 'downtrend_accuracy'],
                'trading_cost': 0.001,
                'initial_capital': 10000
            },
            
            # Visualization Configuration
            'visualization': {
                'output_dir': 'visualizations',
                'dpi': 300,
                'figure_size': [12, 8],
                'style': 'darkgrid',
                'color_palette': 'husl',
                'plots': {
                    'training_history': True,
                    'predictions': True,
                    'technical_indicators': True,
                    'performance_metrics': True,
                    'feature_importance': True,
                    'residuals_analysis': True,
                    'dashboard': True
                }
            },
            
            # Logging Configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_handler': {
                    'enabled': True,
                    'filename': 'logs/lstm_gold.log',
                    'max_bytes': 10485760,
                    'backup_count': 5
                },
                'console_handler': {
                    'enabled': True,
                    'level': 'INFO'
                }
            },
            
            # Advanced Configuration
            'advanced': {
                'random_seed': 42,
                'num_workers': 4,
                'prefetch_factor': 2,
                'max_memory_usage': 0.8,
                'cache_size': 1000,
                'use_attention': False,
                'use_bidirectional': False
            }
        }
    
    def _load_config_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file and merge with defaults.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._deep_update(self.config, file_config)
                
        except Exception as e:
            logging.warning(f"Failed to load config file {config_path}: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Deep update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        if self.config['model']['sequence_length'] <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.config['model']['hidden_sizes']:
            raise ValueError("hidden_sizes cannot be empty")
        if any(h <= 0 for h in self.config['model']['hidden_sizes']):
            raise ValueError("All hidden sizes must be positive")
        
        # Training validation
        if self.config['training']['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        if self.config['training']['epochs'] <= 0:
            raise ValueError("epochs must be positive")
        if self.config['training']['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.config['training']['validation_split'] < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 < self.config['training']['test_split'] < 1:
            raise ValueError("test_split must be between 0 and 1")
        
        # Data validation
        if self.config['data']['days_to_fetch'] <= 0:
            raise ValueError("days_to_fetch must be positive")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config['data']['cache_dir'],
            self.config['training']['checkpoint_dir'],
            self.config['visualization']['output_dir'],
            os.path.dirname(self.config['logging']['file_handler']['filename']),
            'models',
            'data'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")
        return self.config[section].copy()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        self._deep_update(self.config, updates)
        self._validate_config()
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config to {output_path}: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "LSTM Gold Price Prediction Configuration:\n"
        config_str += "=" * 50 + "\n"
        
        for section, params in self.config.items():
            config_str += f"\n{section.title()}:\n"
            config_str += "-" * len(section) + "\n"
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, dict):
                        config_str += f"  {key}:\n"
                        for subkey, subvalue in value.items():
                            config_str += f"    {subkey}: {subvalue}\n"
                    else:
                        config_str += f"  {key}: {value}\n"
            else:
                config_str += f"  {params}\n"
        
        return config_str


class Config:
    """
    Legacy configuration class for backward compatibility.
    """
class Config:
    """
    Legacy configuration class for backward compatibility.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with default values and optional overrides.
        
        Args:
            **kwargs: Optional configuration overrides
        """
        # Use ConfigManager for the actual configuration
        self._config_manager = ConfigManager()
        self._config = self._config_manager.get_config()
        
        # Apply any overrides
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'model.hidden_size'
                section, param = key.split('.', 1)
                if section in self._config:
                    if isinstance(self._config[section], dict):
                        self._config[section][param] = value
            else:
                # Handle top-level keys
                if key in self._config:
                    self._config[key] = value
        
        # Create flat attributes for backward compatibility
        self._create_flat_attributes()
        
        # File Paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, 'data')
        self.models_dir = os.path.join(self.project_root, 'models')
        self.logs_dir = os.path.join(self.project_root, 'logs')
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_flat_attributes(self):
        """Create flat attributes for backward compatibility."""
        # Data Configuration
        self.api_url = self._config['data']['api_url']
        self.sequence_length = self._config['model']['sequence_length']
        self.prediction_days = 30  # Default value
        self.train_test_split = 1 - self._config['training']['validation_split'] - self._config['training']['test_split']
        
        # Model Architecture
        self.input_size = self._config['model']['input_size']
        self.hidden_size = self._config['model']['hidden_sizes'][0] if self._config['model']['hidden_sizes'] else 64
        self.num_layers = len(self._config['model']['hidden_sizes'])
        self.output_size = self._config['model']['output_size']
        self.dropout_rate = self._config['model']['dropout_rate']
        
        # Training Configuration
        self.learning_rate = self._config['training']['learning_rate']
        self.epochs = self._config['training']['epochs']
        self.batch_size = self._config['training']['batch_size']
        self.patience = self._config['training']['early_stopping_patience']
        self.gradient_clip = self._config['training']['gradient_clip_norm']
        
        # Optimizer Configuration
        self.beta1 = self._config['training']['beta1']
        self.beta2 = self._config['training']['beta2']
        self.epsilon = self._config['training']['epsilon']
        
        # Data Preprocessing
        self.normalize_data = self._config['preprocessing']['normalize_features']
        self.add_technical_indicators = True
        self.moving_average_window = 10
        self.volatility_window = self._config['preprocessing']['volatility_window']
    
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
        return str(self._config_manager)


# Default configuration instances
default_config = Config()
