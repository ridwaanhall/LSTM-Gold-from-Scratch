"""
Logging utilities for LSTM Gold Price Prediction
"""

import logging
import os
from datetime import datetime
from typing import Optional


class LSTMLogger:
    """
    Professional logging system for the LSTM Gold Price Prediction project.
    Provides both file and console logging with proper formatting.
    """
    
    def __init__(self, name: str = "LSTM_Gold_Predictor", 
                 log_file: Optional[str] = None, 
                 log_level: str = "INFO"):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplication
        self.logger.handlers.clear()
        
        # Create formatters
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        self._setup_console_handler()
        
        # File handler
        if log_file:
            self._setup_file_handler(log_file)
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str):
        """Setup file logging handler."""
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_config(self, config):
        """Log configuration parameters."""
        self.info("Configuration Parameters:")
        self.info("=" * 50)
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        for key, value in config_dict.items():
            if not key.startswith('_'):
                self.info(f"{key}: {value}")
        self.info("=" * 50)
    
    def log_training_start(self, total_epochs: int, batch_size: int, learning_rate: float):
        """Log training start information."""
        self.info("Starting LSTM Model Training")
        self.info("=" * 40)
        self.info(f"Total Epochs: {total_epochs}")
        self.info(f"Batch Size: {batch_size}")
        self.info(f"Learning Rate: {learning_rate}")
        self.info("=" * 40)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                  learning_rate: Optional[float] = None):
        """Log epoch training progress."""
        message = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            message += f", Val Loss = {val_loss:.6f}"
        if learning_rate is not None:
            message += f", LR = {learning_rate:.8f}"
        self.info(message)
    
    def log_model_save(self, filepath: str, epoch: int):
        """Log model saving."""
        self.info(f"Model saved to {filepath} at epoch {epoch}")
    
    def log_data_info(self, data_shape: tuple, train_size: int, test_size: int):
        """Log data information."""
        self.info(f"Data loaded with shape: {data_shape}")
        self.info(f"Training samples: {train_size}")
        self.info(f"Testing samples: {test_size}")
    
    def log_prediction_start(self, days: int):
        """Log prediction start."""
        self.info(f"Starting prediction for {days} days ahead")
    
    def log_metrics(self, metrics: dict):
        """Log evaluation metrics."""
        self.info("Model Performance Metrics:")
        self.info("-" * 30)
        for metric, value in metrics.items():
            self.info(f"{metric}: {value:.6f}")
        self.info("-" * 30)
    
    def log_api_request(self, url: str, status_code: int, data_points: int):
        """Log API request information."""
        self.info(f"API Request: {url}")
        self.info(f"Status Code: {status_code}")
        self.info(f"Data Points Retrieved: {data_points}")
    
    def log_preprocessing_info(self, original_size: int, processed_size: int, 
                             features_added: list):
        """Log data preprocessing information."""
        self.info(f"Data preprocessing completed:")
        self.info(f"Original size: {original_size}")
        self.info(f"Processed size: {processed_size}")
        if features_added:
            self.info(f"Features added: {', '.join(features_added)}")
    
    def log_error_with_traceback(self, message: str, exception: Exception):
        """Log error with full traceback."""
        self.error(f"{message}: {str(exception)}")
        self.error("Traceback:", exc_info=True)


# Create default logger instance
def get_logger(name: str = "LSTM_Gold_Predictor", 
               log_file: Optional[str] = None, 
               log_level: str = "INFO") -> LSTMLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured LSTMLogger instance
    """
    return LSTMLogger(name, log_file, log_level)
