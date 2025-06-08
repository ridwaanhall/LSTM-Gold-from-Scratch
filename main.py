#!/usr/bin/env python3
"""
LSTM Gold Price Prediction - Main Execution Script

This is the main entry point for the LSTM gold price prediction project.
It orchestrates the entire pipeline from data fetching to model training,
evaluation, and visualization.

Usage:
    python main.py [--config config.yaml] [--mode train|predict|evaluate]

Author: Ridwan Halim (ridwaanhall)
Date: 2025
"""

import argparse
import sys
import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.data.data_fetcher import GoldDataFetcher
from src.data.preprocessor import GoldDataPreprocessor
from src.models.lstm_network import LSTMNetwork
from src.training.optimizer import AdamOptimizer
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import MetricsCalculator
from src.visualization.plotter import Visualizer


# Config wrapper classes for bridging dict-based config with object-based components
class DataConfig:
    """Data configuration wrapper."""
    def __init__(self, config_dict):
        self.days_to_fetch = config_dict.get('days_to_fetch', 365)
        self.api_url = config_dict.get('api_url', 'https://sahabat.pegadaian.co.id/gold/prices/chart?interval=3650&isRequest=true')
        self.data_dir = config_dict.get('cache_dir', 'data/cache')
        self.cache_enabled = config_dict.get('cache_enabled', True)
        self.cache_expiry_hours = config_dict.get('cache_expiry_hours', 24)
        self.max_retries = config_dict.get('max_retries', 3)
        self.api_timeout = config_dict.get('api_timeout', 30)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'days_to_fetch': self.days_to_fetch,
            'api_url': self.api_url,
            'data_dir': self.data_dir,
            'cache_enabled': self.cache_enabled,
            'cache_expiry_hours': self.cache_expiry_hours,
            'max_retries': self.max_retries,
            'api_timeout': self.api_timeout
        }


class ModelConfig:
    """Model configuration wrapper."""
    def __init__(self, config_dict):
        self.input_size = config_dict.get('input_size', 24)
        self.hidden_size = config_dict.get('hidden_size', 64)
        self.num_layers = config_dict.get('num_layers', 2)
        self.output_size = config_dict.get('output_size', 1)
        self.dropout_rate = config_dict.get('dropout_rate', 0.2)
        self.hidden_sizes = config_dict.get('hidden_sizes', [64, 32])
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'hidden_sizes': self.hidden_sizes
        }


class TrainerConfig:
    """Trainer configuration wrapper."""
    def __init__(self, config_dict):
        self.epochs = config_dict.get('epochs', 100)
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 0.001)
        self.early_stopping_patience = config_dict.get('early_stopping_patience', 10)
        self.lr_decay_factor = config_dict.get('lr_decay_factor', 0.95)
        self.lr_decay_patience = config_dict.get('lr_decay_patience', 5)
        self.validation_split = config_dict.get('validation_split', 0.2)
        self.shuffle = config_dict.get('shuffle', True)
        self.verbose = config_dict.get('verbose', True)
        self.save_model = config_dict.get('save_model', True)
        self.save_frequency = config_dict.get('save_frequency', 10)
        self.models_dir = config_dict.get('models_dir', 'models')
        self.model_filename = config_dict.get('model_filename', 'lstm_gold_model.pkl')
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'lr_decay_factor': self.lr_decay_factor,
            'lr_decay_patience': self.lr_decay_patience,
            'validation_split': self.validation_split,
            'shuffle': self.shuffle,
            'verbose': self.verbose,
            'save_model': self.save_model,
            'save_frequency': self.save_frequency,
            'models_dir': self.models_dir,
            'model_filename': self.model_filename
        }


class GoldPredictionPipeline:
    """
    Main pipeline class for LSTM gold price prediction.
    
    This class orchestrates the entire machine learning pipeline including:
    - Data fetching and preprocessing
    - Model creation and training
    - Evaluation and visualization
    - Model persistence and loading
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.data_fetcher = None
        self.preprocessor = None
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.evaluator = None
        self.visualizer = None
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.logger.info("Gold Prediction Pipeline initialized")
    
    def setup_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Create config objects using module-level classes
            data_config = DataConfig(self.config['data'])
            model_config = ModelConfig(self.config['model'])
            trainer_config = TrainerConfig(self.config['training'])
            
            # Ensure directories exist
            os.makedirs(self.config['data']['cache_dir'], exist_ok=True)
            os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
            os.makedirs(self.config['visualization']['output_dir'], exist_ok=True)
            
            # Data components
            self.data_fetcher = GoldDataFetcher(
                config=data_config,
                logger=self.logger
            )
            
            self.preprocessor = GoldDataPreprocessor(
                config=self.config,
                logger=self.logger
            )
            
            # Training components - extend trainer config with model config
            trainer_config.input_size = model_config.input_size
            trainer_config.hidden_size = model_config.hidden_size
            trainer_config.num_layers = model_config.num_layers
            trainer_config.output_size = model_config.output_size
            trainer_config.dropout_rate = model_config.dropout_rate
            trainer_config.gradient_clip = self.config['training'].get('gradient_clip_norm', 1.0)
            trainer_config.loss_function = self.config['training']['loss_function']
            trainer_config.patience = self.config['training']['early_stopping_patience']
            trainer_config.min_delta = self.config['training']['min_delta']
            trainer_config.checkpoint_dir = self.config['training']['checkpoint_dir']
            trainer_config.save_best_only = self.config['training']['save_best_only']
            trainer_config.beta1 = self.config['training']['beta1']
            trainer_config.beta2 = self.config['training']['beta2']
            trainer_config.epsilon = self.config['training']['epsilon']
            
            self.trainer = LSTMTrainer(
                config=trainer_config,
                logger=self.logger
            )
            
            # Set references to trainer's model and optimizer for later access
            self.model = self.trainer.model
            self.optimizer = self.trainer.optimizer
            
            # Evaluation and visualization
            self.evaluator = MetricsCalculator()
            self.visualizer = Visualizer(
                output_dir=self.config['visualization']['output_dir']
            )
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up components: {str(e)}")
            raise
    
    def fetch_data(self) -> None:
        """Fetch gold price data from API."""
        try:
            self.logger.info("Fetching gold price data...")
            
            self.raw_data = self.data_fetcher.fetch_data(
                use_cache=True,
                cache_duration=self.config['data'].get('cache_duration', 3600)
            )
            
            self.logger.info(f"Fetched {len(self.raw_data)} data points")
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def preprocess_data(self) -> None:
        """Preprocess the raw data for training."""
        try:
            self.logger.info("Preprocessing data...")
            
            if self.raw_data is None:
                raise ValueError("No raw data available. Please fetch data first.")
            
            # Preprocess data using the actual method available
            X_sequences, y_sequences, preprocessing_info = self.preprocessor.preprocess_data(self.raw_data)
            
            # Store processed data
            self.processed_data = {
                'features': X_sequences,
                'targets': y_sequences,
                'preprocessing_info': preprocessing_info
            }
            
            # Split data into train/val/test
            self._split_data(X_sequences, y_sequences)
            
            # Update model input size based on actual features
            if len(X_sequences.shape) == 3:  # (samples, timesteps, features)
                self.config['model']['input_size'] = X_sequences.shape[2]
                # Update the model's input size and reinitialize if needed
                self._update_model_input_size(X_sequences.shape[2])
            
            self.logger.info(f"Data preprocessing completed")
            self.logger.info(f"Training samples: {len(self.train_data['X'])}")
            self.logger.info(f"Validation samples: {len(self.val_data['X'])}")
            self.logger.info(f"Test samples: {len(self.test_data['X'])}")
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _split_data(self, X_sequences: np.ndarray, y_sequences: np.ndarray) -> None:
        """
        Split the preprocessed data into training, validation, and test sets.
        
        Args:
            X_sequences: Input sequences (features)
            y_sequences: Target sequences (labels)
        """
        try:
            # Get split ratios from config
            val_split = self.config['training']['validation_split']
            test_split = self.config['training']['test_split']
            
            # Calculate split indices
            total_samples = len(X_sequences)
            test_size = int(total_samples * test_split)
            val_size = int(total_samples * val_split)
            train_size = total_samples - test_size - val_size
            
            self.logger.info(f"Splitting {total_samples} samples into train({train_size}), val({val_size}), test({test_size})")
            
            # Split data chronologically (important for time series)
            # Train: earliest data
            # Validation: middle data  
            # Test: most recent data
            X_train = X_sequences[:train_size]
            y_train = y_sequences[:train_size]
            
            X_val = X_sequences[train_size:train_size + val_size]
            y_val = y_sequences[train_size:train_size + val_size]
            
            X_test = X_sequences[train_size + val_size:]
            y_test = y_sequences[train_size + val_size:]
            
            # Store split data
            self.train_data = {'X': X_train, 'y': y_train}
            self.val_data = {'X': X_val, 'y': y_val}
            self.test_data = {'X': X_test, 'y': y_test}
            
            self.logger.info("Data splitting completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def _update_model_input_size(self, new_input_size: int) -> None:
        """
        Update the model's input size and reinitialize if necessary.
        
        Args:
            new_input_size: The new input size based on actual features
        """
        try:
            # Update trainer config
            if hasattr(self.trainer, 'config'):
                self.trainer.config.input_size = new_input_size
            
            # Update model config and reinitialize model
            if hasattr(self.trainer, 'model'):
                # Update model input size
                self.trainer.model.input_size = new_input_size
                
                # Only update the first LSTM layer's input size
                # Subsequent layers should maintain their hidden_size as input_size
                for i, layer in enumerate(self.trainer.model.layers):
                    if hasattr(layer, 'lstm_cell'):
                        if i == 0:  # Only first layer gets the new input size
                            # Update LSTM cell input size
                            layer.lstm_cell.input_size = new_input_size
                            # Reinitialize parameters with correct dimensions
                            layer.lstm_cell._initialize_parameters()
                            
                            self.logger.info(f"Updated first LSTM layer input size to {new_input_size}")
                        # Other layers keep their input size unchanged (hidden_size from previous layer)
                
                # Update references
                self.model = self.trainer.model
                
                self.logger.info(f"Model input size updated to {new_input_size}")
            
        except Exception as e:
            self.logger.error(f"Error updating model input size: {str(e)}")
            raise

    def train_model(self) -> Dict[str, Any]:
        """Train the LSTM model."""
        try:
            self.logger.info("Starting model training...")
            
            if self.train_data is None or self.val_data is None:
                raise ValueError("No training data available. Please preprocess data first.")
            
            # Combine train and validation data for the trainer
            # (trainer will split internally based on validation_split)
            X_combined = np.concatenate([self.train_data['X'], self.val_data['X']], axis=0)
            y_combined = np.concatenate([self.train_data['y'], self.val_data['y']], axis=0)
            
            # Calculate validation split ratio
            val_split = len(self.val_data['X']) / (len(self.train_data['X']) + len(self.val_data['X']))
            
            # Train the model
            history = self.trainer.train(
                X=X_combined,
                y=y_combined,
                validation_split=val_split
            )
            
            self.logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        try:
            self.logger.info("Evaluating model performance...")
            
            if self.test_data is None:
                raise ValueError("No test data available. Please preprocess data first.")
            
            if self.model is None:
                raise ValueError("No trained model available. Please train model first.")
            
            # Make predictions on test set
            predictions = self.model.predict(self.test_data['X'])
            
            # Ensure predictions are properly shaped
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            # Inverse transform predictions and targets if preprocessing info is available
            if self.processed_data and 'preprocessing_info' in self.processed_data:
                scaler_info = self.processed_data['preprocessing_info']['scaler_info']
                actual = self.preprocessor.inverse_transform_predictions(self.test_data['y'], scaler_info)
                predicted = self.preprocessor.inverse_transform_predictions(predictions, scaler_info)
            else:
                # Use raw values if no preprocessing info available
                actual = self.test_data['y']
                predicted = predictions
            
            # Ensure both arrays have the same shape
            if actual.ndim > 1 and actual.shape[1] == 1:
                actual = actual.flatten()
            if predicted.ndim > 1 and predicted.shape[1] == 1:
                predicted = predicted.flatten()
            
            # Debug logging for array shapes
            self.logger.debug(f"Actual shape: {actual.shape}, Predicted shape: {predicted.shape}")
            
            # Final shape validation
            if actual.shape != predicted.shape:
                self.logger.warning(f"Shape mismatch detected: actual={actual.shape}, predicted={predicted.shape}")
                # Truncate to match the smaller array
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
                self.logger.info(f"Arrays truncated to length {min_len}")
            
            # Calculate metrics
            regression_metrics = self.evaluator.calculate_regression_metrics(actual, predicted)
            trading_metrics = self.evaluator.calculate_trading_metrics(actual, predicted)
            ts_metrics = self.evaluator.calculate_time_series_metrics(actual, predicted)
            
            # Combine all metrics
            all_metrics = {**regression_metrics, **trading_metrics, **ts_metrics}
            
            self.logger.info("Model evaluation completed")
            for metric, value in all_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            return all_metrics, actual, predicted
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def create_visualizations(self, 
                            history: Dict[str, Any],
                            metrics: Dict[str, float],
                            actual: np.ndarray,
                            predicted: np.ndarray) -> None:
        """Create comprehensive visualizations."""
        try:
            self.logger.info("Creating visualizations...")
            
            # Convert raw data to DataFrame for visualization
            df_for_viz = self.preprocessor._convert_to_dataframe(self.raw_data)
            
            # Prepare technical indicators data
            technical_data = {
                'price': df_for_viz['selling_price'].values,
                'volume': df_for_viz.get('volume', np.zeros(len(df_for_viz))).values if 'volume' in df_for_viz.columns else np.zeros(len(df_for_viz))
            }
            
            # Add technical indicators if available
            if hasattr(self.preprocessor, 'technical_indicators'):
                technical_data.update(self.preprocessor.technical_indicators)
            
            # Create comprehensive results dictionary
            results = {
                'history': history,
                'metrics': metrics,
                'actual': actual,
                'predicted': predicted,
                'technical_data': technical_data,
                'config': self.config,
                'dates': df_for_viz['date'].iloc[-len(actual):] if len(df_for_viz) >= len(actual) else None
            }
            
            # Create dashboard
            self.visualizer.create_dashboard(results)
            
            # Save summary report
            self.visualizer.save_summary_report(results)
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save the trained model and preprocessor."""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"models/lstm_gold_model_{timestamp}.pkl"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model and preprocessor."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.config = model_data['config']
            
            self.logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_future(self, days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future gold prices.
        
        Args:
            days: Number of days to predict
            
        Returns:
            Tuple of (dates, predictions)
        """
        try:
            self.logger.info(f"Predicting gold prices for next {days} days...")
            
            if self.model is None or self.preprocessor is None:
                raise ValueError("No trained model available. Please train or load a model first.")
            
            # Debug: Check processed_data structure
            self.logger.debug(f"Processed data keys: {list(self.processed_data.keys()) if self.processed_data else 'None'}")
            
            # Get the last sequence from processed data
            features = self.processed_data['features']
            sequence_length = self.config['model']['sequence_length']
            self.logger.debug(f"Features shape: {features.shape}, sequence_length: {sequence_length}")
            
            last_sequence = features[-sequence_length:]
            last_sequence = last_sequence.reshape(1, *last_sequence.shape)
            self.logger.debug(f"Last sequence shape: {last_sequence.shape}")
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for i in range(days):
                # Predict next value
                pred = self.model.predict(current_sequence)
                self.logger.debug(f"Prediction {i} shape: {pred.shape}, value: {pred[0, 0] if pred.size > 0 else 'empty'}")
                predictions.append(pred[0, 0])
                
                # Update sequence for next prediction (simplified approach)
                # In practice, you might want to incorporate the prediction back into features
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]  # Update last price feature
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            self.logger.debug(f"Predictions array shape before inverse transform: {predictions.shape}")
            
            if self.processed_data and 'preprocessing_info' in self.processed_data:
                scaler_info = self.processed_data['preprocessing_info']['scaler_info']
                predictions = self.preprocessor.inverse_transform_predictions(predictions, scaler_info)
                self.logger.debug(f"Predictions shape after inverse transform: {predictions.shape}")
            else:
                predictions = predictions.flatten()
                self.logger.debug(f"Predictions shape after flatten: {predictions.shape}")
            
            # Generate future dates
            if isinstance(self.raw_data, list) and len(self.raw_data) > 0:
                # Extract date from last record
                last_record = self.raw_data[-1]
                last_date_str = last_record.get('date', last_record.get('timestamp', ''))
                try:
                    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                except:
                    last_date = datetime.now()
            else:
                last_date = datetime.now()
            
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            self.logger.info(f"Future predictions completed")
            return np.array(future_dates), predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Error predicting future prices: {str(e)}")
            raise
    
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline from data fetching to evaluation."""
        try:
            self.logger.info("Starting full pipeline execution...")
            
            # Setup components
            self.setup_components()
            
            # Data pipeline
            self.fetch_data()
            self.preprocess_data()
            
            # Training pipeline
            history = self.train_model()
            
            # Evaluation pipeline
            metrics, actual, predicted = self.evaluate_model()
            
            # Visualization pipeline
            self.create_visualizations(history, metrics, actual, predicted)
            
            # Save model
            self.save_model()
            
            self.logger.info("Full pipeline execution completed successfully")
            
            # Print summary
            print("\n" + "="*60)
            print("LSTM GOLD PRICE PREDICTION - EXECUTION SUMMARY")
            print("="*60)
            print(f"Training completed with {len(history['train_loss'])} epochs")
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
            print("\nKey Performance Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            print("\nVisualizations and model saved successfully!")
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline execution: {str(e)}")
            raise


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="LSTM Gold Price Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run full pipeline with default config
  python main.py --config custom.yaml     # Run with custom configuration
  python main.py --mode train              # Train model only
  python main.py --mode predict --days 30 # Predict next 30 days
  python main.py --mode evaluate           # Evaluate existing model
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'train', 'predict', 'evaluate'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to pre-trained model file'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to predict (for predict mode)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = GoldPredictionPipeline(args.config)
        
        if args.verbose:
            pipeline.logger.setLevel(logging.DEBUG)
        
        # Execute based on mode
        if args.mode == 'full':
            print("Running full pipeline...")
            pipeline.run_full_pipeline()
            
        elif args.mode == 'train':
            print("Training mode...")
            pipeline.setup_components()
            pipeline.fetch_data()
            pipeline.preprocess_data()
            history = pipeline.train_model()
            pipeline.save_model()
            print("Training completed and model saved!")
            
        elif args.mode == 'predict':
            print(f"Prediction mode - forecasting {args.days} days...")
            if args.model_path:
                pipeline.load_model(args.model_path)
            else:
                # Try to find the latest model
                model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                if not model_files:
                    raise FileNotFoundError("No trained model found. Please train a model first.")
                latest_model = max(model_files)
                pipeline.load_model(f'models/{latest_model}')
            
            dates, predictions = pipeline.predict_future(args.days)
            
            print("\nFuture Gold Price Predictions:")
            print("-" * 40)
            for date, price in zip(dates, predictions):
                print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
            
        elif args.mode == 'evaluate':
            print("Evaluation mode...")
            if args.model_path:
                pipeline.load_model(args.model_path)
            pipeline.setup_components()
            pipeline.fetch_data()
            pipeline.preprocess_data()
            metrics, actual, predicted = pipeline.evaluate_model()
            
            print("\nModel Performance Metrics:")
            print("-" * 30)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
