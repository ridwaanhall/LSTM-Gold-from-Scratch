#!/usr/bin/env python3
"""
LSTM Gold Price Prediction - Main Execution Script

This is the main entry point for the LSTM gold price prediction project.
It orchestrates the entire pipeline from data fetching to model training,
evaluation, and visualization.

Usage:
    python main.py [--config config.yaml] [--mode train|predict|evaluate]

Author: GitHub Copilot
Date: 2024
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
from src.evaluation.metrics import ModelEvaluator
from src.visualization.plotter import Visualizer


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
            # Data components
            self.data_fetcher = GoldDataFetcher(
                cache_dir=self.config['data']['cache_dir'],
                max_retries=self.config['data']['max_retries']
            )
            
            self.preprocessor = GoldDataPreprocessor(
                sequence_length=self.config['model']['sequence_length'],
                validation_split=self.config['training']['validation_split'],
                test_split=self.config['training']['test_split']
            )
            
            # Model components
            self.model = LSTMNetwork(
                input_size=self.config['model']['input_size'],
                hidden_sizes=self.config['model']['hidden_sizes'],
                output_size=self.config['model']['output_size'],
                dropout_rate=self.config['model']['dropout_rate']
            )
            
            self.optimizer = AdamOptimizer(
                learning_rate=self.config['training']['learning_rate'],
                beta1=self.config['training']['beta1'],
                beta2=self.config['training']['beta2'],
                epsilon=self.config['training']['epsilon']
            )
            
            self.trainer = LSTMTrainer(
                model=self.model,
                optimizer=self.optimizer,
                loss_function=self.config['training']['loss_function'],
                early_stopping_patience=self.config['training']['early_stopping_patience'],
                checkpoint_dir=self.config['training']['checkpoint_dir']
            )
            
            # Evaluation and visualization
            self.evaluator = ModelEvaluator()
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
            
            self.raw_data = self.data_fetcher.fetch_gold_prices(
                days=self.config['data']['days_to_fetch']
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
            
            # Preprocess data
            self.processed_data = self.preprocessor.fit_transform(self.raw_data)
            
            # Create sequences and split data
            sequences, targets = self.preprocessor.create_sequences(
                self.processed_data['features'],
                self.processed_data['targets']
            )
            
            # Split into train/val/test
            splits = self.preprocessor.train_test_split(sequences, targets)
            self.train_data = {'X': splits['X_train'], 'y': splits['y_train']}
            self.val_data = {'X': splits['X_val'], 'y': splits['y_val']}
            self.test_data = {'X': splits['X_test'], 'y': splits['y_test']}
            
            # Update model input size based on actual features
            self.config['model']['input_size'] = self.processed_data['features'].shape[1]
            
            self.logger.info(f"Data preprocessing completed")
            self.logger.info(f"Training samples: {len(self.train_data['X'])}")
            self.logger.info(f"Validation samples: {len(self.val_data['X'])}")
            self.logger.info(f"Test samples: {len(self.test_data['X'])}")
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def train_model(self) -> Dict[str, Any]:
        """Train the LSTM model."""
        try:
            self.logger.info("Starting model training...")
            
            if self.train_data is None or self.val_data is None:
                raise ValueError("No training data available. Please preprocess data first.")
            
            # Train the model
            history = self.trainer.train(
                X_train=self.train_data['X'],
                y_train=self.train_data['y'],
                X_val=self.val_data['X'],
                y_val=self.val_data['y'],
                epochs=self.config['training']['epochs'],
                batch_size=self.config['training']['batch_size'],
                verbose=True
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
            
            # Inverse transform predictions and targets
            actual = self.preprocessor.inverse_transform_targets(self.test_data['y'])
            predicted = self.preprocessor.inverse_transform_targets(predictions)
            
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
            
            # Prepare technical indicators data
            technical_data = {
                'price': self.raw_data['price'].values,
                'volume': self.raw_data.get('volume', np.zeros_like(self.raw_data['price'].values))
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
                'dates': self.raw_data.index[-len(actual):] if hasattr(self.raw_data, 'index') else None
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
            
            # Get the last sequence from processed data
            last_sequence = self.processed_data['features'][-self.config['model']['sequence_length']:]
            last_sequence = last_sequence.reshape(1, *last_sequence.shape)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predict next value
                pred = self.model.predict(current_sequence)
                predictions.append(pred[0, 0])
                
                # Update sequence for next prediction (simplified approach)
                # In practice, you might want to incorporate the prediction back into features
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]  # Update last price feature
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.preprocessor.inverse_transform_targets(predictions)
            
            # Generate future dates
            last_date = self.raw_data.index[-1] if hasattr(self.raw_data, 'index') else datetime.now()
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
