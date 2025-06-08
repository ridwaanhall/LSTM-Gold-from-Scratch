#!/usr/bin/env python3
"""
LSTM Gold Price Prediction - Demo Script

A simplified demo script to showcase the LSTM gold price prediction system.
This script runs a quick demo with reduced data and epochs for testing purposes.

Author: Ridwan Halim (ridwaanhall)
Date: 2025
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import GoldPredictionPipeline


def run_demo():
    """Run a quick demo of the LSTM gold price prediction system."""
    print("="*60)
    print("LSTM GOLD PRICE PREDICTION - DEMO")
    print("="*60)
    print("This demo will:")
    print("- Fetch last 365 days of gold price data")
    print("- Train a small LSTM model for 10 epochs")
    print("- Evaluate performance and create visualizations")
    print("- Predict next 7 days of gold prices")
    print("="*60)
    
    try:
        # Initialize pipeline with demo configuration
        pipeline = GoldPredictionPipeline()
        
        # Override some config values for demo
        pipeline.config['data']['days_to_fetch'] = 365
        pipeline.config['training']['epochs'] = 10
        pipeline.config['training']['batch_size'] = 16
        pipeline.config['model']['hidden_sizes'] = [32, 16]
        pipeline.config['training']['early_stopping_patience'] = 5
        
        print("\n1. Setting up components...")
        pipeline.setup_components()
        
        print("2. Fetching gold price data...")
        pipeline.fetch_data()
        print(f"   ✓ Fetched {len(pipeline.raw_data)} data points")
        
        print("3. Preprocessing data...")
        pipeline.preprocess_data()
        print(f"   ✓ Training samples: {len(pipeline.train_data['X'])}")
        print(f"   ✓ Validation samples: {len(pipeline.val_data['X'])}")
        print(f"   ✓ Test samples: {len(pipeline.test_data['X'])}")
        
        print("4. Training LSTM model...")
        print("   (This may take a few minutes...)")
        history = pipeline.train_model()
        print(f"   ✓ Training completed in {len(history['train_loss'])} epochs")
        print(f"   ✓ Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"   ✓ Final validation loss: {history['val_loss'][-1]:.4f}")
        
        print("5. Evaluating model performance...")
        metrics, actual, predicted = pipeline.evaluate_model()
        print("   ✓ Key metrics:")
        for key in ['rmse', 'mae', 'r2', 'mape']:
            if key in metrics:
                print(f"     {key.upper()}: {metrics[key]:.4f}")
        
        print("6. Creating visualizations...")
        pipeline.create_visualizations(history, metrics, actual, predicted)
        print("   ✓ Plots saved to 'visualizations/' directory")
        
        print("7. Saving trained model...")
        pipeline.save_model()
        print("   ✓ Model saved to 'models/' directory")
        
        print("8. Predicting future prices...")
        dates, predictions = pipeline.predict_future(7)
        print("   ✓ Next 7 days predictions:")
        for date, price in zip(dates, predictions):
            print(f"     {date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the following directories for outputs:")
        print("- visualizations/ : All plots and charts")
        print("- models/ : Trained model files")
        print("- logs/ : Training and execution logs")
        print("- data/ : Cached data files")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("\nPlease check the logs for more details.")
        print("Make sure you have internet connection for data fetching.")
        return False
    
    return True


def run_quick_test():
    """Run a very quick test with minimal data."""
    print("Running quick test (30 days data, 3 epochs)...")
    
    try:
        pipeline = GoldPredictionPipeline()
        
        # Minimal configuration
        pipeline.config['data']['days_to_fetch'] = 30
        pipeline.config['training']['epochs'] = 3
        pipeline.config['training']['batch_size'] = 8
        pipeline.config['model']['hidden_sizes'] = [16]
        
        pipeline.setup_components()
        pipeline.fetch_data()
        pipeline.preprocess_data()
        history = pipeline.train_model()
        
        print(f"✓ Quick test passed! Final loss: {history['train_loss'][-1]:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        success = run_quick_test()
    else:
        success = run_demo()
    
    sys.exit(0 if success else 1)
