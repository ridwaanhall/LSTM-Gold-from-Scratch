#!/usr/bin/env python3
"""
LSTM Gold Price Prediction - Usage Examples

This script demonstrates various ways to use the LSTM gold price prediction system.
It shows different execution modes and configuration options.

Author: Ridwan Halim (ridwaanhall)
Date: 2025
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import GoldPredictionPipeline


def example_1_quick_training():
    """Example 1: Quick training with minimal data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: QUICK TRAINING")
    print("="*60)
    print("Training a small model with 100 days of data for quick testing")
    
    pipeline = GoldPredictionPipeline()
    
    # Configure for quick training
    pipeline.config['data']['days_to_fetch'] = 100
    pipeline.config['training']['epochs'] = 5
    pipeline.config['training']['batch_size'] = 16
    pipeline.config['model']['hidden_sizes'] = [32, 16]
    
    start_time = time.time()
    
    try:
        pipeline.setup_components()
        pipeline.fetch_data()
        pipeline.preprocess_data()
        history = pipeline.train_model()
        metrics, actual, predicted = pipeline.evaluate_model()
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ Training completed in {elapsed_time:.1f} seconds")
        print(f"✓ Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"✓ Test RMSE: {metrics['rmse']:.4f}")
        print(f"✓ Test R²: {metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_2_custom_architecture():
    """Example 2: Custom model architecture."""
    print("\n" + "="*60)
    print("EXAMPLE 2: CUSTOM MODEL ARCHITECTURE")
    print("="*60)
    print("Training with custom LSTM architecture and hyperparameters")
    
    pipeline = GoldPredictionPipeline()
    
    # Custom architecture
    pipeline.config['model']['hidden_sizes'] = [128, 64, 32]
    pipeline.config['model']['dropout_rate'] = 0.3
    pipeline.config['model']['sequence_length'] = 30
    
    # Custom training parameters
    pipeline.config['training']['learning_rate'] = 0.002
    pipeline.config['training']['batch_size'] = 64
    pipeline.config['training']['epochs'] = 20
    
    try:
        pipeline.setup_components()
        pipeline.fetch_data()
        pipeline.preprocess_data()
        
        print(f"Model architecture: {pipeline.model.hidden_sizes}")
        print(f"Input features: {pipeline.config['model']['input_size']}")
        print(f"Sequence length: {pipeline.config['model']['sequence_length']}")
        
        history = pipeline.train_model()
        pipeline.save_model("models/custom_model.pkl")
        
        print(f"✓ Custom model trained and saved")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_3_prediction_only():
    """Example 3: Load model and make predictions."""
    print("\n" + "="*60)
    print("EXAMPLE 3: PREDICTION WITH PRE-TRAINED MODEL")
    print("="*60)
    print("Loading a pre-trained model and making future predictions")
    
    pipeline = GoldPredictionPipeline()
    
    try:
        # Try to find an existing model
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if not model_files:
            print("No pre-trained model found. Training a quick model first...")
            
            # Quick training
            pipeline.config['data']['days_to_fetch'] = 365
            pipeline.config['training']['epochs'] = 10
            pipeline.setup_components()
            pipeline.fetch_data()
            pipeline.preprocess_data()
            pipeline.train_model()
            pipeline.save_model("models/example_model.pkl")
            model_path = "models/example_model.pkl"
        else:
            model_path = f"models/{model_files[0]}"
            pipeline.load_model(model_path)
        
        print(f"Using model: {model_path}")
        
        # Make predictions for different periods
        for days in [7, 14, 30]:
            dates, predictions = pipeline.predict_future(days)
            
            print(f"\nPredictions for next {days} days:")
            print("-" * 40)
            for i, (date, price) in enumerate(zip(dates[:5], predictions[:5])):
                print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
            if days > 5:
                print(f"  ... and {days-5} more days")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_4_comprehensive_analysis():
    """Example 4: Comprehensive analysis with all features."""
    print("\n" + "="*60)
    print("EXAMPLE 4: COMPREHENSIVE ANALYSIS")
    print("="*60)
    print("Full pipeline with comprehensive evaluation and visualization")
    
    pipeline = GoldPredictionPipeline()
    
    # Full configuration
    pipeline.config['data']['days_to_fetch'] = 1000
    pipeline.config['training']['epochs'] = 30
    pipeline.config['model']['hidden_sizes'] = [64, 32]
    
    try:
        pipeline.setup_components()
        pipeline.fetch_data()
        pipeline.preprocess_data()
        
        print(f"Dataset size: {len(pipeline.raw_data)} days")
        print(f"Training samples: {len(pipeline.train_data['X'])}")
        print(f"Features: {pipeline.processed_data['features'].shape[1]}")
        
        # Training
        history = pipeline.train_model()
        
        # Evaluation
        metrics, actual, predicted = pipeline.evaluate_model()
        
        # Create comprehensive visualizations
        pipeline.create_visualizations(history, metrics, actual, predicted)
        
        # Save everything
        pipeline.save_model()
        
        print("\nComprehensive Analysis Results:")
        print("-" * 40)
        print(f"Training epochs: {len(history['train_loss'])}")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        
        print("\nRegression Metrics:")
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            if metric in metrics:
                print(f"  {metric.upper()}: {metrics[metric]:.4f}")
        
        print("\nTrading Metrics:")
        for metric in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
            if metric in metrics:
                print(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
        
        print("\n✓ All visualizations saved to 'visualizations/' directory")
        print("✓ Model saved to 'models/' directory")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_5_hyperparameter_comparison():
    """Example 5: Compare different hyperparameter settings."""
    print("\n" + "="*60)
    print("EXAMPLE 5: HYPERPARAMETER COMPARISON")
    print("="*60)
    print("Comparing different model configurations")
    
    configurations = [
        {
            'name': 'Small Model',
            'hidden_sizes': [32],
            'learning_rate': 0.001,
            'dropout_rate': 0.1
        },
        {
            'name': 'Medium Model',
            'hidden_sizes': [64, 32],
            'learning_rate': 0.001,
            'dropout_rate': 0.2
        },
        {
            'name': 'Large Model',
            'hidden_sizes': [128, 64, 32],
            'learning_rate': 0.0005,
            'dropout_rate': 0.3
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTraining {config['name']}...")
        
        try:
            pipeline = GoldPredictionPipeline()
            
            # Configure model
            pipeline.config['model']['hidden_sizes'] = config['hidden_sizes']
            pipeline.config['training']['learning_rate'] = config['learning_rate']
            pipeline.config['model']['dropout_rate'] = config['dropout_rate']
            
            # Quick training for comparison
            pipeline.config['data']['days_to_fetch'] = 365
            pipeline.config['training']['epochs'] = 10
            pipeline.config['training']['batch_size'] = 32
            
            pipeline.setup_components()
            pipeline.fetch_data()
            pipeline.preprocess_data()
            history = pipeline.train_model()
            metrics, _, _ = pipeline.evaluate_model()
            
            results.append({
                'name': config['name'],
                'final_loss': history['val_loss'][-1],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            })
            
            print(f"  ✓ {config['name']}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"  ❌ {config['name']}: Error - {str(e)}")
    
    if results:
        print("\nComparison Results:")
        print("-" * 60)
        print(f"{'Model':<15} {'Final Loss':<12} {'RMSE':<10} {'R²':<10}")
        print("-" * 60)
        for result in results:
            print(f"{result['name']:<15} {result['final_loss']:<12.4f} "
                  f"{result['rmse']:<10.4f} {result['r2']:<10.4f}")
        
        # Find best model
        best_model = max(results, key=lambda x: x['r2'])
        print(f"\n✓ Best model: {best_model['name']} (R² = {best_model['r2']:.4f})")


def main():
    """Run all examples."""
    print("LSTM Gold Price Prediction - Usage Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns")
    print("Choose an example to run:")
    print()
    print("1. Quick Training (5 minutes)")
    print("2. Custom Architecture")
    print("3. Prediction Only")
    print("4. Comprehensive Analysis (15+ minutes)")
    print("5. Hyperparameter Comparison (10+ minutes)")
    print("6. Run All Examples")
    print()
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == '1':
        example_1_quick_training()
    elif choice == '2':
        example_2_custom_architecture()
    elif choice == '3':
        example_3_prediction_only()
    elif choice == '4':
        example_4_comprehensive_analysis()
    elif choice == '5':
        example_5_hyperparameter_comparison()
    elif choice == '6':
        print("\nRunning all examples...")
        example_1_quick_training()
        example_2_custom_architecture()
        example_3_prediction_only()
        example_4_comprehensive_analysis()
        example_5_hyperparameter_comparison()
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("Check the following directories for outputs:")
    print("- visualizations/ : Generated plots and charts")
    print("- models/ : Trained model files")
    print("- logs/ : Execution logs")
    print("="*60)


if __name__ == "__main__":
    main()
