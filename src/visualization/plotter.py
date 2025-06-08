"""
LSTM Gold Price Prediction - Visualization Module

This module provides comprehensive plotting and visualization capabilities
for the LSTM gold price prediction project.

Author: Ridwan Halim (ridwaanhall)
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import os

from ..utils.logger import setup_logger


class Visualizer:
    """
    Professional visualization class for LSTM gold price prediction project.
    
    Provides comprehensive plotting capabilities including:
    - Training progress visualization
    - Prediction vs actual comparisons
    - Technical indicator plots
    - Performance metrics visualization
    - Feature importance analysis
    """
    
    def __init__(self, 
                 output_dir: str = "visualizations",
                 style: str = "darkgrid",
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Seaborn style for plots
            figsize: Default figure size
            dpi: Resolution for saved plots
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.logger = setup_logger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 12
        
        self.logger.info(f"Visualizer initialized with output directory: {output_dir}")
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss over epochs.
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training History', fontsize=16, fontweight='bold')
            
            # Training and validation loss
            if 'train_loss' in history and 'val_loss' in history:
                axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
                axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
                axes[0, 0].set_title('Loss Over Epochs')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Learning rate
            if 'learning_rate' in history:
                axes[0, 1].plot(history['learning_rate'], color='green', linewidth=2)
                axes[0, 1].set_title('Learning Rate Schedule')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')
            
            # Gradient norm
            if 'grad_norm' in history:
                axes[1, 0].plot(history['grad_norm'], color='orange', linewidth=2)
                axes[1, 0].set_title('Gradient Norm')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Gradient Norm')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Training time
            if 'epoch_time' in history:
                axes[1, 1].plot(history['epoch_time'], color='purple', linewidth=2)
                axes[1, 1].set_title('Training Time per Epoch')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'training_history.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Training history plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
            raise
    
    def plot_predictions(self,
                        actual: np.ndarray,
                        predicted: np.ndarray,
                        dates: Optional[np.ndarray] = None,
                        title: str = "Gold Price Predictions",
                        save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted gold prices.
        
        Args:
            actual: Actual gold prices
            predicted: Predicted gold prices
            dates: Optional dates for x-axis
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            # Ensure arrays are properly shaped
            actual = np.asarray(actual).flatten()
            predicted = np.asarray(predicted).flatten()
            
            # Validate array shapes
            if actual.shape != predicted.shape:
                self.logger.warning(f"Shape mismatch in plot_predictions: actual={actual.shape}, predicted={predicted.shape}")
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
                self.logger.info(f"Arrays truncated to length {min_len} for plotting")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Time series plot
            x_axis = dates if dates is not None else np.arange(len(actual))
            
            ax1.plot(x_axis, actual, label='Actual', color='blue', linewidth=2, alpha=0.8)
            ax1.plot(x_axis, predicted, label='Predicted', color='red', linewidth=2, alpha=0.8)
            ax1.set_title('Actual vs Predicted Gold Prices')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Gold Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Scatter plot
            ax2.scatter(actual, predicted, alpha=0.6, color='green')
            
            # Perfect prediction line
            min_val = min(np.min(actual), np.min(predicted))
            max_val = max(np.max(actual), np.max(predicted))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax2.set_title('Actual vs Predicted Scatter Plot')
            ax2.set_xlabel('Actual Gold Price')
            ax2.set_ylabel('Predicted Gold Price')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add R² score
            correlation = np.corrcoef(actual, predicted)[0, 1]
            r_squared = correlation ** 2
            ax2.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'predictions.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Predictions plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {str(e)}")
            raise
    
    def plot_technical_indicators(self,
                                data: Dict[str, np.ndarray],
                                dates: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot technical indicators used in the model.
        
        Args:
            data: Dictionary containing price and indicator data
            dates: Optional dates for x-axis
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Technical Indicators Analysis', fontsize=16, fontweight='bold')
            
            x_axis = dates if dates is not None else np.arange(len(data.get('price', [])))
            
            # Price and moving averages
            if 'price' in data:
                axes[0, 0].plot(x_axis, data['price'], label='Price', color='blue', linewidth=2)
                if 'sma_20' in data:
                    axes[0, 0].plot(x_axis, data['sma_20'], label='SMA 20', color='red', alpha=0.7)
                if 'ema_20' in data:
                    axes[0, 0].plot(x_axis, data['ema_20'], label='EMA 20', color='green', alpha=0.7)
                axes[0, 0].set_title('Price and Moving Averages')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # RSI
            if 'rsi' in data:
                axes[0, 1].plot(x_axis, data['rsi'], color='purple', linewidth=2)
                axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                axes[0, 1].set_title('RSI (Relative Strength Index)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_ylim(0, 100)
            
            # MACD
            if 'macd' in data and 'macd_signal' in data:
                axes[1, 0].plot(x_axis, data['macd'], label='MACD', color='blue', linewidth=2)
                axes[1, 0].plot(x_axis, data['macd_signal'], label='Signal', color='red', linewidth=2)
                if 'macd_histogram' in data:
                    axes[1, 0].bar(x_axis, data['macd_histogram'], alpha=0.3, color='green', label='Histogram')
                axes[1, 0].set_title('MACD')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Bollinger Bands
            if 'price' in data and 'bb_upper' in data and 'bb_lower' in data:
                axes[1, 1].plot(x_axis, data['price'], label='Price', color='blue', linewidth=2)
                axes[1, 1].plot(x_axis, data['bb_upper'], label='Upper Band', color='red', alpha=0.7)
                axes[1, 1].plot(x_axis, data['bb_lower'], label='Lower Band', color='green', alpha=0.7)
                axes[1, 1].fill_between(x_axis, data['bb_upper'], data['bb_lower'], alpha=0.1, color='gray')
                axes[1, 1].set_title('Bollinger Bands')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Volume
            if 'volume' in data:
                axes[2, 0].bar(x_axis, data['volume'], alpha=0.6, color='orange')
                axes[2, 0].set_title('Trading Volume')
                axes[2, 0].grid(True, alpha=0.3)
            
            # Price distribution
            if 'price' in data:
                axes[2, 1].hist(data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[2, 1].set_title('Price Distribution')
                axes[2, 1].set_xlabel('Price')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'technical_indicators.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Technical indicators plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting technical indicators: {str(e)}")
            raise
    
    def plot_performance_metrics(self,
                               metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics as a bar chart.
        
        Args:
            metrics: Dictionary containing performance metrics
            save_path: Path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
            
            # Regression metrics
            regression_metrics = {k: v for k, v in metrics.items() 
                                if k in ['mse', 'rmse', 'mae', 'mape', 'r2']}
            
            if regression_metrics:
                ax1.bar(regression_metrics.keys(), regression_metrics.values(), 
                       color='skyblue', alpha=0.8, edgecolor='black')
                ax1.set_title('Regression Metrics')
                ax1.set_ylabel('Value')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (k, v) in enumerate(regression_metrics.items()):
                    ax1.text(i, v + 0.01 * max(regression_metrics.values()), 
                            f'{v:.4f}', ha='center', va='bottom')
            
            # Trading metrics
            trading_metrics = {k: v for k, v in metrics.items() 
                             if k in ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']}
            
            if trading_metrics:
                colors = ['green' if v > 0 else 'red' for v in trading_metrics.values()]
                ax2.bar(trading_metrics.keys(), trading_metrics.values(), 
                       color=colors, alpha=0.8, edgecolor='black')
                ax2.set_title('Trading Metrics')
                ax2.set_ylabel('Value')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (k, v) in enumerate(trading_metrics.items()):
                    ax2.text(i, v + 0.01 * max(abs(v) for v in trading_metrics.values()), 
                            f'{v:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'performance_metrics.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Performance metrics plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            raise
    
    def plot_feature_importance(self,
                              features: List[str],
                              importance: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance analysis.
        
        Args:
            features: List of feature names
            importance: Feature importance values
            save_path: Path to save the plot
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[::-1]
            sorted_features = [features[i] for i in sorted_idx]
            sorted_importance = importance[sorted_idx]
            
            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            bars = ax.barh(range(len(features)), sorted_importance, color=colors)
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, importance_val) in enumerate(zip(bars, sorted_importance)):
                ax.text(importance_val + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance_val:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'feature_importance.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Feature importance plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_residuals_analysis(self,
                              actual: np.ndarray,
                              predicted: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Plot residuals analysis for model diagnostics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            save_path: Path to save the plot
        """
        try:
            residuals = actual - predicted
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Residuals Analysis', fontsize=16, fontweight='bold')
            
            # Residuals vs Predicted
            axes[0, 0].scatter(predicted, residuals, alpha=0.6, color='blue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residuals over time
            axes[1, 1].plot(residuals, color='green', linewidth=1, alpha=0.8)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'residuals_analysis.png')
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Residuals analysis plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting residuals analysis: {str(e)}")
            raise
    
    def create_dashboard(self,
                        results: Dict[str, Any],
                        save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            results: Dictionary containing all results and data
            save_path: Path to save the dashboard
        """
        try:
            # Create individual plots
            if 'history' in results:
                self.plot_training_history(results['history'])
            
            if 'actual' in results and 'predicted' in results:
                self.plot_predictions(
                    results['actual'], 
                    results['predicted'],
                    results.get('dates')
                )
            
            if 'technical_data' in results:
                self.plot_technical_indicators(results['technical_data'])
            
            if 'metrics' in results:
                self.plot_performance_metrics(results['metrics'])
            
            if 'features' in results and 'feature_importance' in results:
                self.plot_feature_importance(
                    results['features'],
                    results['feature_importance']
                )
            
            if 'actual' in results and 'predicted' in results:
                self.plot_residuals_analysis(
                    results['actual'],
                    results['predicted']
                )
            
            self.logger.info("Dashboard created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            raise
    
    def save_summary_report(self,
                          results: Dict[str, Any],
                          save_path: Optional[str] = None) -> None:
        """
        Save a text summary report of the results.
        
        Args:
            results: Dictionary containing all results
            save_path: Path to save the report
        """
        try:
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'summary_report.txt')
            
            with open(save_path, 'w') as f:
                f.write("LSTM Gold Price Prediction - Summary Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Model Configuration
                if 'config' in results:
                    f.write("Model Configuration:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in results['config'].items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Performance Metrics
                if 'metrics' in results:
                    f.write("Performance Metrics:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in results['metrics'].items():
                        f.write(f"{key}: {value:.4f}\n")
                    f.write("\n")
                
                # Training Summary
                if 'history' in results:
                    history = results['history']
                    if 'train_loss' in history and 'val_loss' in history:
                        f.write("Training Summary:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
                        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
                        f.write(f"Best Validation Loss: {min(history['val_loss']):.4f}\n")
                        f.write(f"Total Epochs: {len(history['train_loss'])}\n\n")
            
            self.logger.info(f"Summary report saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving summary report: {str(e)}")
            raise
