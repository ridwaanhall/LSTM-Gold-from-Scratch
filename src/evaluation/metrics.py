"""
Evaluation metrics for LSTM Gold Price Prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..utils.logger import get_logger


class MetricsCalculator:
    """
    Professional metrics calculator for evaluating LSTM model performance.
    """
    
    def __init__(self, logger=None):
        """
        Initialize metrics calculator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger()
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        non_zero_denom = denominator != 0
        if np.any(non_zero_denom):
            smape = np.mean(np.abs(y_true[non_zero_denom] - y_pred[non_zero_denom]) / denominator[non_zero_denom]) * 100
        else:
            smape = 0
        
        # Mean Absolute Scaled Error (MASE) - for time series
        # Using naive forecast (previous value) as benchmark
        if len(y_true) > 1:
            naive_forecast = np.roll(y_true, 1)[1:]  # Previous values
            actual_values = y_true[1:]
            mae_naive = np.mean(np.abs(actual_values - naive_forecast))
            
            if mae_naive != 0:
                mase = mae / mae_naive
            else:
                mase = np.inf
        else:
            mase = np.inf
        
        # Directional Accuracy (for trend prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Explained variance
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'mase': mase,
            'r2': r2,
            'explained_variance': explained_variance,
            'max_error': max_error,
            'directional_accuracy': directional_accuracy
        }
        
        return metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                initial_investment: float = 10000) -> Dict[str, float]:
        """
        Calculate trading-specific metrics for gold price prediction.
        
        Args:
            y_true: True gold prices
            y_pred: Predicted gold prices
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary of trading metrics
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        if len(y_true) < 2:
            return {}
        
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Trading strategy: buy when predicted to go up, sell when predicted to go down
        predicted_directions = np.diff(y_pred) > 0
        actual_directions = np.diff(y_true) > 0
        
        # Simple strategy returns
        strategy_returns = np.where(predicted_directions, true_returns, -true_returns)
        
        # Buy and hold returns
        buy_hold_return = (y_true[-1] - y_true[0]) / y_true[0]
        
        # Strategy cumulative return
        strategy_cumulative_return = np.prod(1 + strategy_returns) - 1
        
        # Sharpe ratio (assuming 252 trading days per year)
        if np.std(strategy_returns) != 0:
            sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(strategy_returns > 0) * 100
        
        # Profit factor
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        if len(negative_returns) > 0 and np.sum(negative_returns) != 0:
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))
        else:
            profit_factor = np.inf if len(positive_returns) > 0 else 0
        
        # Calmar ratio
        if max_drawdown != 0:
            calmar_ratio = strategy_cumulative_return / abs(max_drawdown)
        else:
            calmar_ratio = np.inf if strategy_cumulative_return > 0 else 0
        
        trading_metrics = {
            'strategy_return': strategy_cumulative_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': (strategy_cumulative_return - buy_hold_return) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'final_portfolio_value': initial_investment * (1 + strategy_cumulative_return)
        }
        
        return trading_metrics
    
    def calculate_time_series_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    frequency: str = 'daily') -> Dict[str, float]:
        """
        Calculate time series specific metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary of time series metrics
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Theil's U statistic
        if len(y_true) > 1:
            naive_forecast = np.roll(y_true, 1)[1:]
            actual_values = y_true[1:]
            predicted_values = y_pred[1:]
            
            mse_model = np.mean((actual_values - predicted_values) ** 2)
            mse_naive = np.mean((actual_values - naive_forecast) ** 2)
            
            if mse_naive != 0:
                theil_u = np.sqrt(mse_model) / np.sqrt(mse_naive)
            else:
                theil_u = np.inf
        else:
            theil_u = np.inf
        
        # Autocorrelation of residuals
        residuals = y_true - y_pred
        if len(residuals) > 1:
            residual_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            if np.isnan(residual_autocorr):
                residual_autocorr = 0
        else:
            residual_autocorr = 0
        
        # Forecast bias
        forecast_bias = np.mean(y_pred - y_true)
        
        # Tracking signal
        mad = np.mean(np.abs(residuals))  # Mean Absolute Deviation
        if mad != 0:
            tracking_signal = np.sum(residuals) / (mad * len(residuals))
        else:
            tracking_signal = 0
        
        time_series_metrics = {
            'theil_u': theil_u,
            'residual_autocorr': residual_autocorr,
            'forecast_bias': forecast_bias,
            'tracking_signal': tracking_signal,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals)
        }
        
        return time_series_metrics
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               include_trading: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive evaluation with all metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            include_trading: Whether to include trading metrics
            
        Returns:
            Dictionary of metric categories
        """
        self.logger.info("Calculating comprehensive evaluation metrics")
        
        evaluation = {
            'regression_metrics': self.calculate_regression_metrics(y_true, y_pred),
            'time_series_metrics': self.calculate_time_series_metrics(y_true, y_pred)
        }
        
        if include_trading:
            evaluation['trading_metrics'] = self.calculate_trading_metrics(y_true, y_pred)
        
        # Log metrics
        for category, metrics in evaluation.items():
            self.logger.info(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) > 1000:
                        self.logger.info(f"  {metric}: {value:.2e}")
                    else:
                        self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.info(f"  {metric}: {value}")
        
        return evaluation
    
    def compare_models(self, predictions_dict: Dict[str, np.ndarray], 
                      y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple model predictions.
        
        Args:
            predictions_dict: Dictionary of model name -> predictions
            y_true: True values
            
        Returns:
            Dictionary of model name -> metrics
        """
        comparison = {}
        
        for model_name, y_pred in predictions_dict.items():
            self.logger.info(f"Evaluating model: {model_name}")
            comparison[model_name] = self.calculate_regression_metrics(y_true, y_pred)
        
        return comparison
    
    def get_metric_summary(self, metrics: Dict[str, float], top_n: int = 5) -> Dict[str, float]:
        """
        Get summary of most important metrics.
        
        Args:
            metrics: Full metrics dictionary
            top_n: Number of top metrics to return
            
        Returns:
            Summary metrics
        """
        # Define importance order
        importance_order = [
            'mae', 'rmse', 'mape', 'r2', 'directional_accuracy',
            'smape', 'mase', 'explained_variance', 'max_error'
        ]
        
        summary = {}
        count = 0
        
        for metric in importance_order:
            if metric in metrics and count < top_n:
                summary[metric] = metrics[metric]
                count += 1
        
        return summary
