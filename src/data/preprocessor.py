"""
Data preprocessing and feature engineering for gold price prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..utils.logger import get_logger


class GoldDataPreprocessor:
    """
    Professional data preprocessor for gold price data.
    Handles feature engineering, normalization, and sequence creation for LSTM.
    """
    
    def __init__(self, config, logger=None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or get_logger()
        self.scaler = None
        self.original_columns = None
        self.feature_columns = None
        
    def preprocess_data(self, raw_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            raw_data: Raw gold price data from API
            
        Returns:
            Tuple of (X_sequences, y_sequences, preprocessing_info)
        """
        try:
            self.logger.info("Starting data preprocessing pipeline")
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(raw_data)
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add technical indicators
            if self.config.add_technical_indicators:
                df = self._add_technical_indicators(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Normalize data
            if self.config.normalize_data:
                df_normalized, scaler_info = self._normalize_data(df)
            else:
                df_normalized = df.copy()
                scaler_info = None
            
            # Create sequences for LSTM
            X, y = self._create_sequences(df_normalized)
            
            # Preprocessing information
            preprocessing_info = {
                'original_shape': (len(raw_data),),
                'processed_shape': df.shape,
                'sequence_shape': X.shape,
                'target_shape': y.shape,
                'features': list(df.columns),
                'scaler_info': scaler_info,
                'date_range': {
                    'start': df['date'].min(),
                    'end': df['date'].max()
                }
            }
            
            self.logger.info(f"Preprocessing completed: {X.shape} sequences generated")
            return X, y, preprocessing_info
            
        except Exception as e:
            self.logger.log_error_with_traceback("Error in data preprocessing", e)
            raise
    
    def _convert_to_dataframe(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw API data to pandas DataFrame.
        
        Args:
            raw_data: Raw gold price data
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Extract relevant fields
            processed_data = []
            for record in raw_data:
                try:
                    processed_record = {
                        'date': pd.to_datetime(record['lastUpdate']),
                        'selling_price': float(record['hargaJual']),
                        'buying_price': float(record['hargaBeli']),
                        'spread': float(record['hargaJual']) - float(record['hargaBeli'])
                    }
                    processed_data.append(processed_record)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Skipping invalid record: {record} - {str(e)}")
                    continue
            
            df = pd.DataFrame(processed_data)
            self.logger.info(f"Converted {len(df)} records to DataFrame")
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        # Remove invalid prices (negative or zero)
        df = df[(df['selling_price'] > 0) & (df['buying_price'] > 0)]
        
        # Remove outliers using IQR method
        for col in ['selling_price', 'buying_price']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        cleaned_size = len(df)
        self.logger.info(f"Data cleaning: {original_size} -> {cleaned_size} records")
        return df.reset_index(drop=True)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Use selling price as main price for indicators
        price = df['selling_price']
        
        # Moving averages
        df['ma_short'] = price.rolling(window=self.config.moving_average_window).mean()
        df['ma_long'] = price.rolling(window=self.config.moving_average_window * 2).mean()
        
        # Price relative to moving average
        df['price_ma_ratio'] = price / df['ma_short']
        
        # Volatility (rolling standard deviation)
        df['volatility'] = price.rolling(window=self.config.volatility_window).std()
        
        # Rate of change
        df['roc'] = price.pct_change(periods=5)  # 5-day rate of change
        
        # Bollinger Bands
        rolling_mean = price.rolling(window=20).mean()
        rolling_std = price.rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI (Relative Strength Index)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Day of week and month (cyclical features)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Cyclical encoding for temporal features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 3, 7]:
            df[f'price_lag_{lag}'] = price.shift(lag)
            df[f'spread_lag_{lag}'] = df['spread'].shift(lag)
        
        features_added = [
            'ma_short', 'ma_long', 'price_ma_ratio', 'volatility', 'roc',
            'bb_upper', 'bb_lower', 'bb_position', 'rsi',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ] + [f'price_lag_{lag}' for lag in [1, 3, 7]] + [f'spread_lag_{lag}' for lag in [1, 3, 7]]
        
        self.logger.log_preprocessing_info(
            len(df), len(df), features_added
        )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Count missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f"Handling {missing_count} missing values")
            
            # Forward fill first, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining rows with missing values
            df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (normalized_df, scaler_info)
        """
        df_normalized = df.copy()
        
        # Identify numerical columns (exclude date)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use different scalers for different types of features
        price_cols = ['selling_price', 'buying_price', 'spread', 'ma_short', 'ma_long', 
                     'bb_upper', 'bb_lower'] + [col for col in numerical_cols if 'price_lag' in col or 'spread_lag' in col]
        ratio_cols = ['price_ma_ratio', 'bb_position', 'rsi']
        volatility_cols = ['volatility', 'roc']
        
        scalers = {}
        
        # MinMax scaler for price-related features
        if price_cols:
            price_scaler = MinMaxScaler()
            existing_price_cols = [col for col in price_cols if col in df.columns]
            if existing_price_cols:
                df_normalized[existing_price_cols] = price_scaler.fit_transform(df[existing_price_cols])
                scalers['price'] = price_scaler
        
        # Standard scaler for ratio and indicator features
        other_cols = [col for col in numerical_cols if col not in price_cols]
        if other_cols:
            standard_scaler = StandardScaler()
            df_normalized[other_cols] = standard_scaler.fit_transform(df[other_cols])
            scalers['standard'] = standard_scaler
        
        # Store scaler information
        scaler_info = {
            'scalers': scalers,
            'price_columns': existing_price_cols if 'price' in scalers else [],
            'standard_columns': other_cols
        }
        
        self.scaler = scalers
        
        return df_normalized, scaler_info
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Select feature columns (exclude date)
        feature_cols = [col for col in df.columns if col != 'date']
        self.feature_columns = feature_cols
        
        # Create feature matrix
        data = df[feature_cols].values
        
        # Create sequences
        X, y = [], []
        sequence_length = self.config.sequence_length
        
        for i in range(sequence_length, len(data)):
            # Input sequence
            X.append(data[i-sequence_length:i])
            
            # Target (next selling price)
            if 'selling_price' in feature_cols:
                target_idx = feature_cols.index('selling_price')
                y.append(data[i, target_idx])
            else:
                # Fallback to first column if selling_price not found
                y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created sequences: X{X.shape}, y{y.shape}")
        return X, y
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                   scaler_info: Dict) -> np.ndarray:
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions: Normalized predictions
            scaler_info: Scaler information from preprocessing
            
        Returns:
            Predictions in original scale
        """
        if not scaler_info or 'scalers' not in scaler_info:
            return predictions
        
        try:
            # Get price scaler
            if 'price' in scaler_info['scalers']:
                price_scaler = scaler_info['scalers']['price']
                price_cols = scaler_info['price_columns']
                
                if 'selling_price' in price_cols:
                    # Create dummy array for inverse transform
                    dummy_array = np.zeros((len(predictions), len(price_cols)))
                    price_idx = price_cols.index('selling_price')
                    dummy_array[:, price_idx] = predictions.flatten()
                    
                    # Inverse transform
                    inverse_transformed = price_scaler.inverse_transform(dummy_array)
                    return inverse_transformed[:, price_idx]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in inverse transform: {str(e)}")
            return predictions
    
    def prepare_prediction_input(self, recent_data: pd.DataFrame, 
                               scaler_info: Dict) -> np.ndarray:
        """
        Prepare recent data for making predictions.
        
        Args:
            recent_data: Recent gold price data
            scaler_info: Scaler information
            
        Returns:
            Prepared input sequence for prediction
        """
        try:
            # Apply same preprocessing steps
            if self.config.add_technical_indicators:
                recent_data = self._add_technical_indicators(recent_data)
            
            recent_data = self._handle_missing_values(recent_data)
            
            if self.config.normalize_data and scaler_info:
                recent_data = self._apply_existing_scalers(recent_data, scaler_info)
            
            # Get last sequence
            feature_data = recent_data[self.feature_columns].values
            if len(feature_data) >= self.config.sequence_length:
                return feature_data[-self.config.sequence_length:].reshape(1, -1, len(self.feature_columns))
            else:
                raise ValueError(f"Not enough data for prediction: {len(feature_data)} < {self.config.sequence_length}")
                
        except Exception as e:
            self.logger.error(f"Error preparing prediction input: {str(e)}")
            raise
    
    def _apply_existing_scalers(self, df: pd.DataFrame, scaler_info: Dict) -> pd.DataFrame:
        """
        Apply existing scalers to new data.
        
        Args:
            df: DataFrame to transform
            scaler_info: Scaler information
            
        Returns:
            Transformed DataFrame
        """
        df_transformed = df.copy()
        
        if 'price' in scaler_info['scalers']:
            price_scaler = scaler_info['scalers']['price']
            price_cols = [col for col in scaler_info['price_columns'] if col in df.columns]
            if price_cols:
                df_transformed[price_cols] = price_scaler.transform(df[price_cols])
        
        if 'standard' in scaler_info['scalers']:
            standard_scaler = scaler_info['scalers']['standard']
            standard_cols = [col for col in scaler_info['standard_columns'] if col in df.columns]
            if standard_cols:
                df_transformed[standard_cols] = standard_scaler.transform(df[standard_cols])
        
        return df_transformed
