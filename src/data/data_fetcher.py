"""
Gold price data fetcher from Pegadaian API
"""

import json
import os
import pickle
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..utils.logger import get_logger


class GoldDataFetcher:
    """
    Professional data fetcher for gold price data from Pegadaian API.
    Handles data retrieval, caching, and basic validation.
    """
    
    def __init__(self, config, logger=None):
        """
        Initialize the data fetcher.
        
        Args:
            config: Configuration object containing API URL and paths
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or get_logger()
        self.api_url = config.api_url
        self.cache_file = os.path.join(config.data_dir, 'gold_price_cache.pkl')
        self.raw_data_file = os.path.join(config.data_dir, 'raw_gold_data.json')
        
    def fetch_data(self, use_cache: bool = True, cache_duration: int = 3600) -> List[Dict]:
        """
        Fetch gold price data from API or cache.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_duration: Cache duration in seconds (default: 1 hour)
            
        Returns:
            List of gold price records
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            # Check cache first
            if use_cache and self._is_cache_valid(cache_duration):
                self.logger.info("Loading data from cache")
                return self._load_from_cache()
            
            # Fetch from API
            self.logger.info(f"Fetching data from API: {self.api_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(self.api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Log API response
            self.logger.log_api_request(
                self.api_url, 
                response.status_code, 
                len(data.get('data', {}).get('priceList', []))
            )
            
            # Validate response structure
            if not self._validate_response(data):
                raise ValueError("Invalid API response structure")
            
            # Extract price list
            price_list = data['data']['priceList']
            
            # Save raw data
            self._save_raw_data(data)
            
            # Cache processed data
            self._save_to_cache(price_list)
            
            self.logger.info(f"Successfully fetched {len(price_list)} price records")
            return price_list
            
        except requests.RequestException as e:
            self.logger.error(f"Network error while fetching data: {str(e)}")
            # Try to load from cache as fallback
            if os.path.exists(self.cache_file):
                self.logger.warning("Using cached data as fallback")
                return self._load_from_cache()
            raise
        
        except Exception as e:
            self.logger.log_error_with_traceback("Failed to fetch gold price data", e)
            raise
    
    def _validate_response(self, data: Dict) -> bool:
        """
        Validate API response structure.
        
        Args:
            data: API response data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check response code
            if data.get('responseCode') != '2000000100':
                self.logger.warning(f"API returned non-success code: {data.get('responseCode')}")
                return False
            
            # Check data structure
            if 'data' not in data or 'priceList' not in data['data']:
                self.logger.error("Missing required data structure in API response")
                return False
            
            price_list = data['data']['priceList']
            if not isinstance(price_list, list) or len(price_list) == 0:
                self.logger.error("Empty or invalid price list in API response")
                return False
            
            # Validate first few records
            for i, record in enumerate(price_list[:5]):
                required_fields = ['hargaJual', 'hargaBeli', 'lastUpdate']
                for field in required_fields:
                    if field not in record:
                        self.logger.error(f"Missing field '{field}' in record {i}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating response: {str(e)}")
            return False
    
    def _is_cache_valid(self, cache_duration: int) -> bool:
        """
        Check if cache is valid based on age.
        
        Args:
            cache_duration: Maximum cache age in seconds
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(self.cache_file):
            return False
        
        cache_age = datetime.now().timestamp() - os.path.getmtime(self.cache_file)
        return cache_age < cache_duration
    
    def _load_from_cache(self) -> List[Dict]:
        """
        Load data from cache file.
        
        Returns:
            Cached price data
        """
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            raise
    
    def _save_to_cache(self, data: List[Dict]):
        """
        Save data to cache file.
        
        Args:
            data: Price data to cache
        """
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Data cached to {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")
    
    def _save_raw_data(self, data: Dict):
        """
        Save raw API response data.
        
        Args:
            data: Raw API response
        """
        try:
            with open(self.raw_data_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Raw data saved to {self.raw_data_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save raw data: {str(e)}")
    
    def get_data_summary(self, data: List[Dict]) -> Dict:
        """
        Get summary statistics of the fetched data.
        
        Args:
            data: Gold price data
            
        Returns:
            Summary statistics
        """
        if not data:
            return {}
        
        try:
            # Extract numeric values
            selling_prices = [float(record['hargaJual']) for record in data if record.get('hargaJual')]
            buying_prices = [float(record['hargaBeli']) for record in data if record.get('hargaBeli')]
            
            # Date range
            dates = [record['lastUpdate'] for record in data if record.get('lastUpdate')]
            dates.sort()
            
            summary = {
                'total_records': len(data),
                'date_range': {
                    'start': dates[0] if dates else None,
                    'end': dates[-1] if dates else None
                },
                'selling_price': {
                    'min': min(selling_prices) if selling_prices else 0,
                    'max': max(selling_prices) if selling_prices else 0,
                    'mean': np.mean(selling_prices) if selling_prices else 0,
                    'std': np.std(selling_prices) if selling_prices else 0
                },
                'buying_price': {
                    'min': min(buying_prices) if buying_prices else 0,
                    'max': max(buying_prices) if buying_prices else 0,
                    'mean': np.mean(buying_prices) if buying_prices else 0,
                    'std': np.std(buying_prices) if buying_prices else 0
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {str(e)}")
            return {}
    
    def validate_data_quality(self, data: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate data quality and report issues.
        
        Args:
            data: Gold price data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not data:
            issues.append("No data available")
            return False, issues
        
        # Check for minimum data points
        if len(data) < self.config.sequence_length:
            issues.append(f"Insufficient data points: {len(data)} < {self.config.sequence_length}")
        
        # Check for missing values
        missing_selling = sum(1 for record in data if not record.get('hargaJual'))
        missing_buying = sum(1 for record in data if not record.get('hargaBeli'))
        missing_dates = sum(1 for record in data if not record.get('lastUpdate'))
        
        if missing_selling > 0:
            issues.append(f"Missing selling prices: {missing_selling} records")
        if missing_buying > 0:
            issues.append(f"Missing buying prices: {missing_buying} records")
        if missing_dates > 0:
            issues.append(f"Missing dates: {missing_dates} records")
        
        # Check for duplicate dates
        dates = [record['lastUpdate'] for record in data if record.get('lastUpdate')]
        if len(dates) != len(set(dates)):
            issues.append("Duplicate dates found in data")
        
        # Check for outliers (prices that are too different from the mean)
        try:
            selling_prices = [float(record['hargaJual']) for record in data 
                            if record.get('hargaJual') and record['hargaJual'].isdigit()]
            if selling_prices:
                mean_price = np.mean(selling_prices)
                std_price = np.std(selling_prices)
                outliers = [p for p in selling_prices if abs(p - mean_price) > 3 * std_price]
                if outliers:
                    issues.append(f"Potential outliers detected: {len(outliers)} records")
        except Exception as e:
            issues.append(f"Error checking for outliers: {str(e)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clear_cache(self):
        """Clear cached data."""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                self.logger.info("Cache cleared successfully")
            if os.path.exists(self.raw_data_file):
                os.remove(self.raw_data_file)
                self.logger.info("Raw data file cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
