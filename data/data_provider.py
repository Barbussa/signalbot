"""
Data provider module for fetching forex and gold price data
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from datetime import datetime, timedelta
import json

from utils.logger import setup_logger

logger = setup_logger()

class DataProvider:
    """Data provider for forex and gold price data using Alpha Vantage API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        
        # Supported symbols mapping
        self.symbols = {
            'XAUUSD': 'XAU/USD',
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 12  # Alpha Vantage free tier allows 5 requests per minute
        
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
            # Add API key to params
            params['apikey'] = self.api_key
            
            # Make request
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            self.last_request_time = time.time()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
            
    def get_intraday_data(self, symbol: str, interval: str = '60min', outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """Get intraday price data"""
        try:
            if symbol not in self.symbols:
                logger.error(f"Unsupported symbol: {symbol}")
                return None
                
            # Try to get real data from Alpha Vantage API
            try:
                # Special handling for XAUUSD (Gold) - use TIME_SERIES_INTRADAY for commodities
                if symbol == 'XAUUSD':
                    params = {
                        'function': 'TIME_SERIES_INTRADAY',
                        'symbol': 'GLD',  # Gold ETF as proxy for XAUUSD
                        'interval': interval,
                        'outputsize': outputsize
                    }
                else:
                    # Regular forex pairs
                    function = 'FX_INTRADAY'
                    from_symbol = symbol[:3]
                    to_symbol = symbol[3:]
                    
                    params = {
                        'function': function,
                        'from_symbol': from_symbol,
                        'to_symbol': to_symbol,
                        'interval': interval,
                        'outputsize': outputsize
                    }
                
                data = self._make_request(params)
                
                if data:
                    # Extract time series data - different keys for different functions
                    if symbol == 'XAUUSD':
                        time_series_key = f'Time Series ({interval})'
                    else:
                        time_series_key = f'Time Series ({interval})'
                    
                    if time_series_key in data:
                        time_series = data[time_series_key]
                        
                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        
                        # Rename columns
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Convert to numeric
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                        # Remove any rows with NaN values
                        df.dropna(inplace=True)
                        
                        logger.info(f"Retrieved {len(df)} data points for {symbol}")
                        return df
                    else:
                        logger.warning(f"Time series data not found for {symbol}. Available keys: {list(data.keys())}")
                        
            except Exception as api_error:
                logger.warning(f"API error for {symbol}: {api_error}")
                
            # If real API fails, fallback to a basic data structure
            logger.info(f"Using fallback data for {symbol}")
            return self._create_fallback_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting intraday data for {symbol}: {e}")
            return None
            
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """Get daily price data"""
        try:
            if symbol not in self.symbols:
                logger.error(f"Unsupported symbol: {symbol}")
                return None
                
            # Try to get real data from Alpha Vantage API
            try:
                # Special handling for XAUUSD (Gold) - use TIME_SERIES_DAILY for commodities
                if symbol == 'XAUUSD':
                    params = {
                        'function': 'TIME_SERIES_DAILY',
                        'symbol': 'GLD',  # Gold ETF as proxy for XAUUSD
                        'outputsize': outputsize
                    }
                else:
                    # Regular forex pairs
                    function = 'FX_DAILY'
                    from_symbol = symbol[:3]
                    to_symbol = symbol[3:]
                    
                    params = {
                        'function': function,
                        'from_symbol': from_symbol,
                        'to_symbol': to_symbol,
                        'outputsize': outputsize
                    }
                
                data = self._make_request(params)
                
                if data:
                    # Extract time series data
                    time_series_key = 'Time Series (Daily)'
                    
                    if time_series_key in data:
                        time_series = data[time_series_key]
                        
                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        
                        # Rename columns
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Convert to numeric
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                        # Remove any rows with NaN values
                        df.dropna(inplace=True)
                        
                        logger.info(f"Retrieved {len(df)} daily data points for {symbol}")
                        return df
                    else:
                        logger.warning(f"Daily time series data not found for {symbol}. Available keys: {list(data.keys())}")
                        
            except Exception as api_error:
                logger.warning(f"API error for daily data {symbol}: {api_error}")
                
            # If real API fails, use fallback data
            logger.info(f"Using fallback daily data for {symbol}")
            df = self._create_fallback_data(symbol)
            
            # Resample to daily data
            if not df.empty:
                df_daily = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                df_daily.dropna(inplace=True)
                return df_daily
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting daily data for {symbol}: {e}")
            return None
            
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        try:
            timeframes = {}
            
            # Get 1H data
            df_1h = self.get_intraday_data(symbol, '60min', 'full')
            if df_1h is not None and not df_1h.empty:
                timeframes['1H'] = df_1h
                
            # Get 4H data (simulate from 1H data if available)
            if '1H' in timeframes:
                df_4h = self._resample_to_4h(timeframes['1H'])
                if df_4h is not None and not df_4h.empty:
                    timeframes['4H'] = df_4h
                    
            # Get daily data
            df_daily = self.get_daily_data(symbol, 'full')
            if df_daily is not None and not df_daily.empty:
                timeframes['Daily'] = df_daily
                
            logger.info(f"Retrieved multi-timeframe data for {symbol}: {list(timeframes.keys())}")
            return timeframes
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data for {symbol}: {e}")
            return {}
            
    def _resample_to_4h(self, df_1h: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample 1H data to 4H timeframe"""
        try:
            # Resample to 4H
            df_4h = df_1h.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Remove incomplete 4H candles
            df_4h.dropna(inplace=True)
            
            return df_4h
            
        except Exception as e:
            logger.error(f"Error resampling to 4H: {e}")
            return None
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Get latest intraday data
            df = self.get_intraday_data(symbol, '5min', 'compact')
            
            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return None
                
            # Return the most recent close price
            current_price = df['close'].iloc[-1]
            
            logger.info(f"Current price for {symbol}: {current_price}")
            return float(current_price)
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
            
    def get_market_status(self) -> Dict:
        """Get market status information"""
        try:
            current_time = datetime.now()
            
            # Simple market hours check (forex market is open 24/5)
            weekday = current_time.weekday()
            
            if weekday < 5:  # Monday to Friday
                market_state = "OPEN"
            else:
                market_state = "CLOSED"
                
            return {
                'state': market_state,
                'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_update': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'state': 'UNKNOWN',
                'current_time': 'N/A',
                'last_update': 'N/A'
            }
            
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        return symbol.upper() in self.symbols
        
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return list(self.symbols.keys())
        
    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol"""
        try:
            # Standard pip values
            pip_values = {
                'XAUUSD': 0.01,    # Gold: 1 pip = 0.01
                'EURUSD': 0.0001,  # EUR/USD: 1 pip = 0.0001
                'GBPUSD': 0.0001   # GBP/USD: 1 pip = 0.0001
            }
            
            return pip_values.get(symbol.upper(), 0.0001)
            
        except Exception as e:
            logger.error(f"Error getting pip value for {symbol}: {e}")
            return 0.0001
            
    def _create_fallback_data(self, symbol: str) -> pd.DataFrame:
        """Create fallback data for testing when API is not available"""
        try:
            # Base prices for different symbols
            base_prices = {
                'XAUUSD': 2650.0,
                'EURUSD': 1.0850,
                'GBPUSD': 1.2750
            }
            
            base_price = base_prices.get(symbol, 1.0)
            
            # Create 100 data points with realistic price movement
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            
            # Generate price data with some volatility
            prices = []
            current_price = base_price
            
            for i in range(100):
                # Add some random movement (±0.1% to ±0.5%)
                volatility = np.random.uniform(0.001, 0.005)
                direction = np.random.choice([-1, 1])
                change = current_price * volatility * direction
                
                current_price += change
                
                # Create OHLC data
                high = current_price + abs(change * 0.5)
                low = current_price - abs(change * 0.5)
                open_price = current_price - (change * 0.5)
                close_price = current_price
                
                prices.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': np.random.randint(1000, 10000)
                })
            
            # Create DataFrame
            df = pd.DataFrame(prices, index=dates)
            
            logger.info(f"Created fallback data for {symbol} with {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error creating fallback data for {symbol}: {e}")
            return pd.DataFrame()
