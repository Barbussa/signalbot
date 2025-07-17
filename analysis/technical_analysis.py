"""
Technical analysis module for trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from utils.logger import setup_logger

logger = setup_logger()

class TechnicalAnalysis:
    """Main technical analysis class"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
            
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            return prices.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series()
            
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return prices.ewm(span=period).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series()
            
    def find_swing_points(self, highs: pd.Series, lows: pd.Series, window: int = 20) -> Tuple[float, float]:
        """Find recent swing high and low points"""
        try:
            # Find recent swing high and low
            recent_data = min(len(highs), window)
            
            swing_high = highs.tail(recent_data).max()
            swing_low = lows.tail(recent_data).min()
            
            return swing_high, swing_low
            
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return None, None
            
    def calculate_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            recent_prices = prices.tail(window)
            
            # Simple support/resistance calculation
            resistance = recent_prices.max()
            support = recent_prices.min()
            
            # Calculate pivot points
            pivot = (resistance + support + recent_prices.iloc[-1]) / 3
            
            return {
                'resistance': resistance,
                'support': support,
                'pivot': pivot
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
            
    def calculate_volatility(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate price volatility"""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(period)
            return volatility.iloc[-1] if not volatility.empty else 0
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
            
    def detect_trend(self, prices: pd.Series, short_period: int = 10, long_period: int = 20) -> str:
        """Detect market trend direction"""
        try:
            short_ma = self.calculate_sma(prices, short_period)
            long_ma = self.calculate_sma(prices, long_period)
            
            if short_ma.empty or long_ma.empty:
                return "SIDEWAYS"
                
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            
            if current_short > current_long:
                return "UPTREND"
            elif current_short < current_long:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return "SIDEWAYS"
            
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
            
    def is_rsi_overbought(self, rsi_value: float, threshold: float = 70) -> bool:
        """Check if RSI indicates overbought conditions"""
        return rsi_value > threshold
        
    def is_rsi_oversold(self, rsi_value: float, threshold: float = 30) -> bool:
        """Check if RSI indicates oversold conditions"""
        return rsi_value < threshold
        
    def is_rsi_neutral(self, rsi_value: float, lower: float = 40, upper: float = 60) -> bool:
        """Check if RSI is in neutral zone"""
        return lower <= rsi_value <= upper
        
    def calculate_price_change(self, prices: pd.Series, periods: int = 1) -> Dict[str, float]:
        """Calculate price change over specified periods"""
        try:
            if len(prices) < periods + 1:
                return {'change': 0, 'change_percent': 0}
                
            current_price = prices.iloc[-1]
            previous_price = prices.iloc[-(periods + 1)]
            
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            return {
                'change': change,
                'change_percent': change_percent
            }
            
        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return {'change': 0, 'change_percent': 0}
            
    def analyze_momentum(self, prices: pd.Series, period: int = 10) -> Dict[str, float]:
        """Analyze price momentum"""
        try:
            momentum = prices.diff(period)
            momentum_sma = momentum.rolling(window=5).mean()
            
            return {
                'momentum': momentum.iloc[-1] if not momentum.empty else 0,
                'momentum_sma': momentum_sma.iloc[-1] if not momentum_sma.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'momentum': 0, 'momentum_sma': 0}
            
    def get_signal_strength(self, indicators: Dict) -> float:
        """Calculate signal strength based on multiple indicators"""
        try:
            strength = 0
            max_strength = 0
            
            # RSI contribution
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi > 70:
                    strength += 2  # Strong overbought
                elif rsi > 60:
                    strength += 1  # Moderate overbought
                elif rsi < 30:
                    strength += 2  # Strong oversold
                elif rsi < 40:
                    strength += 1  # Moderate oversold
                max_strength += 2
                
            # Trend contribution
            if 'trend' in indicators:
                trend = indicators['trend']
                if trend in ['UPTREND', 'DOWNTREND']:
                    strength += 1
                max_strength += 1
                
            # Fibonacci contribution
            if 'fib_distance' in indicators:
                fib_distance = indicators['fib_distance']
                if fib_distance < 10:  # Very close to 0.618 level
                    strength += 2
                elif fib_distance < 20:
                    strength += 1
                max_strength += 2
                
            # Calculate percentage
            if max_strength > 0:
                return (strength / max_strength) * 100
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0
