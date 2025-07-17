"""
Fibonacci retracement analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from utils.logger import setup_logger

logger = setup_logger()

class FibonacciAnalysis:
    """Fibonacci retracement analysis class"""
    
    def __init__(self):
        # Standard Fibonacci retracement levels
        self.fib_levels = {
            '0.236': 0.236,
            '0.382': 0.382,
            '0.500': 0.500,
            '0.618': 0.618,
            '0.786': 0.786
        }
        
    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if swing_high is None or swing_low is None:
                return {}
                
            # Calculate the range
            range_value = swing_high - swing_low
            
            # Calculate each Fibonacci level
            fib_levels = {}
            
            for level_name, level_value in self.fib_levels.items():
                # For upward retracement (from low to high)
                fib_levels[f'up_{level_name}'] = swing_low + (range_value * level_value)
                
                # For downward retracement (from high to low)
                fib_levels[f'down_{level_name}'] = swing_high - (range_value * level_value)
                
            return fib_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
            
    def find_fibonacci_support_resistance(self, prices: pd.Series, window: int = 50) -> Dict[str, Dict]:
        """Find Fibonacci support and resistance levels"""
        try:
            if len(prices) < window:
                window = len(prices)
                
            recent_prices = prices.tail(window)
            
            # Find swing high and low
            swing_high = recent_prices.max()
            swing_low = recent_prices.min()
            
            # Find indices of swing points
            high_idx = recent_prices.idxmax()
            low_idx = recent_prices.idxmin()
            
            # Calculate Fibonacci levels
            fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low)
            
            # Determine if we're in uptrend or downtrend based on swing points
            if high_idx > low_idx:
                # Recent swing high after swing low - potential uptrend
                trend = 'uptrend'
                relevant_levels = {k: v for k, v in fib_levels.items() if k.startswith('up_')}
            else:
                # Recent swing low after swing high - potential downtrend
                trend = 'downtrend'
                relevant_levels = {k: v for k, v in fib_levels.items() if k.startswith('down_')}
                
            return {
                'trend': trend,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'levels': relevant_levels,
                'all_levels': fib_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding Fibonacci support/resistance: {e}")
            return {}
            
    def get_618_level(self, swing_high: float, swing_low: float, trend: str = 'auto') -> float:
        """Get the 0.618 Fibonacci level specifically"""
        try:
            if swing_high is None or swing_low is None:
                return None
                
            range_value = swing_high - swing_low
            
            if trend == 'uptrend' or trend == 'auto':
                # 0.618 retracement from swing low
                return swing_low + (range_value * 0.618)
            else:
                # 0.618 retracement from swing high
                return swing_high - (range_value * 0.618)
                
        except Exception as e:
            logger.error(f"Error getting 0.618 level: {e}")
            return None
            
    def calculate_distance_to_fib_level(self, current_price: float, fib_level: float, pip_value: float = 0.0001) -> float:
        """Calculate distance from current price to Fibonacci level in pips"""
        try:
            if fib_level is None:
                return float('inf')
                
            distance = abs(current_price - fib_level)
            return distance / pip_value
            
        except Exception as e:
            logger.error(f"Error calculating distance to Fibonacci level: {e}")
            return float('inf')
            
    def is_price_near_fib_level(self, current_price: float, fib_level: float, threshold_pips: float = 20, pip_value: float = 0.0001) -> bool:
        """Check if current price is near a Fibonacci level"""
        try:
            if fib_level is None:
                return False
                
            distance = self.calculate_distance_to_fib_level(current_price, fib_level, pip_value)
            return distance <= threshold_pips
            
        except Exception as e:
            logger.error(f"Error checking if price is near Fibonacci level: {e}")
            return False
            
    def analyze_fibonacci_signal(self, prices: pd.Series, current_price: float, window: int = 50) -> Dict:
        """Analyze Fibonacci levels for trading signals"""
        try:
            # Get Fibonacci analysis
            fib_analysis = self.find_fibonacci_support_resistance(prices, window)
            
            if not fib_analysis:
                return {'signal': 'NONE', 'confidence': 0}
                
            # Get the 0.618 level specifically
            trend = fib_analysis['trend']
            swing_high = fib_analysis['swing_high']
            swing_low = fib_analysis['swing_low']
            
            # Calculate 0.618 level
            if trend == 'uptrend':
                fib_618 = swing_low + ((swing_high - swing_low) * 0.618)
                signal_bias = 'BUY'  # Expecting bounce from 0.618 in uptrend
            else:
                fib_618 = swing_high - ((swing_high - swing_low) * 0.618)
                signal_bias = 'SELL'  # Expecting rejection from 0.618 in downtrend
                
            # Calculate distance to 0.618 level
            distance_to_618 = self.calculate_distance_to_fib_level(current_price, fib_618)
            
            # Check if price is near 0.618 level
            near_618 = self.is_price_near_fib_level(current_price, fib_618, threshold_pips=30)
            
            # Calculate confidence based on proximity to 0.618 level
            if distance_to_618 <= 10:
                confidence = 90
            elif distance_to_618 <= 20:
                confidence = 75
            elif distance_to_618 <= 30:
                confidence = 60
            elif distance_to_618 <= 50:
                confidence = 45
            else:
                confidence = 20
                
            return {
                'signal': signal_bias if near_618 else 'NONE',
                'confidence': confidence,
                'fib_618': fib_618,
                'distance_to_618': distance_to_618,
                'trend': trend,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'near_618': near_618,
                'all_levels': fib_analysis['levels']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Fibonacci signal: {e}")
            return {'signal': 'NONE', 'confidence': 0}
            
    def get_fibonacci_targets(self, entry_price: float, swing_high: float, swing_low: float, signal_type: str) -> Dict[str, float]:
        """Calculate Fibonacci-based take profit and stop loss levels"""
        try:
            range_value = swing_high - swing_low
            
            if signal_type == 'BUY':
                # For buy signals
                take_profit_1 = entry_price + (range_value * 0.382)
                take_profit_2 = entry_price + (range_value * 0.618)
                take_profit_3 = swing_high  # Full retracement
                
                stop_loss = swing_low - (range_value * 0.236)  # Below swing low
                
            else:  # SELL
                # For sell signals
                take_profit_1 = entry_price - (range_value * 0.382)
                take_profit_2 = entry_price - (range_value * 0.618)
                take_profit_3 = swing_low  # Full retracement
                
                stop_loss = swing_high + (range_value * 0.236)  # Above swing high
                
            return {
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            return {}
            
    def validate_fibonacci_setup(self, prices: pd.Series, current_price: float, window: int = 50) -> bool:
        """Validate if current setup is suitable for Fibonacci analysis"""
        try:
            if len(prices) < window:
                return False
                
            recent_prices = prices.tail(window)
            
            # Check for sufficient price movement
            swing_high = recent_prices.max()
            swing_low = recent_prices.min()
            
            price_range = swing_high - swing_low
            average_price = recent_prices.mean()
            
            # Range should be at least 0.5% of average price for meaningful analysis
            min_range = average_price * 0.005
            
            return price_range >= min_range
            
        except Exception as e:
            logger.error(f"Error validating Fibonacci setup: {e}")
            return False
