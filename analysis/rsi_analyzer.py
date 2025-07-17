"""
RSI (Relative Strength Index) analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import setup_logger

logger = setup_logger()

class RSIAnalyzer:
    """RSI analysis and signal generation class"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.technical_analysis = TechnicalAnalysis()
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI using the technical analysis module"""
        return self.technical_analysis.calculate_rsi(prices, self.period)
        
    def analyze_rsi_signal(self, prices: pd.Series, current_price: float) -> Dict:
        """Analyze RSI for trading signals"""
        try:
            if len(prices) < self.period + 1:
                return {'signal': 'NONE', 'confidence': 0, 'rsi': 0}
                
            # Calculate RSI
            rsi_series = self.calculate_rsi(prices)
            
            if rsi_series.empty:
                return {'signal': 'NONE', 'confidence': 0, 'rsi': 0}
                
            current_rsi = rsi_series.iloc[-1]
            previous_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else current_rsi
            
            # Determine signal based on RSI levels
            signal = 'NONE'
            confidence = 0
            
            # Check for oversold condition (potential BUY signal)
            if current_rsi < self.oversold:
                signal = 'BUY'
                confidence = min(90, (self.oversold - current_rsi) * 3)  # Higher confidence for lower RSI
                
            # Check for overbought condition (potential SELL signal)
            elif current_rsi > self.overbought:
                signal = 'SELL'
                confidence = min(90, (current_rsi - self.overbought) * 3)  # Higher confidence for higher RSI
                
            # Check for RSI divergence
            divergence = self.detect_rsi_divergence(prices, rsi_series)
            
            # Adjust confidence based on divergence
            if divergence['type'] != 'NONE':
                confidence = min(95, confidence + 15)
                
            return {
                'signal': signal,
                'confidence': confidence,
                'rsi': current_rsi,
                'previous_rsi': previous_rsi,
                'divergence': divergence,
                'overbought_level': self.overbought,
                'oversold_level': self.oversold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing RSI signal: {e}")
            return {'signal': 'NONE', 'confidence': 0, 'rsi': 0}
            
    def detect_rsi_divergence(self, prices: pd.Series, rsi_series: pd.Series, window: int = 10) -> Dict:
        """Detect RSI divergence patterns"""
        try:
            if len(prices) < window or len(rsi_series) < window:
                return {'type': 'NONE', 'strength': 0}
                
            recent_prices = prices.tail(window)
            recent_rsi = rsi_series.tail(window)
            
            # Find local highs and lows
            price_highs = recent_prices.rolling(window=3, center=True).max()
            price_lows = recent_prices.rolling(window=3, center=True).min()
            
            rsi_highs = recent_rsi.rolling(window=3, center=True).max()
            rsi_lows = recent_rsi.rolling(window=3, center=True).min()
            
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                price_trend = price_lows.iloc[-1] < price_lows.iloc[-2]
                rsi_trend = rsi_lows.iloc[-1] > rsi_lows.iloc[-2]
                
                if price_trend and rsi_trend:
                    return {'type': 'BULLISH', 'strength': 70}
                    
            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                price_trend = price_highs.iloc[-1] > price_highs.iloc[-2]
                rsi_trend = rsi_highs.iloc[-1] < rsi_highs.iloc[-2]
                
                if price_trend and rsi_trend:
                    return {'type': 'BEARISH', 'strength': 70}
                    
            return {'type': 'NONE', 'strength': 0}
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return {'type': 'NONE', 'strength': 0}
            
    def get_rsi_targets(self, current_price: float, signal_type: str, atr_value: float = None) -> Dict[str, float]:
        """Calculate RSI-based take profit and stop loss levels"""
        try:
            if atr_value is None:
                atr_value = current_price * 0.01  # Default 1% ATR
                
            if signal_type == 'BUY':
                # For buy signals from oversold conditions
                take_profit = current_price + (atr_value * 2)  # 2 ATR profit target
                stop_loss = current_price - (atr_value * 1)    # 1 ATR stop loss
                
            elif signal_type == 'SELL':
                # For sell signals from overbought conditions
                take_profit = current_price - (atr_value * 2)  # 2 ATR profit target
                stop_loss = current_price + (atr_value * 1)    # 1 ATR stop loss
                
            else:
                return {}
                
            return {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': 2.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating RSI targets: {e}")
            return {}
            
    def validate_rsi_signal(self, rsi_value: float, signal_type: str) -> bool:
        """Validate RSI signal strength"""
        try:
            if signal_type == 'BUY':
                return rsi_value < self.oversold
            elif signal_type == 'SELL':
                return rsi_value > self.overbought
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating RSI signal: {e}")
            return False
            
    def get_rsi_strength(self, rsi_value: float) -> str:
        """Get RSI strength description"""
        try:
            if rsi_value >= 80:
                return "Extremely Overbought"
            elif rsi_value >= 70:
                return "Overbought"
            elif rsi_value >= 60:
                return "Moderately Overbought"
            elif rsi_value >= 40:
                return "Neutral"
            elif rsi_value >= 30:
                return "Moderately Oversold"
            elif rsi_value >= 20:
                return "Oversold"
            else:
                return "Extremely Oversold"
                
        except Exception as e:
            logger.error(f"Error getting RSI strength: {e}")
            return "Unknown"
            
    def calculate_rsi_momentum(self, rsi_series: pd.Series, window: int = 5) -> float:
        """Calculate RSI momentum (rate of change)"""
        try:
            if len(rsi_series) < window:
                return 0
                
            current_rsi = rsi_series.iloc[-1]
            past_rsi = rsi_series.iloc[-window]
            
            momentum = current_rsi - past_rsi
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating RSI momentum: {e}")
            return 0
            
    def analyze_multi_timeframe_rsi(self, price_data: Dict[str, pd.Series]) -> Dict:
        """Analyze RSI across multiple timeframes"""
        try:
            timeframe_analysis = {}
            
            for timeframe, prices in price_data.items():
                if len(prices) < self.period + 1:
                    continue
                    
                rsi_series = self.calculate_rsi(prices)
                if rsi_series.empty:
                    continue
                    
                current_rsi = rsi_series.iloc[-1]
                rsi_signal = self.analyze_rsi_signal(prices, prices.iloc[-1])
                
                timeframe_analysis[timeframe] = {
                    'rsi': current_rsi,
                    'signal': rsi_signal['signal'],
                    'confidence': rsi_signal['confidence'],
                    'strength': self.get_rsi_strength(current_rsi),
                    'momentum': self.calculate_rsi_momentum(rsi_series)
                }
                
            # Determine overall signal based on multiple timeframes
            overall_signal = self.get_multi_timeframe_consensus(timeframe_analysis)
            
            return {
                'timeframes': timeframe_analysis,
                'overall_signal': overall_signal['signal'],
                'overall_confidence': overall_signal['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing multi-timeframe RSI: {e}")
            return {'timeframes': {}, 'overall_signal': 'NONE', 'overall_confidence': 0}
            
    def get_multi_timeframe_consensus(self, timeframe_analysis: Dict) -> Dict:
        """Get consensus signal from multiple timeframes"""
        try:
            if not timeframe_analysis:
                return {'signal': 'NONE', 'confidence': 0}
                
            signals = []
            confidences = []
            
            # Weight timeframes (Daily > 4H > 1H)
            weights = {'Daily': 3, '4H': 2, '1H': 1}
            
            for timeframe, analysis in timeframe_analysis.items():
                weight = weights.get(timeframe, 1)
                signal = analysis['signal']
                confidence = analysis['confidence']
                
                if signal != 'NONE':
                    signals.extend([signal] * weight)
                    confidences.extend([confidence] * weight)
                    
            if not signals:
                return {'signal': 'NONE', 'confidence': 0}
                
            # Get most common signal
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            if buy_count > sell_count:
                consensus_signal = 'BUY'
                consensus_confidence = sum(confidences) / len(confidences)
            elif sell_count > buy_count:
                consensus_signal = 'SELL'
                consensus_confidence = sum(confidences) / len(confidences)
            else:
                consensus_signal = 'NONE'
                consensus_confidence = 0
                
            return {
                'signal': consensus_signal,
                'confidence': min(95, consensus_confidence * 1.2)  # Boost confidence for multi-timeframe agreement
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe consensus: {e}")
            return {'signal': 'NONE', 'confidence': 0}
