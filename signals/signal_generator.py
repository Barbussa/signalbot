"""
Signal generation module combining RSI and Fibonacci analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from data.data_provider import DataProvider
from analysis.rsi_analyzer import RSIAnalyzer
from analysis.fibonacci import FibonacciAnalysis
from analysis.technical_analysis import TechnicalAnalysis
from signals.risk_management import RiskManagement
from storage.signal_storage import SignalStorage
from utils.logger import setup_logger

logger = setup_logger()

class SignalGenerator:
    """Main signal generation class combining RSI and Fibonacci analysis"""
    
    def __init__(self, api_key: str):
        self.data_provider = DataProvider(api_key)
        self.rsi_analyzer = RSIAnalyzer()
        self.fibonacci_analyzer = FibonacciAnalysis()
        self.technical_analyzer = TechnicalAnalysis()
        self.risk_manager = RiskManagement()
        self.signal_storage = SignalStorage()
        
        # Signal generation parameters
        self.min_confidence = 60  # Minimum confidence level for signal generation
        self.symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
        
    async def generate_signals(self) -> List[Dict]:
        """Generate trading signals for all monitored symbols"""
        try:
            signals = []
            
            for symbol in self.symbols:
                logger.info(f"Generating signals for {symbol}")
                
                # Get multi-timeframe data
                timeframe_data = self.data_provider.get_multi_timeframe_data(symbol)
                
                if not timeframe_data:
                    logger.warning(f"No data available for {symbol}")
                    continue
                    
                # Generate signal for this symbol
                signal = await self.generate_symbol_signal(symbol, timeframe_data)
                
                if signal and signal['signal'] != 'NONE':
                    signals.append(signal)
                    
                    # Store signal in database
                    self.signal_storage.store_signal(signal)
                    
            logger.info(f"Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
            
    async def generate_symbol_signal(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Generate signal for a specific symbol"""
        try:
            if not timeframe_data:
                return None
                
            # Get current price
            current_price = self.data_provider.get_current_price(symbol)
            if current_price is None:
                return None
                
            # Analyze each timeframe
            timeframe_analysis = {}
            
            for timeframe, df in timeframe_data.items():
                if len(df) < 50:  # Need sufficient data
                    continue
                    
                # RSI Analysis
                rsi_analysis = self.rsi_analyzer.analyze_rsi_signal(df['close'], current_price)
                
                # Fibonacci Analysis
                fib_analysis = self.fibonacci_analyzer.analyze_fibonacci_signal(df['close'], current_price)
                
                # Technical Analysis
                trend = self.technical_analyzer.detect_trend(df['close'])
                atr = self.technical_analyzer.calculate_atr(df['high'], df['low'], df['close'])
                current_atr = atr.iloc[-1] if not atr.empty else current_price * 0.01
                
                timeframe_analysis[timeframe] = {
                    'rsi': rsi_analysis,
                    'fibonacci': fib_analysis,
                    'trend': trend,
                    'atr': current_atr,
                    'data': df
                }
                
            if not timeframe_analysis:
                return None
                
            # Generate combined signal
            combined_signal = self.combine_signals(symbol, current_price, timeframe_analysis)
            
            if combined_signal['signal'] != 'NONE' and combined_signal['confidence'] >= self.min_confidence:
                return combined_signal
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
            
    def combine_signals(self, symbol: str, current_price: float, timeframe_analysis: Dict) -> Dict:
        """Combine RSI and Fibonacci signals across timeframes"""
        try:
            # Initialize signal data
            signal_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'signal': 'NONE',
                'confidence': 0,
                'timeframe': '',
                'entry_price': current_price,
                'take_profit': 0,
                'stop_loss': 0,
                'risk_reward': 0,
                'position_size': 0,
                'rsi': 0,
                'fib_level': 0,
                'fib_distance': 0,
                'analysis': timeframe_analysis
            }
            
            # Analyze each timeframe and calculate weighted signals
            timeframe_signals = []
            timeframe_weights = {'Daily': 3, '4H': 2, '1H': 1}
            
            for timeframe, analysis in timeframe_analysis.items():
                rsi_analysis = analysis['rsi']
                fib_analysis = analysis['fibonacci']
                
                # Check if RSI and Fibonacci signals align
                rsi_signal = rsi_analysis['signal']
                fib_signal = fib_analysis['signal']
                
                if rsi_signal == fib_signal and rsi_signal != 'NONE':
                    # Signals align - calculate combined confidence
                    combined_confidence = (rsi_analysis['confidence'] + fib_analysis['confidence']) / 2
                    
                    # Check if price is near 0.618 Fibonacci level
                    if fib_analysis['near_618']:
                        combined_confidence += 15  # Boost confidence
                        
                    # Apply timeframe weight
                    weight = timeframe_weights.get(timeframe, 1)
                    weighted_confidence = combined_confidence * weight
                    
                    timeframe_signals.append({
                        'timeframe': timeframe,
                        'signal': rsi_signal,
                        'confidence': weighted_confidence,
                        'rsi_value': rsi_analysis['rsi'],
                        'fib_618': fib_analysis['fib_618'],
                        'fib_distance': fib_analysis['distance_to_618'],
                        'atr': analysis['atr']
                    })
                    
            if not timeframe_signals:
                return signal_data
                
            # Find the strongest signal
            strongest_signal = max(timeframe_signals, key=lambda x: x['confidence'])
            
            # Update signal data with strongest signal
            signal_data['signal'] = strongest_signal['signal']
            signal_data['timeframe'] = strongest_signal['timeframe']
            signal_data['confidence'] = min(95, strongest_signal['confidence'])
            signal_data['rsi'] = strongest_signal['rsi_value']
            signal_data['fib_level'] = strongest_signal['fib_618']
            signal_data['fib_distance'] = strongest_signal['fib_distance']
            
            # Calculate entry, take profit, and stop loss
            self.calculate_trade_levels(signal_data, strongest_signal)
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'signal': 'NONE', 'confidence': 0}
            
    def calculate_trade_levels(self, signal_data: Dict, strongest_signal: Dict):
        """Calculate entry, take profit, and stop loss levels"""
        try:
            symbol = signal_data['symbol']
            current_price = signal_data['current_price']
            signal_type = signal_data['signal']
            atr = strongest_signal['atr']
            
            # Get pip value for the symbol
            pip_value = self.data_provider.get_pip_value(symbol)
            
            # Calculate levels based on ATR and Fibonacci
            if signal_type == 'BUY':
                # For buy signals
                entry_price = current_price
                stop_loss = current_price - (atr * 1.5)  # 1.5 ATR stop loss
                take_profit = current_price + (atr * 3)   # 3 ATR take profit (2:1 R/R)
                
            elif signal_type == 'SELL':
                # For sell signals
                entry_price = current_price
                stop_loss = current_price + (atr * 1.5)  # 1.5 ATR stop loss
                take_profit = current_price - (atr * 3)   # 3 ATR take profit (2:1 R/R)
                
            else:
                return
                
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Calculate position size using risk management
            position_size = self.risk_manager.calculate_position_size(
                account_balance=10000,  # Default account balance
                risk_percent=2,         # 2% risk per trade
                entry_price=entry_price,
                stop_loss=stop_loss,
                pip_value=pip_value
            )
            
            # Update signal data
            signal_data['entry_price'] = entry_price
            signal_data['take_profit'] = take_profit
            signal_data['stop_loss'] = stop_loss
            signal_data['risk_reward'] = risk_reward
            signal_data['position_size'] = position_size
            
        except Exception as e:
            logger.error(f"Error calculating trade levels: {e}")
            
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a specific symbol and return detailed analysis"""
        try:
            # Get multi-timeframe data
            timeframe_data = self.data_provider.get_multi_timeframe_data(symbol)
            
            if not timeframe_data:
                return None
                
            # Get current price
            current_price = self.data_provider.get_current_price(symbol)
            if current_price is None:
                return None
                
            # Analyze each timeframe
            analysis_results = {}
            
            for timeframe, df in timeframe_data.items():
                if len(df) < 50:
                    continue
                    
                # RSI Analysis
                rsi_analysis = self.rsi_analyzer.analyze_rsi_signal(df['close'], current_price)
                
                # Fibonacci Analysis
                fib_analysis = self.fibonacci_analyzer.analyze_fibonacci_signal(df['close'], current_price)
                
                # Technical Analysis
                trend = self.technical_analyzer.detect_trend(df['close'])
                support_resistance = self.technical_analyzer.calculate_support_resistance(df['close'])
                
                analysis_results[timeframe] = {
                    'rsi': rsi_analysis['rsi'],
                    'rsi_signal': rsi_analysis['signal'],
                    'fib_618': fib_analysis.get('fib_618', 0),
                    'fib_distance': fib_analysis.get('distance_to_618', 0),
                    'trend': trend,
                    'support': support_resistance.get('support', 0),
                    'resistance': support_resistance.get('resistance', 0)
                }
                
            # Create summary analysis
            summary = self.create_analysis_summary(symbol, current_price, analysis_results)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
            
    def create_analysis_summary(self, symbol: str, current_price: float, analysis_results: Dict) -> Dict:
        """Create analysis summary from timeframe results"""
        try:
            # Get primary timeframe data (prefer Daily, then 4H, then 1H)
            primary_timeframe = None
            for tf in ['Daily', '4H', '1H']:
                if tf in analysis_results:
                    primary_timeframe = tf
                    break
                    
            if not primary_timeframe:
                return {}
                
            primary_data = analysis_results[primary_timeframe]
            
            # Calculate overall sentiment
            rsi_sentiment = "Bullish" if primary_data['rsi'] < 50 else "Bearish"
            trend_sentiment = primary_data['trend']
            
            # Determine overall sentiment
            if trend_sentiment == "UPTREND":
                sentiment = "Bullish"
            elif trend_sentiment == "DOWNTREND":
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
                
            # Calculate strength score
            strength = self.calculate_analysis_strength(analysis_results)
            
            # Generate recommendation
            recommendation = self.generate_recommendation(analysis_results, sentiment, strength)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'rsi_1h': analysis_results.get('1H', {}).get('rsi', 0),
                'rsi_4h': analysis_results.get('4H', {}).get('rsi', 0),
                'rsi_daily': analysis_results.get('Daily', {}).get('rsi', 0),
                'fib_618': primary_data.get('fib_618', 0),
                'distance_to_618': primary_data.get('fib_distance', 0),
                'sentiment': sentiment,
                'strength': strength,
                'recommendation': recommendation,
                'support': primary_data.get('support', 0),
                'resistance': primary_data.get('resistance', 0),
                'trend': trend_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return {}
            
    def calculate_analysis_strength(self, analysis_results: Dict) -> int:
        """Calculate analysis strength score (1-10)"""
        try:
            strength_score = 0
            max_score = 0
            
            for timeframe, data in analysis_results.items():
                # RSI strength
                rsi = data.get('rsi', 50)
                if rsi > 70 or rsi < 30:
                    strength_score += 2
                elif rsi > 60 or rsi < 40:
                    strength_score += 1
                max_score += 2
                
                # Trend strength
                trend = data.get('trend', 'SIDEWAYS')
                if trend in ['UPTREND', 'DOWNTREND']:
                    strength_score += 1
                max_score += 1
                
                # Fibonacci proximity
                fib_distance = data.get('fib_distance', 100)
                if fib_distance < 20:
                    strength_score += 2
                elif fib_distance < 50:
                    strength_score += 1
                max_score += 2
                
            if max_score > 0:
                return min(10, int((strength_score / max_score) * 10))
            else:
                return 5
                
        except Exception as e:
            logger.error(f"Error calculating analysis strength: {e}")
            return 5
            
    def generate_recommendation(self, analysis_results: Dict, sentiment: str, strength: int) -> str:
        """Generate trading recommendation based on analysis"""
        try:
            if strength >= 7:
                if sentiment == "Bullish":
                    return "🟢 STRONG BUY - Multiple indicators suggest bullish momentum"
                elif sentiment == "Bearish":
                    return "🔴 STRONG SELL - Multiple indicators suggest bearish momentum"
                else:
                    return "⚪ NEUTRAL - Mixed signals, wait for clearer direction"
            elif strength >= 5:
                if sentiment == "Bullish":
                    return "🟡 WEAK BUY - Some bullish signals, proceed with caution"
                elif sentiment == "Bearish":
                    return "🟡 WEAK SELL - Some bearish signals, proceed with caution"
                else:
                    return "⚪ NEUTRAL - Insufficient signals for clear direction"
            else:
                return "⚪ HOLD - Weak signals, avoid trading until clearer setup"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "⚪ ANALYSIS ERROR - Unable to generate recommendation"
            
    async def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            return self.data_provider.get_market_status()
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'state': 'UNKNOWN', 'last_update': 'N/A'}
