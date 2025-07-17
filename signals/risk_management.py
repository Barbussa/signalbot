"""
Risk management module for position sizing and trade management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from utils.logger import setup_logger

logger = setup_logger()

class RiskManagement:
    """Risk management class for calculating position sizes and managing risk"""
    
    def __init__(self):
        self.default_risk_percent = 2  # 2% risk per trade
        self.max_risk_percent = 5      # Maximum 5% risk per trade
        self.min_risk_reward = 1.5     # Minimum 1.5:1 risk/reward ratio
        self.max_position_size = 10    # Maximum 10% of account per position
        
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss: float, pip_value: float) -> float:
        """Calculate position size based on risk parameters"""
        try:
            # Validate inputs
            if account_balance <= 0 or risk_percent <= 0 or entry_price <= 0:
                return 0
                
            # Limit risk percentage
            risk_percent = min(risk_percent, self.max_risk_percent)
            
            # Calculate risk amount in account currency
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate stop loss distance in pips
            stop_loss_pips = abs(entry_price - stop_loss) / pip_value
            
            if stop_loss_pips == 0:
                return 0
                
            # Calculate position size
            # Position size = Risk amount / (Stop loss in pips * pip value * lot size)
            # For forex, standard lot size is 100,000 units
            lot_size = 100000
            position_size = risk_amount / (stop_loss_pips * pip_value * lot_size)
            
            # Convert to percentage of account
            position_size_percent = min(position_size * 100, self.max_position_size)
            
            return position_size_percent
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
            
    def calculate_risk_reward_ratio(self, entry_price: float, take_profit: float, stop_loss: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            if entry_price == 0:
                return 0
                
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return 0
                
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward ratio: {e}")
            return 0
            
    def validate_trade_setup(self, entry_price: float, take_profit: float, stop_loss: float) -> Dict[str, bool]:
        """Validate trade setup against risk management rules"""
        try:
            validation = {
                'valid': True,
                'has_stop_loss': False,
                'has_take_profit': False,
                'good_risk_reward': False,
                'reasonable_levels': False
            }
            
            # Check if stop loss is set
            if stop_loss != 0 and stop_loss != entry_price:
                validation['has_stop_loss'] = True
            else:
                validation['valid'] = False
                
            # Check if take profit is set
            if take_profit != 0 and take_profit != entry_price:
                validation['has_take_profit'] = True
            else:
                validation['valid'] = False
                
            # Check risk/reward ratio
            if validation['has_stop_loss'] and validation['has_take_profit']:
                risk_reward = self.calculate_risk_reward_ratio(entry_price, take_profit, stop_loss)
                if risk_reward >= self.min_risk_reward:
                    validation['good_risk_reward'] = True
                else:
                    validation['valid'] = False
                    
            # Check if levels are reasonable (not too close or too far)
            if validation['has_stop_loss'] and validation['has_take_profit']:
                stop_distance = abs(entry_price - stop_loss) / entry_price
                profit_distance = abs(take_profit - entry_price) / entry_price
                
                # Stop loss should be between 0.1% and 5% of price
                # Take profit should be between 0.2% and 10% of price
                if (0.001 <= stop_distance <= 0.05) and (0.002 <= profit_distance <= 0.10):
                    validation['reasonable_levels'] = True
                else:
                    validation['valid'] = False
                    
            return validation
            
        except Exception as e:
            logger.error(f"Error validating trade setup: {e}")
            return {'valid': False}
            
    def calculate_maximum_loss(self, account_balance: float, position_size_percent: float, 
                             entry_price: float, stop_loss: float) -> float:
        """Calculate maximum potential loss for a trade"""
        try:
            position_value = account_balance * (position_size_percent / 100)
            loss_percent = abs(entry_price - stop_loss) / entry_price
            maximum_loss = position_value * loss_percent
            
            return maximum_loss
            
        except Exception as e:
            logger.error(f"Error calculating maximum loss: {e}")
            return 0
            
    def calculate_maximum_profit(self, account_balance: float, position_size_percent: float, 
                               entry_price: float, take_profit: float) -> float:
        """Calculate maximum potential profit for a trade"""
        try:
            position_value = account_balance * (position_size_percent / 100)
            profit_percent = abs(take_profit - entry_price) / entry_price
            maximum_profit = position_value * profit_percent
            
            return maximum_profit
            
        except Exception as e:
            logger.error(f"Error calculating maximum profit: {e}")
            return 0
            
    def adjust_position_size_for_volatility(self, base_position_size: float, 
                                          volatility: float, avg_volatility: float) -> float:
        """Adjust position size based on market volatility"""
        try:
            if avg_volatility == 0:
                return base_position_size
                
            volatility_ratio = volatility / avg_volatility
            
            # Reduce position size in high volatility, increase in low volatility
            if volatility_ratio > 1.5:
                # High volatility - reduce position size
                adjusted_size = base_position_size * 0.7
            elif volatility_ratio > 1.2:
                # Moderate high volatility
                adjusted_size = base_position_size * 0.85
            elif volatility_ratio < 0.8:
                # Low volatility - can increase position size slightly
                adjusted_size = base_position_size * 1.1
            else:
                # Normal volatility
                adjusted_size = base_position_size
                
            # Ensure we don't exceed maximum position size
            return min(adjusted_size, self.max_position_size)
            
        except Exception as e:
            logger.error(f"Error adjusting position size for volatility: {e}")
            return base_position_size
            
    def calculate_correlation_adjustment(self, existing_positions: List[Dict], 
                                       new_symbol: str) -> float:
        """Calculate position size adjustment based on correlation with existing positions"""
        try:
            # Simple correlation mapping (in practice, this would use historical correlation data)
            correlation_map = {
                ('EURUSD', 'GBPUSD'): 0.7,
                ('EURUSD', 'XAUUSD'): -0.3,
                ('GBPUSD', 'XAUUSD'): -0.2
            }
            
            if not existing_positions:
                return 1.0  # No adjustment needed
                
            total_correlation = 0
            position_count = 0
            
            for position in existing_positions:
                existing_symbol = position.get('symbol', '')
                
                # Check correlation between symbols
                correlation_key = tuple(sorted([existing_symbol, new_symbol]))
                correlation = correlation_map.get(correlation_key, 0)
                
                if abs(correlation) > 0.5:  # Significant correlation
                    total_correlation += abs(correlation)
                    position_count += 1
                    
            if position_count > 0:
                avg_correlation = total_correlation / position_count
                # Reduce position size based on correlation
                adjustment = 1 - (avg_correlation * 0.3)  # Max 30% reduction
                return max(0.5, adjustment)  # Minimum 50% of original size
                
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0
            
    def get_risk_metrics(self, account_balance: float, position_size_percent: float, 
                        entry_price: float, take_profit: float, stop_loss: float) -> Dict:
        """Get comprehensive risk metrics for a trade"""
        try:
            metrics = {}
            
            # Basic calculations
            metrics['position_size_percent'] = position_size_percent
            metrics['risk_reward_ratio'] = self.calculate_risk_reward_ratio(entry_price, take_profit, stop_loss)
            metrics['maximum_loss'] = self.calculate_maximum_loss(account_balance, position_size_percent, entry_price, stop_loss)
            metrics['maximum_profit'] = self.calculate_maximum_profit(account_balance, position_size_percent, entry_price, take_profit)
            
            # Risk percentages
            metrics['risk_percent'] = (metrics['maximum_loss'] / account_balance) * 100
            metrics['profit_percent'] = (metrics['maximum_profit'] / account_balance) * 100
            
            # Trade validation
            metrics['validation'] = self.validate_trade_setup(entry_price, take_profit, stop_loss)
            
            # Risk assessment
            if metrics['risk_percent'] <= 2:
                metrics['risk_level'] = 'LOW'
            elif metrics['risk_percent'] <= 5:
                metrics['risk_level'] = 'MODERATE'
            else:
                metrics['risk_level'] = 'HIGH'
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def optimize_levels(self, entry_price: float, atr: float, signal_type: str) -> Dict[str, float]:
        """Optimize take profit and stop loss levels using ATR"""
        try:
            if signal_type == 'BUY':
                # For buy signals
                stop_loss = entry_price - (atr * 1.5)
                take_profit_1 = entry_price + (atr * 2)
                take_profit_2 = entry_price + (atr * 3)
                take_profit_3 = entry_price + (atr * 4)
                
            elif signal_type == 'SELL':
                # For sell signals
                stop_loss = entry_price + (atr * 1.5)
                take_profit_1 = entry_price - (atr * 2)
                take_profit_2 = entry_price - (atr * 3)
                take_profit_3 = entry_price - (atr * 4)
                
            else:
                return {}
                
            return {
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3
            }
            
        except Exception as e:
            logger.error(f"Error optimizing levels: {e}")
            return {}
