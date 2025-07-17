"""
Configuration management module
"""

import json
import os
from typing import Dict, Any, Optional
import logging

from utils.logger import setup_logger

logger = setup_logger()

class Config:
    """Configuration management class"""
    
    def __init__(self, config_file: str = 'config/settings.json'):
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Configuration loaded from {self.config_file}")
                    return config
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                return self.get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "bot": {
                "name": "XAUUSD Forex Trading Bot",
                "version": "1.0.0",
                "update_interval": 300,
                "max_signals_per_day": 10
            },
            "trading": {
                "symbols": ["XAUUSD", "EURUSD", "GBPUSD"],
                "timeframes": ["1H", "4H", "Daily"],
                "min_confidence": 60,
                "max_daily_signals": 5
            },
            "rsi": {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "fibonacci": {
                "primary_level": 0.618,
                "secondary_levels": [0.236, 0.382, 0.5, 0.786],
                "proximity_threshold": 20
            },
            "risk_management": {
                "default_risk_percent": 2,
                "max_risk_percent": 5,
                "min_risk_reward": 1.5,
                "max_position_size": 10
            },
            "alerts": {
                "send_notifications": True,
                "notification_types": ["signal", "error", "status"],
                "max_alerts_per_hour": 10
            },
            "data": {
                "api_timeout": 30,
                "retry_attempts": 3,
                "rate_limit_delay": 12
            }
        }
        
    def save_config(self) -> bool:
        """Save configuration to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            logger.error(f"Error getting configuration key {key}: {e}")
            return default
            
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by key (supports nested keys with dot notation)"""
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
                
            # Set the value
            config[keys[-1]] = value
            
            # Save configuration
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error setting configuration key {key}: {e}")
            return False
            
    def update(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        try:
            for key, value in updates.items():
                self.set(key, value)
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
            
    def get_bot_config(self) -> Dict[str, Any]:
        """Get bot-specific configuration"""
        return self.get('bot', {})
        
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return self.get('trading', {})
        
    def get_rsi_config(self) -> Dict[str, Any]:
        """Get RSI configuration"""
        return self.get('rsi', {})
        
    def get_fibonacci_config(self) -> Dict[str, Any]:
        """Get Fibonacci configuration"""
        return self.get('fibonacci', {})
        
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get('risk_management', {})
        
    def get_alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration"""
        return self.get('alerts', {})
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get('data', {})
        
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration values"""
        validation = {
            'valid': True,
            'errors': []
        }
        
        try:
            # Validate RSI settings
            rsi_period = self.get('rsi.period', 14)
            if not isinstance(rsi_period, int) or rsi_period < 1:
                validation['errors'].append('RSI period must be a positive integer')
                validation['valid'] = False
                
            # Validate risk management settings
            risk_percent = self.get('risk_management.default_risk_percent', 2)
            if not isinstance(risk_percent, (int, float)) or risk_percent <= 0 or risk_percent > 100:
                validation['errors'].append('Risk percent must be between 0 and 100')
                validation['valid'] = False
                
            # Validate symbols
            symbols = self.get('trading.symbols', [])
            if not isinstance(symbols, list) or not symbols:
                validation['errors'].append('At least one trading symbol must be specified')
                validation['valid'] = False
                
            # Validate timeframes
            timeframes = self.get('trading.timeframes', [])
            valid_timeframes = ['1H', '4H', 'Daily']
            if not isinstance(timeframes, list) or not all(tf in valid_timeframes for tf in timeframes):
                validation['errors'].append(f'Timeframes must be from: {valid_timeframes}')
                validation['valid'] = False
                
        except Exception as e:
            validation['errors'].append(f'Configuration validation error: {e}')
            validation['valid'] = False
            
        return validation
