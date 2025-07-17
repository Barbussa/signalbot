"""
Market scanner module for continuous signal monitoring and generation
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from signals.signal_generator import SignalGenerator
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger()

class MarketScanner:
    """Market scanner for continuous signal monitoring"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.signal_generator = SignalGenerator(api_key)
        self.config = Config()
        
        # Scanner settings
        self.scan_interval = self.config.get('bot.update_interval', 300)  # 5 minutes default
        self.max_signals_per_day = self.config.get('trading.max_daily_signals', 5)
        self.symbols = self.config.get('trading.symbols', ['XAUUSD', 'EURUSD', 'GBPUSD'])
        
        # State management
        self.running = False
        self.last_scan_time = None
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Signal tracking
        self.recent_signals = []
        self.signal_cooldown = {}  # Track cooldown periods for each symbol
        self.min_signal_interval = 1800  # 30 minutes minimum between signals for same symbol
        
        # Callback for sending signals
        self.signal_callback = None
        
    def set_signal_callback(self, callback):
        """Set callback function for when signals are generated"""
        self.signal_callback = callback
        
    async def start_scanning(self):
        """Start the market scanning process"""
        try:
            self.running = True
            logger.info("Market scanner started")
            
            while self.running:
                try:
                    await self.scan_markets()
                    await asyncio.sleep(self.scan_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Market scanner stopped by user")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in market scanning loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in market scanner: {e}")
            
        finally:
            self.running = False
            logger.info("Market scanner stopped")
            
    async def stop_scanning(self):
        """Stop the market scanning process"""
        self.running = False
        logger.info("Market scanner stop requested")
        
    async def scan_markets(self):
        """Scan markets for trading signals"""
        try:
            current_time = datetime.now()
            
            # Reset daily signal count if it's a new day
            if current_time.date() != self.last_reset_date:
                self.daily_signal_count = 0
                self.last_reset_date = current_time.date()
                logger.info("Daily signal count reset")
                
            # Check if we've reached the daily limit
            if self.daily_signal_count >= self.max_signals_per_day:
                logger.info(f"Daily signal limit reached ({self.max_signals_per_day})")
                return
                
            # Check market status
            market_status = await self.signal_generator.get_market_status()
            if market_status.get('state') != 'OPEN':
                logger.info("Market is closed, skipping scan")
                return
                
            logger.info("Starting market scan...")
            
            # Generate signals for all symbols
            signals = await self.signal_generator.generate_signals()
            
            # Process generated signals
            for signal in signals:
                await self.process_signal(signal)
                
            self.last_scan_time = current_time
            logger.info(f"Market scan completed. Found {len(signals)} signals")
            
        except Exception as e:
            logger.error(f"Error scanning markets: {e}")
            
    async def process_signal(self, signal: Dict):
        """Process a generated signal"""
        try:
            symbol = signal.get('symbol', '')
            signal_type = signal.get('signal', '')
            confidence = signal.get('confidence', 0)
            
            # Check if signal meets quality criteria
            if not self.validate_signal_quality(signal):
                logger.info(f"Signal quality check failed for {symbol}")
                return
                
            # Check cooldown period for this symbol
            if self.is_symbol_in_cooldown(symbol):
                logger.info(f"Symbol {symbol} is in cooldown period")
                return
                
            # Check for duplicate signals
            if self.is_duplicate_signal(signal):
                logger.info(f"Duplicate signal detected for {symbol}")
                return
                
            # Process valid signal
            await self.send_signal_notification(signal)
            
            # Update tracking
            self.recent_signals.append(signal)
            self.signal_cooldown[symbol] = datetime.now()
            self.daily_signal_count += 1
            
            logger.info(f"Signal processed: {symbol} {signal_type} (confidence: {confidence}%)")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            
    def validate_signal_quality(self, signal: Dict) -> bool:
        """Validate signal quality before sending"""
        try:
            # Check minimum confidence
            min_confidence = self.config.get('trading.min_confidence', 60)
            if signal.get('confidence', 0) < min_confidence:
                return False
                
            # Check risk/reward ratio
            risk_reward = signal.get('risk_reward', 0)
            min_risk_reward = self.config.get('risk_management.min_risk_reward', 1.5)
            if risk_reward < min_risk_reward:
                return False
                
            # Check if all required fields are present
            required_fields = ['symbol', 'signal', 'entry_price', 'take_profit', 'stop_loss']
            if not all(field in signal and signal[field] for field in required_fields):
                return False
                
            # Check if signal is for a supported symbol
            if signal.get('symbol') not in self.symbols:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal quality: {e}")
            return False
            
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        try:
            if symbol not in self.signal_cooldown:
                return False
                
            last_signal_time = self.signal_cooldown[symbol]
            time_since_last = (datetime.now() - last_signal_time).total_seconds()
            
            return time_since_last < self.min_signal_interval
            
        except Exception as e:
            logger.error(f"Error checking cooldown for {symbol}: {e}")
            return False
            
    def is_duplicate_signal(self, new_signal: Dict) -> bool:
        """Check if this is a duplicate of a recent signal"""
        try:
            symbol = new_signal.get('symbol', '')
            signal_type = new_signal.get('signal', '')
            entry_price = new_signal.get('entry_price', 0)
            
            # Check recent signals from last 2 hours
            cutoff_time = datetime.now() - timedelta(hours=2)
            
            for recent_signal in self.recent_signals:
                try:
                    signal_time = datetime.fromisoformat(recent_signal.get('timestamp', ''))
                    if signal_time < cutoff_time:
                        continue
                        
                    # Check if same symbol and signal type
                    if (recent_signal.get('symbol') == symbol and 
                        recent_signal.get('signal') == signal_type):
                        
                        # Check if entry prices are very close (within 0.1%)
                        recent_entry = recent_signal.get('entry_price', 0)
                        if recent_entry > 0:
                            price_diff = abs(entry_price - recent_entry) / recent_entry
                            if price_diff < 0.001:  # 0.1%
                                return True
                                
                except Exception:
                    continue
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate signal: {e}")
            return False
            
    async def send_signal_notification(self, signal: Dict):
        """Send signal notification via callback"""
        try:
            if self.signal_callback:
                await self.signal_callback(signal)
            else:
                logger.warning("No signal callback set")
                
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")
            
    def cleanup_old_signals(self):
        """Clean up old signals from memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Clean up recent signals
            self.recent_signals = [
                signal for signal in self.recent_signals
                if datetime.fromisoformat(signal.get('timestamp', '')) > cutoff_time
            ]
            
            # Clean up cooldown tracking
            symbols_to_remove = []
            for symbol, last_time in self.signal_cooldown.items():
                if (datetime.now() - last_time).total_seconds() > self.min_signal_interval:
                    symbols_to_remove.append(symbol)
                    
            for symbol in symbols_to_remove:
                del self.signal_cooldown[symbol]
                
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
            
    def get_scanner_status(self) -> Dict:
        """Get current scanner status"""
        try:
            return {
                'running': self.running,
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'daily_signal_count': self.daily_signal_count,
                'max_daily_signals': self.max_signals_per_day,
                'scan_interval': self.scan_interval,
                'symbols_monitored': len(self.symbols),
                'symbols_in_cooldown': len(self.signal_cooldown),
                'recent_signals_count': len(self.recent_signals)
            }
            
        except Exception as e:
            logger.error(f"Error getting scanner status: {e}")
            return {}
            
    def update_settings(self, settings: Dict):
        """Update scanner settings"""
        try:
            if 'scan_interval' in settings:
                self.scan_interval = settings['scan_interval']
                logger.info(f"Scan interval updated to {self.scan_interval} seconds")
                
            if 'max_signals_per_day' in settings:
                self.max_signals_per_day = settings['max_signals_per_day']
                logger.info(f"Max daily signals updated to {self.max_signals_per_day}")
                
            if 'symbols' in settings:
                self.symbols = settings['symbols']
                logger.info(f"Monitored symbols updated to {self.symbols}")
                
            if 'min_signal_interval' in settings:
                self.min_signal_interval = settings['min_signal_interval']
                logger.info(f"Min signal interval updated to {self.min_signal_interval} seconds")
                
        except Exception as e:
            logger.error(f"Error updating scanner settings: {e}")
            
    async def force_scan(self) -> List[Dict]:
        """Force an immediate market scan"""
        try:
            logger.info("Forcing immediate market scan")
            
            # Temporarily bypass daily limit for manual scan
            original_limit = self.max_signals_per_day
            self.max_signals_per_day = 999
            
            # Perform scan
            signals = await self.signal_generator.generate_signals()
            
            # Restore original limit
            self.max_signals_per_day = original_limit
            
            logger.info(f"Force scan completed. Found {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in force scan: {e}")
            return []
            
    async def test_connection(self) -> bool:
        """Test connection to data provider"""
        try:
            # Test market status endpoint
            market_status = await self.signal_generator.get_market_status()
            
            if market_status and 'state' in market_status:
                logger.info("Connection test successful")
                return True
            else:
                logger.error("Connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False
            
    def get_statistics(self) -> Dict:
        """Get scanner statistics"""
        try:
            # Calculate uptime
            uptime = 0
            if self.last_scan_time:
                uptime = (datetime.now() - self.last_scan_time).total_seconds()
                
            # Count signals by symbol
            signal_counts = {}
            for signal in self.recent_signals:
                symbol = signal.get('symbol', 'UNKNOWN')
                signal_counts[symbol] = signal_counts.get(symbol, 0) + 1
                
            return {
                'uptime_seconds': uptime,
                'total_scans': len(self.recent_signals),
                'daily_signals': self.daily_signal_count,
                'signals_by_symbol': signal_counts,
                'average_confidence': self.calculate_average_confidence(),
                'success_rate': self.calculate_success_rate()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
            
    def calculate_average_confidence(self) -> float:
        """Calculate average confidence of recent signals"""
        try:
            if not self.recent_signals:
                return 0.0
                
            total_confidence = sum(signal.get('confidence', 0) for signal in self.recent_signals)
            return total_confidence / len(self.recent_signals)
            
        except Exception as e:
            logger.error(f"Error calculating average confidence: {e}")
            return 0.0
            
    def calculate_success_rate(self) -> float:
        """Calculate success rate of recent signals (placeholder)"""
        try:
            # This would require tracking signal outcomes
            # For now, return a placeholder value
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.0
