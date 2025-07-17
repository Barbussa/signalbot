"""
Signal storage module for storing and retrieving trading signals
"""

import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from utils.logger import setup_logger

logger = setup_logger()

class SignalStorage:
    """Signal storage class for managing trading signals"""
    
    def __init__(self, storage_dir: str = 'data'):
        self.storage_dir = storage_dir
        self.signals_file = os.path.join(storage_dir, 'signals.csv')
        self.performance_file = os.path.join(storage_dir, 'performance.json')
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
        
    def _initialize_files(self):
        """Initialize storage files if they don't exist"""
        try:
            # Initialize signals CSV file
            if not os.path.exists(self.signals_file):
                with open(self.signals_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'symbol', 'signal', 'timeframe', 'confidence',
                        'entry_price', 'take_profit', 'stop_loss', 'risk_reward',
                        'position_size', 'rsi', 'fib_level', 'fib_distance', 'status'
                    ])
                    
            # Initialize performance JSON file
            if not os.path.exists(self.performance_file):
                initial_performance = {
                    'total_signals': 0,
                    'winning_signals': 0,
                    'losing_signals': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'net_profit': 0.0,
                    'average_winner': 0.0,
                    'average_loser': 0.0,
                    'largest_winner': 0.0,
                    'largest_loser': 0.0,
                    'profit_factor': 0.0,
                    'last_updated': datetime.now().isoformat()
                }
                
                with open(self.performance_file, 'w') as f:
                    json.dump(initial_performance, f, indent=4)
                    
        except Exception as e:
            logger.error(f"Error initializing storage files: {e}")
            
    def store_signal(self, signal_data: Dict) -> bool:
        """Store a trading signal"""
        try:
            # Prepare signal data for CSV
            row_data = [
                signal_data.get('timestamp', datetime.now().isoformat()),
                signal_data.get('symbol', ''),
                signal_data.get('signal', ''),
                signal_data.get('timeframe', ''),
                signal_data.get('confidence', 0),
                signal_data.get('entry_price', 0),
                signal_data.get('take_profit', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('risk_reward', 0),
                signal_data.get('position_size', 0),
                signal_data.get('rsi', 0),
                signal_data.get('fib_level', 0),
                signal_data.get('fib_distance', 0),
                'OPEN'  # Default status
            ]
            
            # Write to CSV file
            with open(self.signals_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
                
            logger.info(f"Signal stored: {signal_data['symbol']} {signal_data['signal']}")
            
            # Update performance statistics
            self._update_performance_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return False
            
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent trading signals"""
        try:
            signals = []
            
            if not os.path.exists(self.signals_file):
                return signals
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                all_signals = list(reader)
                
            # Get the most recent signals
            recent_signals = all_signals[-limit:] if len(all_signals) > limit else all_signals
            
            # Convert to proper format
            for signal in reversed(recent_signals):
                signals.append({
                    'timestamp': signal['timestamp'],
                    'symbol': signal['symbol'],
                    'signal': signal['signal'],
                    'timeframe': signal['timeframe'],
                    'confidence': float(signal['confidence']) if signal['confidence'] else 0,
                    'entry_price': float(signal['entry_price']) if signal['entry_price'] else 0,
                    'take_profit': float(signal['take_profit']) if signal['take_profit'] else 0,
                    'stop_loss': float(signal['stop_loss']) if signal['stop_loss'] else 0,
                    'risk_reward': float(signal['risk_reward']) if signal['risk_reward'] else 0,
                    'position_size': float(signal['position_size']) if signal['position_size'] else 0,
                    'rsi': float(signal['rsi']) if signal['rsi'] else 0,
                    'fib_level': float(signal['fib_level']) if signal['fib_level'] else 0,
                    'fib_distance': float(signal['fib_distance']) if signal['fib_distance'] else 0,
                    'status': signal['status']
                })
                
            return signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
            
    def get_signals_by_symbol(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get signals for a specific symbol"""
        try:
            signals = []
            
            if not os.path.exists(self.signals_file):
                return signals
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for signal in reader:
                    if signal['symbol'] == symbol:
                        signals.append({
                            'timestamp': signal['timestamp'],
                            'signal': signal['signal'],
                            'timeframe': signal['timeframe'],
                            'confidence': float(signal['confidence']) if signal['confidence'] else 0,
                            'entry_price': float(signal['entry_price']) if signal['entry_price'] else 0,
                            'take_profit': float(signal['take_profit']) if signal['take_profit'] else 0,
                            'stop_loss': float(signal['stop_loss']) if signal['stop_loss'] else 0,
                            'status': signal['status']
                        })
                        
            # Return most recent signals first
            return signals[-limit:] if len(signals) > limit else signals
            
        except Exception as e:
            logger.error(f"Error getting signals for symbol {symbol}: {e}")
            return []
            
    def get_signals_count_today(self) -> int:
        """Get count of signals generated today"""
        try:
            today = datetime.now().date()
            count = 0
            
            if not os.path.exists(self.signals_file):
                return count
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for signal in reader:
                    try:
                        signal_date = datetime.fromisoformat(signal['timestamp']).date()
                        if signal_date == today:
                            count += 1
                    except ValueError:
                        continue
                        
            return count
            
        except Exception as e:
            logger.error(f"Error getting today's signal count: {e}")
            return 0
            
    def get_total_signals(self) -> int:
        """Get total number of signals generated"""
        try:
            count = 0
            
            if not os.path.exists(self.signals_file):
                return count
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                count = sum(1 for _ in reader)
                
            return count
            
        except Exception as e:
            logger.error(f"Error getting total signals count: {e}")
            return 0
            
    def update_signal_status(self, timestamp: str, symbol: str, status: str, 
                           profit_loss: Optional[float] = None) -> bool:
        """Update signal status (e.g., HIT_TP, HIT_SL, CLOSED)"""
        try:
            # Read existing signals
            signals = []
            
            if not os.path.exists(self.signals_file):
                return False
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                signals = list(reader)
                
            # Update the specific signal
            updated = False
            for signal in signals:
                if signal['timestamp'] == timestamp and signal['symbol'] == symbol:
                    signal['status'] = status
                    if profit_loss is not None:
                        signal['profit_loss'] = profit_loss
                    updated = True
                    break
                    
            if not updated:
                return False
                
            # Write back to CSV
            with open(self.signals_file, 'w', newline='') as f:
                fieldnames = signals[0].keys() if signals else []
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(signals)
                
            logger.info(f"Signal status updated: {symbol} {timestamp} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return False
            
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            if not os.path.exists(self.performance_file):
                return {}
                
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
                
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
            
    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            stats = {
                'total_signals': 0,
                'winning_signals': 0,
                'losing_signals': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'average_winner': 0.0,
                'average_loser': 0.0,
                'largest_winner': 0.0,
                'largest_loser': 0.0,
                'profit_factor': 0.0,
                'last_updated': datetime.now().isoformat()
            }
            
            if not os.path.exists(self.signals_file):
                return
                
            # Calculate statistics from signals
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                
                winners = []
                losers = []
                
                for signal in reader:
                    stats['total_signals'] += 1
                    
                    if signal['status'] == 'HIT_TP':
                        stats['winning_signals'] += 1
                        if 'profit_loss' in signal and signal['profit_loss']:
                            profit = float(signal['profit_loss'])
                            winners.append(profit)
                            stats['total_profit'] += profit
                            
                    elif signal['status'] == 'HIT_SL':
                        stats['losing_signals'] += 1
                        if 'profit_loss' in signal and signal['profit_loss']:
                            loss = float(signal['profit_loss'])
                            losers.append(loss)
                            stats['total_loss'] += abs(loss)
                            
            # Calculate derived statistics
            if stats['total_signals'] > 0:
                stats['win_rate'] = (stats['winning_signals'] / stats['total_signals']) * 100
                
            if winners:
                stats['average_winner'] = sum(winners) / len(winners)
                stats['largest_winner'] = max(winners)
                
            if losers:
                stats['average_loser'] = sum(losers) / len(losers)
                stats['largest_loser'] = min(losers)
                
            stats['net_profit'] = stats['total_profit'] - stats['total_loss']
            
            if stats['total_loss'] > 0:
                stats['profit_factor'] = stats['total_profit'] / stats['total_loss']
                
            # Save updated statistics
            with open(self.performance_file, 'w') as f:
                json.dump(stats, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
            
    def cleanup_old_signals(self, days_to_keep: int = 30):
        """Clean up old signals to keep storage size manageable"""
        try:
            if not os.path.exists(self.signals_file):
                return
                
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Read all signals
            signals = []
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                signals = list(reader)
                
            # Filter signals to keep
            filtered_signals = []
            for signal in signals:
                try:
                    signal_date = datetime.fromisoformat(signal['timestamp'])
                    if signal_date > cutoff_date:
                        filtered_signals.append(signal)
                except ValueError:
                    # Keep signals with invalid timestamps
                    filtered_signals.append(signal)
                    
            # Write back filtered signals
            if filtered_signals:
                with open(self.signals_file, 'w', newline='') as f:
                    fieldnames = filtered_signals[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(filtered_signals)
                    
            removed_count = len(signals) - len(filtered_signals)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old signals")
                
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
            
    def export_signals(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> List[Dict]:
        """Export signals for a specific date range"""
        try:
            signals = []
            
            if not os.path.exists(self.signals_file):
                return signals
                
            with open(self.signals_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for signal in reader:
                    try:
                        signal_date = datetime.fromisoformat(signal['timestamp'])
                        
                        # Check date range
                        if start_date:
                            start_dt = datetime.fromisoformat(start_date)
                            if signal_date < start_dt:
                                continue
                                
                        if end_date:
                            end_dt = datetime.fromisoformat(end_date)
                            if signal_date > end_dt:
                                continue
                                
                        signals.append(dict(signal))
                        
                    except ValueError:
                        # Include signals with invalid timestamps
                        signals.append(dict(signal))
                        
            return signals
            
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return []
