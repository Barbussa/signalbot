"""
Telegram bot command handlers
"""

import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from signals.signal_generator import SignalGenerator
from storage.signal_storage import SignalStorage
from utils.logger import setup_logger

logger = setup_logger()

class BotCommands:
    """Handles all bot commands"""
    
    def __init__(self, signal_generator: SignalGenerator, storage: SignalStorage):
        self.signal_generator = signal_generator
        self.storage = storage
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🤖 <b>Welcome to XAUUSD & Forex Trading Signal Bot</b>

This bot provides automated trading signals for:
• XAUUSD (Gold)
• EUR/USD
• GBP/USD

<b>Features:</b>
📊 Multi-timeframe analysis (1H, 4H, Daily)
📈 RSI + Fibonacci 0.618 level analysis
🎯 Automated TP/SL recommendations
⚡ Real-time signal notifications

<b>Commands:</b>
/help - Show all commands
/status - Check bot status
/signals - View recent signals
/subscribe - Subscribe to signal notifications
/analyze [SYMBOL] - Analyze specific pair
/settings - Bot configuration

<i>Use /help for detailed command information</i>
"""
        await update.message.reply_text(welcome_message, parse_mode='HTML')
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
📋 <b>Bot Commands Help</b>

<b>Basic Commands:</b>
/start - Initialize bot
/help - Show this help message
/status - Check bot and market status

<b>Signal Commands:</b>
/signals - View last 10 signals
/subscribe - Subscribe to live signals
/unsubscribe - Unsubscribe from signals
/analyze [SYMBOL] - Analyze specific pair
  Example: /analyze XAUUSD

<b>Configuration:</b>
/settings - View/modify bot settings

<b>Supported Pairs:</b>
• XAUUSD (Gold/USD)
• EUR/USD (Euro/Dollar)
• GBP/USD (Pound/Dollar)

<b>Analysis Parameters:</b>
• RSI Period: 14
• Fibonacci Levels: 0.618 (primary trigger)
• Timeframes: 1H, 4H, Daily
• Risk/Reward: Minimum 1:2

<i>For technical support or questions, contact the administrator.</i>
"""
        await update.message.reply_text(help_message, parse_mode='HTML')
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Check market status and bot health
            market_status = await self.signal_generator.get_market_status()
            
            status_message = f"""
🟢 <b>Bot Status: ACTIVE</b>

📊 <b>Market Status:</b>
• Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Market State: {market_status.get('state', 'Unknown')}
• Last Update: {market_status.get('last_update', 'N/A')}

📈 <b>Monitoring:</b>
• XAUUSD: ✅ Active
• EUR/USD: ✅ Active  
• GBP/USD: ✅ Active

🔍 <b>Analysis Status:</b>
• RSI Calculation: ✅ Working
• Fibonacci Levels: ✅ Working
• Signal Generation: ✅ Working

📊 <b>Recent Activity:</b>
• Signals Today: {self.storage.get_signals_count_today()}
• Total Signals: {self.storage.get_total_signals()}

<i>Last system check: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await update.message.reply_text(status_message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error checking status. Please try again later.")
            
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        try:
            recent_signals = self.storage.get_recent_signals(limit=10)
            
            if not recent_signals:
                await update.message.reply_text("📭 No recent signals found.")
                return
                
            message = "📊 <b>Recent Trading Signals</b>\n\n"
            
            for signal in recent_signals:
                signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴"
                message += f"""
{signal_emoji} <b>{signal['symbol']}</b> - {signal['signal']}
Entry: {signal['entry_price']:.5f}
TP: {signal['take_profit']:.5f} | SL: {signal['stop_loss']:.5f}
Time: {signal['timestamp']}
---
"""
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await update.message.reply_text("❌ Error retrieving signals. Please try again later.")
            
    async def subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscribe command"""
        chat_id = update.message.chat_id
        
        # Add to subscribers via callback
        if hasattr(context, 'bot_data') and 'add_subscriber' in context.bot_data:
            context.bot_data['add_subscriber'](chat_id)
        
        await update.message.reply_text("""
✅ <b>Successfully subscribed to trading signals!</b>

You will now receive:
🔔 Real-time signal notifications
📊 Market analysis updates
⚡ Entry/Exit recommendations

<i>Use /unsubscribe to stop receiving notifications</i>
""", parse_mode='HTML')
        
    async def unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unsubscribe command"""
        chat_id = update.message.chat_id
        
        # Remove from subscribers via callback
        if hasattr(context, 'bot_data') and 'remove_subscriber' in context.bot_data:
            context.bot_data['remove_subscriber'](chat_id)
        
        await update.message.reply_text("""
❌ <b>Successfully unsubscribed from trading signals</b>

You will no longer receive:
🔕 Signal notifications
📊 Market updates

<i>Use /subscribe to resume notifications</i>
""", parse_mode='HTML')
        
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        try:
            if not context.args:
                await update.message.reply_text("❌ Please specify a symbol. Example: /analyze XAUUSD")
                return
                
            symbol = context.args[0].upper()
            
            if symbol not in ['XAUUSD', 'EURUSD', 'GBPUSD']:
                await update.message.reply_text("❌ Unsupported symbol. Use: XAUUSD, EURUSD, or GBPUSD")
                return
                
            # Show typing indicator
            await update.message.reply_text("🔍 Analyzing market data...")
            
            # Perform analysis
            analysis = await self.signal_generator.analyze_symbol(symbol)
            
            if not analysis:
                await update.message.reply_text("❌ Unable to analyze symbol. Please try again later.")
                return
                
            # Format analysis message
            message = f"""
📊 <b>Technical Analysis - {symbol}</b>

<b>Current Price:</b> {analysis['current_price']:.5f}

<b>RSI Analysis:</b>
• 1H RSI: {analysis['rsi_1h']:.2f}
• 4H RSI: {analysis['rsi_4h']:.2f}
• Daily RSI: {analysis['rsi_daily']:.2f}

<b>Fibonacci Levels:</b>
• 0.618 Level: {analysis['fib_618']:.5f}
• Distance to 0.618: {analysis['distance_to_618']:.1f} pips

<b>Market Sentiment:</b>
• Overall: {analysis['sentiment']}
• Strength: {analysis['strength']}/10

<b>Recommendation:</b>
{analysis['recommendation']}

<i>Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in analyze command: {e}")
            await update.message.reply_text("❌ Error performing analysis. Please try again later.")
            
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        keyboard = [
            [InlineKeyboardButton("RSI Settings", callback_data='settings_rsi')],
            [InlineKeyboardButton("Fibonacci Settings", callback_data='settings_fib')],
            [InlineKeyboardButton("Risk Management", callback_data='settings_risk')],
            [InlineKeyboardButton("Timeframes", callback_data='settings_timeframes')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        settings_message = """
⚙️ <b>Bot Settings</b>

<b>Current Configuration:</b>
• RSI Period: 14
• RSI Overbought: 70
• RSI Oversold: 30
• Fibonacci Level: 0.618
• Default Risk %: 2%
• Min Risk/Reward: 1:2

<b>Active Timeframes:</b>
• 1 Hour ✅
• 4 Hour ✅
• Daily ✅

<i>Click buttons below to modify settings</i>
"""
        
        await update.message.reply_text(
            settings_message, 
            parse_mode='HTML',
            reply_markup=reply_markup
        )
        
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data.startswith('settings_'):
            setting_type = query.data.split('_')[1]
            
            if setting_type == 'rsi':
                message = """
📊 <b>RSI Settings</b>

<b>Current Settings:</b>
• Period: 14
• Overbought Level: 70
• Oversold Level: 30

<b>Recommendations:</b>
• Period: 14 (standard)
• Overbought: 70-80
• Oversold: 20-30

<i>These settings are optimized for forex trading</i>
"""
            elif setting_type == 'fib':
                message = """
📐 <b>Fibonacci Settings</b>

<b>Current Settings:</b>
• Primary Level: 0.618 (Golden Ratio)
• Secondary Levels: 0.382, 0.5, 0.786

<b>Usage:</b>
• 0.618 level is the primary trigger
• Price action at this level combined with RSI generates signals

<i>0.618 is statistically the most reliable retracement level</i>
"""
            elif setting_type == 'risk':
                message = """
⚠️ <b>Risk Management Settings</b>

<b>Current Settings:</b>
• Default Risk per Trade: 2%
• Minimum Risk/Reward: 1:2
• Maximum Position Size: 5%

<b>Safety Features:</b>
• Automatic position sizing
• Stop loss always calculated
• Risk/reward validation

<i>These settings help protect your capital</i>
"""
            elif setting_type == 'timeframes':
                message = """
⏰ <b>Timeframe Settings</b>

<b>Active Timeframes:</b>
• 1 Hour ✅ (Short-term scalping)
• 4 Hour ✅ (Medium-term swings)
• Daily ✅ (Long-term trends)

<b>Analysis Method:</b>
• Multiple timeframe confirmation
• Higher timeframe bias
• Lower timeframe entry

<i>All timeframes are analyzed for signal validation</i>
"""
            else:
                message = "❌ Unknown setting type"
                
            await query.edit_message_text(text=message, parse_mode='HTML')
