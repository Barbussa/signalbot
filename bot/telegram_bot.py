"""
Telegram bot implementation for trading signals
"""

import asyncio
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.commands import BotCommands
from signals.signal_generator import SignalGenerator
from storage.signal_storage import SignalStorage
from utils.logger import setup_logger

logger = setup_logger()

class TradingSignalBot:
    """Main Telegram bot class for trading signals"""
    
    def __init__(self, token: str, alpha_vantage_key: str):
        self.token = token
        self.alpha_vantage_key = alpha_vantage_key
        self.application = Application.builder().token(token).build()
        self.signal_generator = SignalGenerator(alpha_vantage_key)
        self.storage = SignalStorage()
        self.commands = BotCommands(self.signal_generator, self.storage)
        self.subscribers = set()
        
        # Setup handlers
        self._setup_handlers()
        
        # Set up bot data for commands
        self.application.bot_data['add_subscriber'] = self.add_subscriber
        self.application.bot_data['remove_subscriber'] = self.remove_subscriber
        
    def _setup_handlers(self):
        """Setup command and callback handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.commands.start_command))
        self.application.add_handler(CommandHandler("help", self.commands.help_command))
        self.application.add_handler(CommandHandler("status", self.commands.status_command))
        self.application.add_handler(CommandHandler("signals", self.commands.signals_command))
        self.application.add_handler(CommandHandler("subscribe", self.commands.subscribe_command))
        self.application.add_handler(CommandHandler("unsubscribe", self.commands.unsubscribe_command))
        self.application.add_handler(CommandHandler("analyze", self.commands.analyze_command))
        self.application.add_handler(CommandHandler("settings", self.commands.settings_command))
        
        # Callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self.commands.button_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
        
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
    async def start(self):
        """Start the bot"""
        logger.info("Starting Telegram bot...")
        try:
            # Use the simpler run_polling method which handles the event loop internally
            await self.application.run_polling(drop_pending_updates=True)
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
        
    async def stop(self):
        """Stop the bot"""
        logger.info("Stopping Telegram bot...")
        try:
            # Stop the application gracefully
            await self.application.stop()
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            # Don't raise here to allow graceful shutdown
        
    async def send_signal_to_subscribers(self, signal_data: dict):
        """Send trading signal to all subscribers"""
        if not self.subscribers:
            logger.info("No subscribers to send signal to")
            return
            
        message = self._format_signal_message(signal_data)
        
        for chat_id in self.subscribers.copy():
            try:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Failed to send signal to {chat_id}: {e}")
                # Remove invalid chat_id
                self.subscribers.discard(chat_id)
                
    def _format_signal_message(self, signal_data: dict) -> str:
        """Format signal data into readable message"""
        signal_type = "🟢 BUY" if signal_data['signal'] == 'BUY' else "🔴 SELL"
        
        message = f"""
🚨 <b>TRADING SIGNAL</b> 🚨

<b>Pair:</b> {signal_data['symbol']}
<b>Signal:</b> {signal_type}
<b>Timeframe:</b> {signal_data['timeframe']}
<b>Entry Price:</b> {signal_data['entry_price']:.5f}

📊 <b>Technical Analysis:</b>
• RSI: {signal_data['rsi']:.2f}
• Fibonacci Level: {signal_data['fib_level']:.3f}
• Confidence: {signal_data['confidence']:.1f}%

🎯 <b>Targets:</b>
• Take Profit: {signal_data['take_profit']:.5f}
• Stop Loss: {signal_data['stop_loss']:.5f}

⚠️ <b>Risk Management:</b>
• Risk/Reward: 1:{signal_data['risk_reward']:.2f}
• Suggested Position Size: {signal_data['position_size']:.2f}%

<i>Generated at: {signal_data['timestamp']}</i>
"""
        return message
        
    def add_subscriber(self, chat_id: int):
        """Add subscriber to signal notifications"""
        self.subscribers.add(chat_id)
        logger.info(f"Added subscriber: {chat_id}")
        
    def remove_subscriber(self, chat_id: int):
        """Remove subscriber from signal notifications"""
        self.subscribers.discard(chat_id)
        logger.info(f"Removed subscriber: {chat_id}")
