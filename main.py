#!/usr/bin/env python3
"""
Telegram Trading Signal Bot for XAUUSD and Forex
Main application entry point
"""

import os
import sys
import logging
import asyncio
import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot.telegram_bot import TradingSignalBot
from scheduler.market_scanner import MarketScanner
from utils.logger import setup_logger
from utils.config import Config

# Setup logging
logger = setup_logger()

# Global variables for graceful shutdown
bot_instance = None
scanner_instance = None

async def shutdown_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    
    if scanner_instance:
        await scanner_instance.stop_scanning()
    
    if bot_instance:
        await bot_instance.stop()
    
    logger.info("Shutdown complete")

async def main():
    """Main application entry point"""
    global bot_instance, scanner_instance
    
    try:
        # Load configuration
        config = Config()
        
        # Get required environment variables
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if not telegram_token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
            return
            
        if not alpha_vantage_key:
            logger.error("ALPHA_VANTAGE_API_KEY environment variable is required")
            return
        
        # Initialize bot
        bot_instance = TradingSignalBot(telegram_token, alpha_vantage_key)
        
        # Initialize market scanner
        scanner_instance = MarketScanner(alpha_vantage_key)
        
        # Set up callback for scanner to send signals to bot
        scanner_instance.set_signal_callback(bot_instance.send_signal_to_subscribers)
        
        # Start market scanning in background
        scanner_task = asyncio.create_task(scanner_instance.start_scanning())
        
        # Start bot
        logger.info("Starting Telegram Trading Signal Bot...")
        
        # Run both tasks concurrently
        await asyncio.gather(
            bot_instance.start(),
            scanner_task,
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Don't re-raise in production
        logger.exception("Full error traceback:")
    finally:
        # Ensure cleanup
        if scanner_instance:
            await scanner_instance.stop_scanning()
        if bot_instance:
            await bot_instance.stop()

def run_bot():
    """Run the bot with proper event loop handling"""
    try:
        # Set up asyncio event loop policy for better compatibility
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the main function
        asyncio.run(main())
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        logger.exception("Full error traceback:")

if __name__ == "__main__":
    run_bot()
