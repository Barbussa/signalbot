# Telegram Trading Signal Bot

A sophisticated Telegram trading signal bot that provides automated technical analysis for multiple currency pairs and commodities, leveraging advanced indicators like RSI and Fibonacci levels.

## Features

- **Multi-currency support**: XAUUSD, EURUSD, GBPUSD
- **Automated technical analysis** using RSI and Fibonacci retracement
- **Real-time signal generation** with confidence levels
- **Multi-timeframe data processing** (1H, 4H, Daily)
- **Risk management** with position sizing calculations
- **Integrated Telegram bot interface** for easy interaction
- **Rate limiting** to comply with API restrictions
- **Fallback data system** for demonstration purposes

## Installation

### Requirements

- Python 3.11+
- Telegram Bot Token
- Alpha Vantage API Key

### Setup

1. Install dependencies:
```bash
pip install python-telegram-bot requests pandas numpy python-dotenv
```

2. Create a `.env` file in the project root with your API keys:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

3. Run the bot:
```bash
python main.py
```

## Project Structure

```
trading_bot/
├── main.py                    # Main application entry point
├── config/
│   └── settings.json         # Configuration settings
├── bot/
│   ├── telegram_bot.py       # Telegram bot implementation
│   └── commands.py           # Bot command handlers
├── signals/
│   ├── signal_generator.py   # Main signal generation logic
│   └── risk_management.py    # Risk management calculations
├── analysis/
│   ├── technical_analysis.py # Technical indicator calculations
│   ├── rsi_analyzer.py       # RSI analysis implementation
│   └── fibonacci.py          # Fibonacci retracement analysis
├── data/
│   ├── data_provider.py      # Alpha Vantage API integration
│   └── performance.json      # Performance tracking data
├── storage/
│   └── signal_storage.py     # Signal storage and retrieval
├── scheduler/
│   └── market_scanner.py     # Market monitoring and scanning
└── utils/
    ├── config.py             # Configuration management
    └── logger.py             # Logging setup
```

## Usage

### Telegram Commands

- `/start` - Initialize the bot and get welcome message
- `/help` - Show available commands
- `/status` - Check bot status and statistics
- `/signals` - View recent trading signals
- `/subscribe` - Subscribe to signal notifications
- `/unsubscribe` - Unsubscribe from notifications
- `/analyze [SYMBOL]` - Perform technical analysis on a symbol
- `/settings` - Configure bot settings

### Configuration

Edit `config/settings.json` to customize:
- Trading symbols to monitor
- Risk management parameters
- Signal generation criteria
- Update intervals
- Technical indicator settings

## Technical Analysis

The bot uses a combination of:
- **RSI (Relative Strength Index)** for momentum analysis
- **Fibonacci retracement levels** for support/resistance
- **Multiple timeframe analysis** for signal confirmation
- **Risk/reward ratio calculations** for position sizing

## API Integration

- **Alpha Vantage API** for real-time market data
- **Telegram Bot API** for user interaction
- **Rate limiting** implemented for API compliance

## Error Handling

- Comprehensive logging system
- Graceful error handling with fallback mechanisms
- Automatic retry logic for API failures
- Signal validation and quality checks

## Development

The application follows a modular architecture with:
- **Presentation Layer**: Telegram bot interface
- **Business Logic Layer**: Signal generation and analysis
- **Data Access Layer**: Market data provider and storage
- **External Services**: Alpha Vantage API integration

## License

This project is for educational and demonstration purposes.

## Support

For issues or questions, please check the logs in the `logs/` directory for detailed error information.