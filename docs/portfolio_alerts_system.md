# Portfolio Alert System Documentation

## Overview

The Portfolio Alert System (`src/portfolio/alerts.py`) is a comprehensive alert management system that cross-references new trading signals with user portfolio holdings and generates contextual alerts via multiple notification channels.

## Key Features

### 1. **Intelligent Alert Generation**
- **Position-Based Alerts**: Automatic alerts for signals affecting your current holdings
- **Signal Cross-Referencing**: Matches trading signals with portfolio positions
- **Contextual Messages**: Alerts include P&L, position size, and actionable recommendations
- **Priority Classification**: CRITICAL, HIGH, MEDIUM, LOW priority levels

### 2. **Multi-Channel Notifications**
- **Email**: Rich HTML emails with detailed signal information
- **Telegram**: Instant mobile notifications via Telegram bot
- **SMS**: SMS support (integration ready for Twilio/similar)
- **Template-Based**: Customizable alert message templates

### 3. **Alert Types**

| Alert Type | Description | Example |
|------------|-------------|---------|
| `POSITION_SIGNAL` | Signal for current holding | "BUY signal for your holding: THYAO" |
| `STRONG_SELL_HOLDING` | Critical sell signal for holding | "⚠️ URGENT: Strong Sell Signal for GARAN" |
| `SELL_HOLDING` | Sell signal for holding | "⚠️ Sell Signal for Your Holding - AKBNK" |
| `STRONG_BUY_OPPORTUNITY` | Strong buy for non-holding | "🎯 Strong Buy Opportunity - EREGL" |
| `BUY_OPPORTUNITY` | Buy signal for non-holding | "📈 Buy Opportunity - SAHOL" |
| `WATCHLIST_SIGNAL` | Signal for watchlist stock | Alert for stocks you're watching |
| `RISK_WARNING` | High risk warning | Position risk alerts |

### 4. **Alert Rules System**
- **Configurable Rules**: Customize when alerts are generated
- **Confidence Thresholds**: Set minimum confidence levels
- **Signal Filtering**: Choose which signal types trigger alerts
- **Throttling**: Prevent alert spam with time-based throttling
- **Custom Priorities**: Define priority levels per rule

### 5. **Alert Management**
- **History Tracking**: Complete audit trail of all alerts
- **Deduplication**: Intelligent throttling prevents duplicate alerts
- **Summary Statistics**: Analytics on alert patterns
- **Batch Processing**: Efficient processing of multiple signals

## Installation & Setup

### 1. Dependencies
The alert system requires the following packages (already in your project):
- `pandas`
- `numpy`
- Standard library: `smtplib`, `email`, `json`, `logging`

Optional for Telegram:
- `requests` (for Telegram API)

### 2. Environment Configuration

Create or update your `.env` file with notification settings:

```bash
# Email Notifications (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=recipient@example.com

# Telegram Notifications
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 3. Gmail Setup (for Email Alerts)

If using Gmail:
1. Enable 2-factor authentication on your Google account
2. Generate an "App Password" for the trading system
3. Use the app password in `SMTP_PASSWORD`

### 4. Telegram Setup (for Mobile Alerts)

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Start a chat with your bot
4. Get your chat ID using: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`

## Usage Examples

### Basic Usage

```python
from src.portfolio.manager import create_portfolio
from src.portfolio.alerts import create_alert_manager

# Create portfolio with holdings
portfolio = create_portfolio(
    name="My BIST Portfolio",
    initial_cash=100000.0
)
portfolio.buy(symbol="THYAO", shares=100, price=250.0)
portfolio.buy(symbol="GARAN", shares=200, price=85.0)

# Create alert manager
alert_manager = create_alert_manager(
    portfolio_manager=portfolio,
    watchlist=['EREGL', 'AKBNK', 'SAHOL']
)

# Process new signals
signals = [
    {
        'stock_code': 'THYAO',
        'signal': 'STRONG_SELL',
        'confidence_score': 85.0,
        'current_price': 265.0,
        'target_price': 245.0,
        'expected_return': -0.075
    }
]

alerts = alert_manager.process_signals(signals)

# Send notifications
alert_manager.send_alerts(alerts)
```

### Advanced: Custom Alert Rules

```python
from src.portfolio.alerts import AlertRule, AlertType, AlertPriority, NotificationChannel, SignalDirection

custom_rules = [
    # Critical: Strong sell for large positions
    AlertRule(
        rule_id="critical_sell_large_position",
        name="Critical Sell - Large Position",
        alert_type=AlertType.STRONG_SELL_HOLDING,
        min_confidence=70.0,
        signal_types=[SignalDirection.STRONG_SELL],
        only_holdings=True,
        channels=[NotificationChannel.EMAIL, NotificationChannel.TELEGRAM, NotificationChannel.SMS],
        priority=AlertPriority.CRITICAL,
        throttle_hours=6  # Allow more frequent alerts for critical signals
    ),

    # High: Strong buy opportunities
    AlertRule(
        rule_id="strong_buy_opportunity",
        name="High Confidence Buy",
        alert_type=AlertType.STRONG_BUY_OPPORTUNITY,
        min_confidence=80.0,
        signal_types=[SignalDirection.STRONG_BUY],
        only_holdings=False,
        channels=[NotificationChannel.EMAIL, NotificationChannel.TELEGRAM],
        priority=AlertPriority.HIGH,
        throttle_hours=24
    )
]

alert_manager = create_alert_manager(
    portfolio_manager=portfolio,
    alert_rules=custom_rules
)
```

### Integration with Signal Generator

```python
from src.signals.generator import create_signal_generator, create_model_output
import pandas as pd

# Create signal generator
signal_generator = create_signal_generator()

# Generate signals from model outputs
model_outputs = [
    create_model_output('lstm', 'regression', prediction=245.0, confidence=0.85),
    create_model_output('random_forest', 'classification', prediction=0, confidence=0.80),
    create_model_output('sentiment', 'nlp', prediction=-0.7, confidence=0.75)
]

signal = signal_generator.generate_signal(
    stock_code='THYAO',
    model_outputs=model_outputs,
    current_price=265.0,
    historical_prices=pd.Series([270, 268, 265, 263, 265] * 10)
)

# Process with alert manager
alerts = alert_manager.process_signals([signal])
alert_manager.send_alerts(alerts)
```

## Alert Message Examples

### Example 1: Strong Sell for Holding

```
⚠️ URGENT: Strong Sell Signal for Your Holding - THYAO

A STRONG_SELL signal has been generated for THYAO, which you currently hold in your portfolio.

Signal Confidence: 85.0%

Your Position: 100 shares
Current P&L: +1,500.00 TRY (+6.00%)

Current Price: 265.00 TRY
Target Price: 245.00 TRY
Expected Return: -7.55%

Suggested Stop Loss: 251.75 TRY
Suggested Take Profit: 258.25 TRY

⚠️ Recommendation: Consider selling your position to protect gains/minimize losses.

Analysis: Multiple models indicate downward trend with high confidence.
```

### Example 2: Strong Buy Opportunity

```
🎯 Strong Buy Opportunity - EREGL

A STRONG_BUY signal has been generated for EREGL.

Signal Confidence: 85.0%

Current Price: 45.00 TRY
Target Price: 50.00 TRY
Expected Return: +11.11%

Suggested Stop Loss: 42.75 TRY
Suggested Take Profit: 48.00 TRY

✅ Recommendation: Strong buying opportunity. Consider opening a position.

Analysis: Strong fundamentals and positive sentiment across all models.
```

## Alert Manager Methods

### Core Methods

| Method | Description |
|--------|-------------|
| `process_signals(signals, current_prices)` | Process signals and generate alerts |
| `send_alerts(alerts, channels)` | Send alerts via specified channels |
| `add_to_watchlist(symbol)` | Add symbol to watchlist |
| `remove_from_watchlist(symbol)` | Remove symbol from watchlist |
| `get_alert_history(symbol, start_date, limit)` | Retrieve alert history |
| `get_alert_summary()` | Get summary statistics |
| `clear_alert_history()` | Clear alert history |

### Alert Object Properties

| Property | Type | Description |
|----------|------|-------------|
| `alert_id` | str | Unique identifier |
| `alert_type` | AlertType | Type of alert |
| `priority` | AlertPriority | Priority level |
| `symbol` | str | Stock symbol |
| `signal_direction` | SignalDirection | Trading signal |
| `confidence_score` | float | Confidence (0-100) |
| `title` | str | Alert title |
| `message` | str | Alert message |
| `position_size` | float | Shares held (if applicable) |
| `position_value` | float | Position value |
| `unrealized_pnl` | float | Unrealized P&L |
| `target_price` | float | Target price |
| `current_price` | float | Current price |
| `expected_return` | float | Expected return % |

## Notification Channels

### Email (SMTP)
- Rich HTML formatting
- Includes all signal details
- Color-coded priority badges
- Professional layout

**Configuration Required:**
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `NOTIFICATION_EMAIL`

### Telegram
- Instant mobile notifications
- Markdown formatting
- Works with Telegram bot API

**Configuration Required:**
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### SMS (Ready for Integration)
- Short summary messages
- Twilio integration template included
- Requires API credentials

## Alert Throttling

Alert throttling prevents notification spam by limiting how frequently the same symbol can trigger alerts:

- **CRITICAL alerts**: 6-12 hour throttle (configurable)
- **HIGH alerts**: 24 hour throttle
- **MEDIUM/LOW alerts**: 24-48 hour throttle

Override throttling for urgent signals by setting `throttle_hours=0` in alert rules.

## Best Practices

### 1. **Configure Alert Rules Carefully**
```python
# Start conservative, then adjust
alert_rules = [
    AlertRule(
        min_confidence=75.0,  # Start high
        throttle_hours=24,    # Prevent spam
        priority=AlertPriority.HIGH
    )
]
```

### 2. **Monitor Alert Effectiveness**
```python
# Regularly review alert history
summary = alert_manager.get_alert_summary()
print(f"Total alerts: {summary['total_alerts']}")
print(f"By priority: {summary['alerts_by_priority']}")

# Identify noise
most_alerted = summary['most_alerted_symbols']
# Adjust rules for frequently-alerted symbols
```

### 3. **Use Multiple Channels Wisely**
- **CRITICAL**: Email + Telegram + SMS
- **HIGH**: Email + Telegram
- **MEDIUM**: Email only
- **LOW**: In-app/logging only

### 4. **Integrate with Scheduler**
```python
# Run alert system on schedule
from src.signals.scheduler import SignalScheduler

scheduler = SignalScheduler()
scheduler.schedule_job(
    func=lambda: alert_manager.process_signals(get_latest_signals()),
    interval_minutes=30
)
```

## Integration Points

The alert system integrates with:

1. **Portfolio Manager** (`src/portfolio/manager.py`)
   - Retrieves current holdings
   - Calculates P&L for alerts
   - Position information

2. **Signal Generator** (`src/signals/generator.py`)
   - Receives trading signals
   - Uses confidence scores
   - Accesses signal metadata

3. **Signal Scheduler** (`src/signals/scheduler.py`)
   - Automated signal processing
   - Scheduled alert generation
   - Background monitoring

4. **Database** (`src/data/storage/database.py`)
   - Can store alerts in database
   - Alert history persistence
   - Analytics queries

## Troubleshooting

### Emails Not Sending

1. **Check SMTP configuration**
   ```python
   print(alert_manager.smtp_config)
   ```

2. **Test SMTP connection**
   ```python
   import smtplib
   server = smtplib.SMTP('smtp.gmail.com', 587)
   server.starttls()
   server.login('user', 'password')
   ```

3. **Check firewall/security**
   - Gmail: Enable "Less secure app access" or use App Password
   - Corporate email: Check firewall rules

### Telegram Not Working

1. **Verify bot token**
   ```bash
   curl https://api.telegram.org/bot<TOKEN>/getMe
   ```

2. **Get chat ID**
   - Send message to bot
   - Check: `https://api.telegram.org/bot<TOKEN>/getUpdates`

3. **Check dependencies**
   ```bash
   pip install requests
   ```

### No Alerts Generated

1. **Check alert rules**
   ```python
   for rule in alert_manager.alert_rules:
       print(f"{rule.name}: enabled={rule.enabled}, min_conf={rule.min_confidence}")
   ```

2. **Verify signal confidence meets threshold**
3. **Check throttling**
   ```python
   print(alert_manager.last_alert_time)
   ```

## Performance Considerations

- **Alert History**: Periodically clear old alerts
  ```python
  alert_manager.clear_alert_history()
  ```

- **Batch Processing**: Process signals in batches for efficiency
- **Throttling**: Prevents excessive API calls to notification services
- **Async Notifications**: Consider async sending for large alert volumes

## Security Notes

- **Never commit `.env` file** with credentials
- **Use app-specific passwords** for email
- **Rotate Telegram bot tokens** periodically
- **Validate alert content** to prevent injection attacks
- **Limit notification recipients** to authorized users

## Future Enhancements

Planned features:
- [ ] Push notifications (iOS/Android)
- [ ] WhatsApp integration
- [ ] Voice call alerts for critical signals
- [ ] Alert templates editor (web UI)
- [ ] Machine learning for alert optimization
- [ ] Alert acknowledgment tracking
- [ ] Multi-language support
- [ ] Custom alert webhooks

## Support

For issues or questions:
- Check the example: `/examples/portfolio_alerts_example.py`
- Review logs: Check logger output for debugging
- Test individual components: Run unit tests
- Check dependencies: Ensure all packages installed

## Complete Example

See `/home/user/BISTML/examples/portfolio_alerts_example.py` for a comprehensive working example that demonstrates all features of the alert system.
