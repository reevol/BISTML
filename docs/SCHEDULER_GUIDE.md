# BIST Signal Scheduler Guide

Complete guide for using the automated signal generation scheduler for Borsa Istanbul (BIST) trading hours.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Market Hours](#market-hours)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Holiday Management](#holiday-management)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## Overview

The BIST Signal Scheduler is a robust, production-ready scheduling system that automatically generates trading signals during Borsa Istanbul trading hours. It handles:

- **30-minute interval signals**: Runs at :00 and :30 past each hour (10:00 - 17:30)
- **Hourly signals**: Runs at the top of each hour (10:00 - 17:00)
- **Market hours validation**: Automatically skips execution when market is closed
- **Holiday handling**: Respects Turkish public holidays
- **Error recovery**: Robust retry logic and error handling

## Features

### Core Features

1. **Intelligent Scheduling**
   - APScheduler-based with timezone awareness (Europe/Istanbul)
   - Automatic market hours detection
   - Holiday calendar integration
   - Missed job handling with coalescing

2. **Market Hours Detection**
   - Regular trading hours: 10:00 - 18:00 Turkey Time
   - Weekend detection (no trading Saturday/Sunday)
   - Turkish public holiday detection
   - Optional pre-market and post-market support

3. **Flexible Signal Generation**
   - Support for multiple intervals (30m, 1h)
   - Custom signal generation functions
   - Market open/close routines
   - Manual trigger capability

4. **Robust Error Handling**
   - Automatic retry on failure
   - Job execution history tracking
   - Detailed logging
   - Graceful shutdown

5. **Monitoring and Observability**
   - Job execution tracking
   - Performance metrics
   - Market status reporting
   - Schedule visualization

## Installation

### Prerequisites

```bash
# Required packages
pip install apscheduler>=3.10.0 pandas>=2.0.0
```

The scheduler is already included in the project's `requirements.txt`.

### Verify Installation

```python
from src.signals.scheduler import SignalScheduler, BISTMarketHours

# Check market status
is_open = BISTMarketHours.is_market_open()
print(f"Market is {'open' if is_open else 'closed'}")
```

## Quick Start

### Basic Usage

```python
from src.signals.scheduler import create_default_scheduler

# Define your signal generation functions
def generate_30m_signals():
    print("Generating 30-minute signals...")
    # Your signal generation logic here
    return {"signals": [...]}

def generate_hourly_signals():
    print("Generating hourly signals...")
    # Your signal generation logic here
    return {"signals": [...]}

# Create and start scheduler
scheduler = create_default_scheduler(
    signal_generator_30m=generate_30m_signals,
    signal_generator_1h=generate_hourly_signals,
    run_on_start=True  # Run immediately if market is open
)

scheduler.start()

# Keep running
try:
    import time
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    scheduler.stop()
```

### Using the Example Script

```bash
# Run with default settings
python examples/run_signal_scheduler.py

# Run in daemon mode
python examples/run_signal_scheduler.py --daemon

# Dry run (testing without actual signal generation)
python examples/run_signal_scheduler.py --dry-run

# Trigger immediate signal generation
python examples/run_signal_scheduler.py --immediate

# Check market status
python examples/run_signal_scheduler.py --status
```

## Market Hours

### BIST Trading Schedule

| Session | Time (Turkey) | Description |
|---------|---------------|-------------|
| Pre-market | 09:40 - 10:00 | Order collection (optional) |
| Regular | 10:00 - 18:00 | Main trading session |
| Post-market | 18:00 - 18:10 | Final trades (optional) |

### Trading Days

- **Active**: Monday - Friday
- **Closed**: Saturday, Sunday, Turkish public holidays

### Example: Check Market Status

```python
from src.signals.scheduler import BISTMarketHours
from datetime import datetime

# Check if market is open now
is_open = BISTMarketHours.is_market_open()
print(f"Market is open: {is_open}")

# Check if today is a trading day
is_trading_day = BISTMarketHours.is_trading_day()
print(f"Today is a trading day: {is_trading_day}")

# Get time until market close
minutes_remaining = BISTMarketHours.get_trading_minutes_remaining()
print(f"Minutes until close: {minutes_remaining}")

# Get next market open time
next_open = BISTMarketHours.get_next_market_open()
print(f"Next market open: {next_open}")
```

## Configuration

### Configuration File

Edit `configs/scheduler_config.yaml` to customize behavior:

```yaml
scheduler:
  run_on_start: true
  enable_extended_hours: false
  timezone: "Europe/Istanbul"

intervals:
  thirty_minute:
    enabled: true
  hourly:
    enabled: true

execution:
  coalesce: true
  max_instances: 1
  misfire_grace_time: 300
  max_retries: 3

logging:
  level: "INFO"
  log_file: "${LOG_DIR}/scheduler.log"
```

### Environment Variables

Set these in your `.env` file:

```bash
# Signal generation frequency
SIGNAL_FREQUENCY=30

# Logging
LOG_LEVEL=INFO
LOG_DIR=/home/user/BISTML/logs

# Notifications (optional)
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email@gmail.com
NOTIFICATION_EMAIL=alerts@yourdomain.com

TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

## Usage Examples

### Example 1: Custom Schedule

```python
from src.signals.scheduler import SignalScheduler

# Create scheduler without default schedules
scheduler = SignalScheduler()

# Add custom schedule - every 15 minutes during trading hours
scheduler.scheduler.add_job(
    func=my_custom_function,
    trigger='cron',
    day_of_week='mon-fri',
    hour='10-17',
    minute='*/15',  # Every 15 minutes
    timezone=BISTMarketHours.TURKEY_TZ
)

scheduler.start()
```

### Example 2: Market Open/Close Hooks

```python
def on_market_open():
    print("Market opened! Running initialization...")
    # - Update overnight news
    # - Check global markets
    # - Load fresh data

def on_market_close():
    print("Market closed! Running cleanup...")
    # - Generate daily report
    # - Update models
    # - Backup data

scheduler = create_default_scheduler(...)
scheduler.add_market_open_job(on_market_open)
scheduler.add_market_close_job(on_market_close)
```

### Example 3: Manual Trigger

```python
# Manually trigger signal generation
scheduler.trigger_manual_run(interval='30m')  # 30-minute signals
scheduler.trigger_manual_run(interval='1h')   # Hourly signals
```

### Example 4: Pause/Resume

```python
# Temporarily pause scheduling
scheduler.pause()
print("Scheduler paused")

# Resume scheduling
scheduler.resume()
print("Scheduler resumed")
```

### Example 5: Job History

```python
# Get recent job execution history
history = scheduler.get_job_history(limit=20)
print(history)

# Output:
#    job_id  scheduled_time      execution_time      success
# 0  signal_30m  2024-11-16 10:00  2024-11-16 10:00:05  True
# 1  signal_30m  2024-11-16 10:30  2024-11-16 10:30:03  True
# 2  signal_1h   2024-11-16 11:00  2024-11-16 11:00:08  True
```

## API Reference

### SignalScheduler Class

#### Constructor

```python
SignalScheduler(
    signal_generator_30m: Callable = None,
    signal_generator_1h: Callable = None,
    run_on_start: bool = False,
    enable_extended_hours: bool = False
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Start the scheduler |
| `stop(wait=True)` | Stop the scheduler |
| `pause()` | Pause job execution |
| `resume()` | Resume job execution |
| `add_30m_schedule()` | Add 30-minute interval job |
| `add_hourly_schedule()` | Add hourly interval job |
| `add_market_open_job(func)` | Add job at market open |
| `add_market_close_job(func)` | Add job at market close |
| `trigger_manual_run(interval)` | Manually trigger signal generation |
| `print_schedule()` | Print current schedule |
| `get_job_history(limit)` | Get job execution history |
| `get_market_status()` | Get current market status |

### BISTMarketHours Class

#### Class Methods

| Method | Description |
|--------|-------------|
| `is_trading_day(date)` | Check if date is a trading day |
| `is_market_open(dt, include_extended_hours)` | Check if market is open |
| `get_next_market_open(from_dt)` | Get next market open time |
| `get_trading_minutes_remaining(dt)` | Get minutes until close |

### Helper Functions

```python
# Create scheduler with default configuration
create_default_scheduler(
    signal_generator_30m: Callable = None,
    signal_generator_1h: Callable = None,
    run_on_start: bool = False
) -> SignalScheduler
```

## Holiday Management

### Turkish Public Holidays

The scheduler automatically recognizes these Turkish holidays:

**Fixed Holidays:**
- January 1: New Year's Day
- April 23: National Sovereignty and Children's Day
- May 1: Labor Day
- May 19: Commemoration of Atatürk, Youth and Sports Day
- July 15: Democracy and National Unity Day
- August 30: Victory Day
- October 29: Republic Day

**Religious Holidays (dates vary):**
- Ramadan Feast (3 days)
- Sacrifice Feast (4 days)

### Updating Holidays

To add custom holidays or update religious holiday dates:

```python
from datetime import datetime
from src.signals.scheduler import BISTMarketHours

# Add custom holiday
BISTMarketHours.TURKISH_HOLIDAYS.add(datetime(2025, 12, 31).date())

# Or update in scheduler_config.yaml:
# holidays:
#   custom_dates:
#     - "2025-12-31"
```

## Monitoring and Logging

### Log Locations

```
logs/
├── scheduler.log          # Main scheduler log
├── signal_generation.log  # Signal generation log
└── errors.log            # Error log
```

### Log Format

```
2024-11-16 10:00:05 - scheduler - INFO - Starting 30-minute signal generation
2024-11-16 10:00:08 - scheduler - INFO - 30-minute signal generation completed in 3.2s
```

### Monitoring Market Status

```python
# Get comprehensive market status
status = scheduler.get_market_status()
print(status)

# Output:
# {
#     'current_time': datetime(2024, 11, 16, 14, 30, tzinfo=ZoneInfo('Europe/Istanbul')),
#     'is_trading_day': True,
#     'is_market_open': True,
#     'market_open_time': time(10, 0),
#     'market_close_time': time(18, 0),
#     'minutes_remaining': 210
# }
```

### Logging Best Practices

1. **Set appropriate log level**:
   - `DEBUG`: Development and troubleshooting
   - `INFO`: Normal operation (default)
   - `WARNING`: Potential issues
   - `ERROR`: Actual errors

2. **Monitor logs regularly**:
   ```bash
   # Watch logs in real-time
   tail -f logs/scheduler.log

   # Search for errors
   grep ERROR logs/scheduler.log
   ```

3. **Set up log rotation**:
   - Configured automatically in `scheduler_config.yaml`
   - Default: 10MB per file, 5 backup files

## Troubleshooting

### Common Issues

#### 1. Scheduler Not Running

**Problem**: Jobs aren't executing

**Solution**:
```python
# Check if scheduler is running
if scheduler.scheduler.running:
    print("Scheduler is running")
else:
    print("Scheduler is not running - call scheduler.start()")

# Check scheduled jobs
scheduler.print_schedule()
```

#### 2. Jobs Running Outside Market Hours

**Problem**: Signal generation runs when market is closed

**Solution**:
```python
# Verify market hours check is enabled
# The scheduler automatically validates market hours before execution
# Check logs for "Market is closed - skipping signal generation"
```

#### 3. Missing Dependencies

**Problem**: `ImportError` or `ModuleNotFoundError`

**Solution**:
```bash
# Install required packages
pip install -r requirements.txt

# Verify APScheduler installation
python -c "import apscheduler; print(apscheduler.__version__)"
```

#### 4. Timezone Issues

**Problem**: Jobs run at wrong times

**Solution**:
```python
# Verify timezone is correct
from src.signals.scheduler import BISTMarketHours
import datetime

now_turkey = datetime.datetime.now(BISTMarketHours.TURKEY_TZ)
print(f"Current Turkey time: {now_turkey}")
```

#### 5. Job Execution Failures

**Problem**: Jobs fail to execute

**Solution**:
```python
# Check job history for errors
history = scheduler.get_job_history(limit=10)
failed_jobs = history[history['success'] == False]
print(failed_jobs)

# Review error logs
# tail -f logs/scheduler.log | grep ERROR
```

### Debug Mode

Enable detailed logging:

```python
import logging

# Set to DEBUG level
logging.getLogger('apscheduler').setLevel(logging.DEBUG)
logging.getLogger('src.signals.scheduler').setLevel(logging.DEBUG)
```

### Getting Help

1. Check logs: `logs/scheduler.log`
2. Review configuration: `configs/scheduler_config.yaml`
3. Test market hours: `python examples/run_signal_scheduler.py --status`
4. Run in dry-run mode: `python examples/run_signal_scheduler.py --dry-run`

## Best Practices

1. **Always validate market hours** before signal generation
2. **Use try-except blocks** in signal generation functions
3. **Monitor job history** regularly
4. **Keep holiday calendar updated** annually
5. **Test in dry-run mode** before production
6. **Set up notifications** for job failures
7. **Use graceful shutdown** handlers
8. **Log all important events**
9. **Implement retry logic** for transient failures
10. **Monitor system resources** during execution

## Production Deployment

### Running as a Service

Create a systemd service (Linux):

```ini
# /etc/systemd/system/bist-scheduler.service
[Unit]
Description=BIST Signal Generation Scheduler
After=network.target

[Service]
Type=simple
User=bistml
WorkingDirectory=/home/user/BISTML
ExecStart=/usr/bin/python3 examples/run_signal_scheduler.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable bist-scheduler
sudo systemctl start bist-scheduler
sudo systemctl status bist-scheduler
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "examples/run_signal_scheduler.py", "--daemon"]
```

### Monitoring in Production

- Set up health checks
- Configure alerting (email/Telegram)
- Monitor resource usage
- Track job success rates
- Review logs daily

## Support

For issues or questions:
1. Check this guide
2. Review example scripts
3. Check logs and debug output
4. Consult APScheduler documentation: https://apscheduler.readthedocs.io/
