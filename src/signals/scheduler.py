"""
BIST Trading Signal Scheduler

This module provides a robust scheduler for generating trading signals during
Borsa Istanbul (BIST) trading hours using APScheduler.

Features:
- Automatic signal generation at 30-minute and hourly intervals
- BIST trading hours detection (10:00-18:00 Turkey Time)
- Turkish public holiday handling
- Market hours validation before signal generation
- Graceful error handling and retry logic
- Comprehensive logging
- Support for manual trigger and backfill
"""

import logging
import os
from datetime import datetime, time, timedelta
from typing import Callable, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BISTMarketHours:
    """
    Handler for BIST market hours and trading calendar.

    BIST trading hours:
    - Regular session: 10:00 - 18:00 Turkey Time (GMT+3)
    - Pre-market: 09:40 - 10:00 (optional monitoring)
    - Post-market: 18:00 - 18:10 (optional monitoring)
    """

    # Turkey timezone
    TURKEY_TZ = ZoneInfo("Europe/Istanbul")

    # Regular trading hours
    MARKET_OPEN = time(10, 0)   # 10:00 AM
    MARKET_CLOSE = time(18, 0)  # 6:00 PM

    # Extended hours (optional)
    PRE_MARKET_OPEN = time(9, 40)
    POST_MARKET_CLOSE = time(18, 10)

    # Turkish public holidays 2024-2025 (update annually)
    # These are fixed and religious holidays that BIST observes
    TURKISH_HOLIDAYS = {
        # 2024
        datetime(2024, 1, 1).date(),   # New Year's Day
        datetime(2024, 4, 10).date(),  # Ramadan Feast Day 1
        datetime(2024, 4, 11).date(),  # Ramadan Feast Day 2
        datetime(2024, 4, 12).date(),  # Ramadan Feast Day 3
        datetime(2024, 4, 23).date(),  # National Sovereignty Day
        datetime(2024, 5, 1).date(),   # Labor Day
        datetime(2024, 5, 19).date(),  # Youth and Sports Day
        datetime(2024, 6, 16).date(),  # Sacrifice Feast Day 1
        datetime(2024, 6, 17).date(),  # Sacrifice Feast Day 2
        datetime(2024, 6, 18).date(),  # Sacrifice Feast Day 3
        datetime(2024, 6, 19).date(),  # Sacrifice Feast Day 4
        datetime(2024, 7, 15).date(),  # Democracy and National Unity Day
        datetime(2024, 8, 30).date(),  # Victory Day
        datetime(2024, 10, 29).date(), # Republic Day

        # 2025 (approximate dates for religious holidays)
        datetime(2025, 1, 1).date(),   # New Year's Day
        datetime(2025, 3, 31).date(),  # Ramadan Feast Day 1 (approximate)
        datetime(2025, 4, 1).date(),   # Ramadan Feast Day 2
        datetime(2025, 4, 2).date(),   # Ramadan Feast Day 3
        datetime(2025, 4, 23).date(),  # National Sovereignty Day
        datetime(2025, 5, 1).date(),   # Labor Day
        datetime(2025, 5, 19).date(),  # Youth and Sports Day
        datetime(2025, 6, 7).date(),   # Sacrifice Feast Day 1 (approximate)
        datetime(2025, 6, 8).date(),   # Sacrifice Feast Day 2
        datetime(2025, 6, 9).date(),   # Sacrifice Feast Day 3
        datetime(2025, 6, 10).date(),  # Sacrifice Feast Day 4
        datetime(2025, 7, 15).date(),  # Democracy and National Unity Day
        datetime(2025, 8, 30).date(),  # Victory Day
        datetime(2025, 10, 29).date(), # Republic Day
    }

    @classmethod
    def is_trading_day(cls, date: Optional[datetime] = None) -> bool:
        """
        Check if a given date is a trading day (not weekend or holiday).

        Args:
            date: Date to check (defaults to today in Turkey timezone)

        Returns:
            True if it's a trading day, False otherwise
        """
        if date is None:
            date = datetime.now(cls.TURKEY_TZ)

        # Convert to Turkey timezone if needed
        if date.tzinfo is None:
            date = date.replace(tzinfo=cls.TURKEY_TZ)
        else:
            date = date.astimezone(cls.TURKEY_TZ)

        # Check if weekend (Saturday = 5, Sunday = 6)
        if date.weekday() >= 5:
            logger.debug(f"{date.date()} is a weekend")
            return False

        # Check if holiday
        if date.date() in cls.TURKISH_HOLIDAYS:
            logger.info(f"{date.date()} is a Turkish public holiday")
            return False

        return True

    @classmethod
    def is_market_open(cls, dt: Optional[datetime] = None,
                      include_extended_hours: bool = False) -> bool:
        """
        Check if the market is currently open.

        Args:
            dt: Datetime to check (defaults to now in Turkey timezone)
            include_extended_hours: Include pre-market and post-market hours

        Returns:
            True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now(cls.TURKEY_TZ)

        # Convert to Turkey timezone if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=cls.TURKEY_TZ)
        else:
            dt = dt.astimezone(cls.TURKEY_TZ)

        # First check if it's a trading day
        if not cls.is_trading_day(dt):
            return False

        # Check time
        current_time = dt.time()

        if include_extended_hours:
            return cls.PRE_MARKET_OPEN <= current_time <= cls.POST_MARKET_CLOSE
        else:
            return cls.MARKET_OPEN <= current_time < cls.MARKET_CLOSE

    @classmethod
    def get_next_market_open(cls, from_dt: Optional[datetime] = None) -> datetime:
        """
        Get the next market open time.

        Args:
            from_dt: Starting datetime (defaults to now)

        Returns:
            Datetime of next market open
        """
        if from_dt is None:
            from_dt = datetime.now(cls.TURKEY_TZ)

        # Start from the next day if we're past market close
        current_time = from_dt.time()
        if current_time >= cls.MARKET_CLOSE:
            from_dt = from_dt + timedelta(days=1)

        # Find next trading day
        next_date = from_dt.replace(hour=10, minute=0, second=0, microsecond=0)
        max_days_ahead = 10  # Prevent infinite loop

        for _ in range(max_days_ahead):
            if cls.is_trading_day(next_date):
                return next_date
            next_date += timedelta(days=1)

        logger.warning("Could not find next market open within 10 days")
        return next_date

    @classmethod
    def get_trading_minutes_remaining(cls, dt: Optional[datetime] = None) -> int:
        """
        Get number of trading minutes remaining today.

        Args:
            dt: Current datetime (defaults to now)

        Returns:
            Minutes remaining until market close, 0 if market is closed
        """
        if dt is None:
            dt = datetime.now(cls.TURKEY_TZ)

        if not cls.is_market_open(dt):
            return 0

        # Calculate time to close
        close_time = dt.replace(hour=18, minute=0, second=0, microsecond=0)
        remaining = close_time - dt

        return max(0, int(remaining.total_seconds() / 60))


class SignalScheduler:
    """
    Scheduler for automated trading signal generation during BIST trading hours.

    This scheduler manages periodic signal generation jobs with:
    - 30-minute interval signals
    - Hourly signals
    - Market hours awareness
    - Holiday handling
    - Error recovery
    """

    def __init__(self,
                 signal_generator_30m: Optional[Callable] = None,
                 signal_generator_1h: Optional[Callable] = None,
                 run_on_start: bool = False,
                 enable_extended_hours: bool = False):
        """
        Initialize the signal scheduler.

        Args:
            signal_generator_30m: Callable for 30-minute signal generation
            signal_generator_1h: Callable for hourly signal generation
            run_on_start: Whether to run signal generation on scheduler start
            enable_extended_hours: Include pre-market and post-market hours
        """
        self.scheduler = BackgroundScheduler(
            timezone=BISTMarketHours.TURKEY_TZ,
            job_defaults={
                'coalesce': True,  # Combine missed runs
                'max_instances': 1,  # Only one instance per job
                'misfire_grace_time': 300  # 5 minutes grace period
            }
        )

        self.signal_generator_30m = signal_generator_30m
        self.signal_generator_1h = signal_generator_1h
        self.enable_extended_hours = enable_extended_hours

        # Track job execution
        self.job_history: List[Dict] = []
        self.max_history = 1000

        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        logger.info("SignalScheduler initialized")

        if run_on_start:
            self._schedule_immediate_run()

    def _job_executed_listener(self, event: JobExecutionEvent):
        """Log job execution events."""
        job_info = {
            'job_id': event.job_id,
            'scheduled_time': event.scheduled_run_time,
            'execution_time': datetime.now(BISTMarketHours.TURKEY_TZ),
            'success': not event.exception
        }

        if event.exception:
            job_info['error'] = str(event.exception)
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.info(f"Job {event.job_id} completed successfully")

        # Maintain job history
        self.job_history.append(job_info)
        if len(self.job_history) > self.max_history:
            self.job_history.pop(0)

    def _validate_market_hours(self) -> bool:
        """
        Validate that market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        is_open = BISTMarketHours.is_market_open(
            include_extended_hours=self.enable_extended_hours
        )

        if not is_open:
            logger.info("Market is closed - skipping signal generation")
            next_open = BISTMarketHours.get_next_market_open()
            logger.info(f"Next market open: {next_open}")

        return is_open

    def _run_30m_signals(self):
        """Execute 30-minute signal generation with market hours validation."""
        logger.info("Starting 30-minute signal generation")

        if not self._validate_market_hours():
            return

        if self.signal_generator_30m is None:
            logger.warning("No 30-minute signal generator configured")
            return

        try:
            start_time = datetime.now(BISTMarketHours.TURKEY_TZ)
            result = self.signal_generator_30m()

            elapsed = (datetime.now(BISTMarketHours.TURKEY_TZ) - start_time).total_seconds()
            logger.info(f"30-minute signal generation completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error in 30-minute signal generation: {str(e)}", exc_info=True)
            raise

    def _run_hourly_signals(self):
        """Execute hourly signal generation with market hours validation."""
        logger.info("Starting hourly signal generation")

        if not self._validate_market_hours():
            return

        if self.signal_generator_1h is None:
            logger.warning("No hourly signal generator configured")
            return

        try:
            start_time = datetime.now(BISTMarketHours.TURKEY_TZ)
            result = self.signal_generator_1h()

            elapsed = (datetime.now(BISTMarketHours.TURKEY_TZ) - start_time).total_seconds()
            logger.info(f"Hourly signal generation completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error in hourly signal generation: {str(e)}", exc_info=True)
            raise

    def add_30m_schedule(self):
        """
        Add 30-minute interval job during market hours.

        Runs at :00 and :30 past each hour from 10:00 to 17:30.
        """
        # Schedule for every 30 minutes during market hours
        # Runs at 10:00, 10:30, 11:00, 11:30, ..., 17:00, 17:30
        self.scheduler.add_job(
            func=self._run_30m_signals,
            trigger=CronTrigger(
                day_of_week='mon-fri',  # Monday to Friday
                hour='10-17',           # 10 AM to 5 PM
                minute='0,30',          # At :00 and :30
                timezone=BISTMarketHours.TURKEY_TZ
            ),
            id='signal_30m',
            name='30-Minute Signal Generation',
            replace_existing=True
        )
        logger.info("30-minute signal generation scheduled (10:00-17:30, every 30 min)")

    def add_hourly_schedule(self):
        """
        Add hourly job during market hours.

        Runs at the top of each hour from 10:00 to 17:00.
        """
        self.scheduler.add_job(
            func=self._run_hourly_signals,
            trigger=CronTrigger(
                day_of_week='mon-fri',  # Monday to Friday
                hour='10-17',           # 10 AM to 5 PM
                minute='0',             # At :00
                timezone=BISTMarketHours.TURKEY_TZ
            ),
            id='signal_hourly',
            name='Hourly Signal Generation',
            replace_existing=True
        )
        logger.info("Hourly signal generation scheduled (10:00-17:00, every hour)")

    def add_market_open_job(self, func: Callable):
        """
        Add a job to run at market open (10:00).

        Args:
            func: Function to execute at market open
        """
        self.scheduler.add_job(
            func=func,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='10',
                minute='0',
                timezone=BISTMarketHours.TURKEY_TZ
            ),
            id='market_open',
            name='Market Open Job',
            replace_existing=True
        )
        logger.info("Market open job scheduled (10:00 daily)")

    def add_market_close_job(self, func: Callable):
        """
        Add a job to run at market close (18:00).

        Args:
            func: Function to execute at market close
        """
        self.scheduler.add_job(
            func=func,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='18',
                minute='0',
                timezone=BISTMarketHours.TURKEY_TZ
            ),
            id='market_close',
            name='Market Close Job',
            replace_existing=True
        )
        logger.info("Market close job scheduled (18:00 daily)")

    def _schedule_immediate_run(self):
        """Schedule an immediate run of signal generation if market is open."""
        if self._validate_market_hours():
            logger.info("Scheduling immediate signal generation run")

            if self.signal_generator_30m:
                self.scheduler.add_job(
                    func=self._run_30m_signals,
                    id='immediate_30m',
                    name='Immediate 30m Signal Run'
                )

            if self.signal_generator_1h:
                self.scheduler.add_job(
                    func=self._run_hourly_signals,
                    id='immediate_1h',
                    name='Immediate Hourly Signal Run'
                )

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Signal scheduler started")
            self.print_schedule()
        else:
            logger.warning("Scheduler is already running")

    def stop(self, wait: bool = True):
        """
        Stop the scheduler.

        Args:
            wait: Wait for jobs to complete before shutting down
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Signal scheduler stopped")
        else:
            logger.warning("Scheduler is not running")

    def pause(self):
        """Pause the scheduler (jobs won't execute but scheduler keeps running)."""
        self.scheduler.pause()
        logger.info("Signal scheduler paused")

    def resume(self):
        """Resume the scheduler."""
        self.scheduler.resume()
        logger.info("Signal scheduler resumed")

    def trigger_manual_run(self, interval: str = '30m'):
        """
        Manually trigger a signal generation run.

        Args:
            interval: Signal interval ('30m' or '1h')
        """
        logger.info(f"Manual trigger for {interval} signal generation")

        if interval == '30m':
            self._run_30m_signals()
        elif interval == '1h':
            self._run_hourly_signals()
        else:
            raise ValueError(f"Invalid interval: {interval}. Use '30m' or '1h'")

    def print_schedule(self):
        """Print current scheduled jobs."""
        jobs = self.scheduler.get_jobs()

        if not jobs:
            logger.info("No jobs scheduled")
            return

        logger.info("=" * 80)
        logger.info("SCHEDULED JOBS")
        logger.info("=" * 80)

        for job in jobs:
            logger.info(f"Job ID: {job.id}")
            logger.info(f"  Name: {job.name}")
            logger.info(f"  Next run: {job.next_run_time}")
            logger.info(f"  Trigger: {job.trigger}")
            logger.info("-" * 80)

    def get_job_history(self, limit: int = 10) -> pd.DataFrame:
        """
        Get recent job execution history.

        Args:
            limit: Number of recent jobs to return

        Returns:
            DataFrame with job history
        """
        recent_history = self.job_history[-limit:] if self.job_history else []
        return pd.DataFrame(recent_history)

    def get_market_status(self) -> Dict:
        """
        Get current market status information.

        Returns:
            Dictionary with market status details
        """
        now = datetime.now(BISTMarketHours.TURKEY_TZ)
        is_open = BISTMarketHours.is_market_open(now)
        is_trading_day = BISTMarketHours.is_trading_day(now)

        status = {
            'current_time': now,
            'is_trading_day': is_trading_day,
            'is_market_open': is_open,
            'market_open_time': BISTMarketHours.MARKET_OPEN,
            'market_close_time': BISTMarketHours.MARKET_CLOSE,
        }

        if is_trading_day and not is_open:
            status['next_market_open'] = BISTMarketHours.get_next_market_open(now)

        if is_open:
            status['minutes_remaining'] = BISTMarketHours.get_trading_minutes_remaining(now)

        return status


def create_default_scheduler(
    signal_generator_30m: Optional[Callable] = None,
    signal_generator_1h: Optional[Callable] = None,
    run_on_start: bool = False
) -> SignalScheduler:
    """
    Create a scheduler with default configuration.

    Args:
        signal_generator_30m: Function to generate 30-minute signals
        signal_generator_1h: Function to generate hourly signals
        run_on_start: Run signal generation immediately if market is open

    Returns:
        Configured SignalScheduler instance
    """
    scheduler = SignalScheduler(
        signal_generator_30m=signal_generator_30m,
        signal_generator_1h=signal_generator_1h,
        run_on_start=run_on_start
    )

    # Add default schedules if generators are provided
    if signal_generator_30m:
        scheduler.add_30m_schedule()

    if signal_generator_1h:
        scheduler.add_hourly_schedule()

    return scheduler


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the SignalScheduler.

    This demonstrates how to set up and use the scheduler for signal generation.
    """

    def example_30m_signal_generator():
        """Example 30-minute signal generator."""
        logger.info("Generating 30-minute signals...")
        # In production, this would call your actual signal generation logic
        # For example:
        # from src.signals.generator import generate_signals
        # signals = generate_signals(interval='30m')
        # return signals
        return {"status": "success", "signals_generated": 10}

    def example_hourly_signal_generator():
        """Example hourly signal generator."""
        logger.info("Generating hourly signals...")
        # In production, this would call your actual signal generation logic
        return {"status": "success", "signals_generated": 15}

    def market_open_routine():
        """Example market open routine."""
        logger.info("Market has opened! Running initialization tasks...")

    def market_close_routine():
        """Example market close routine."""
        logger.info("Market has closed! Running end-of-day tasks...")

    # Create scheduler
    scheduler = create_default_scheduler(
        signal_generator_30m=example_30m_signal_generator,
        signal_generator_1h=example_hourly_signal_generator,
        run_on_start=True
    )

    # Add market open/close routines
    scheduler.add_market_open_job(market_open_routine)
    scheduler.add_market_close_job(market_close_routine)

    # Check market status
    status = scheduler.get_market_status()
    logger.info(f"Market Status: {status}")

    # Print schedule
    scheduler.print_schedule()

    # Start the scheduler
    scheduler.start()

    logger.info("Scheduler is running. Press Ctrl+C to stop...")

    try:
        # Keep the script running
        import time
        while True:
            time.sleep(60)

            # Print status every minute
            if BISTMarketHours.is_market_open():
                remaining = BISTMarketHours.get_trading_minutes_remaining()
                logger.info(f"Market open - {remaining} minutes remaining")
            else:
                logger.info("Market closed")

    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.stop()
        logger.info("Scheduler stopped successfully")
