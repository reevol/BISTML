"""
Unit tests for BIST Signal Scheduler

Run with:
    pytest tests/test_scheduler.py -v
    python -m pytest tests/test_scheduler.py -v --cov=src.signals.scheduler
"""

import pytest
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock, patch, MagicMock

from src.signals.scheduler import (
    BISTMarketHours,
    SignalScheduler,
    create_default_scheduler
)


class TestBISTMarketHours:
    """Test cases for BISTMarketHours class."""

    def test_turkey_timezone(self):
        """Test Turkey timezone is correctly configured."""
        assert BISTMarketHours.TURKEY_TZ == ZoneInfo("Europe/Istanbul")

    def test_market_hours_constants(self):
        """Test market hours constants are correctly set."""
        assert BISTMarketHours.MARKET_OPEN == time(10, 0)
        assert BISTMarketHours.MARKET_CLOSE == time(18, 0)
        assert BISTMarketHours.PRE_MARKET_OPEN == time(9, 40)
        assert BISTMarketHours.POST_MARKET_CLOSE == time(18, 10)

    @pytest.mark.parametrize("weekday,expected", [
        (0, True),   # Monday
        (1, True),   # Tuesday
        (2, True),   # Wednesday
        (3, True),   # Thursday
        (4, True),   # Friday
        (5, False),  # Saturday
        (6, False),  # Sunday
    ])
    def test_is_trading_day_weekdays(self, weekday, expected):
        """Test weekend detection."""
        # Create a date for each weekday (using a week in Nov 2024)
        base_date = datetime(2024, 11, 11)  # Monday
        test_date = base_date + timedelta(days=weekday)
        test_date = test_date.replace(tzinfo=BISTMarketHours.TURKEY_TZ)

        # Only check if not a holiday
        if test_date.date() not in BISTMarketHours.TURKISH_HOLIDAYS:
            result = BISTMarketHours.is_trading_day(test_date)
            assert result == expected, f"Weekday {weekday} should be {expected}"

    def test_is_trading_day_holiday(self):
        """Test holiday detection."""
        # Test New Year's Day (always a holiday)
        new_year = datetime(2024, 1, 1, 12, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        assert BISTMarketHours.is_trading_day(new_year) is False

        # Test Republic Day (always a holiday)
        republic_day = datetime(2024, 10, 29, 12, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        assert BISTMarketHours.is_trading_day(republic_day) is False

    @pytest.mark.parametrize("hour,minute,expected", [
        (9, 30, False),   # Before market open
        (9, 59, False),   # Just before open
        (10, 0, True),    # Market open
        (10, 30, True),   # Mid-morning
        (14, 0, True),    # Afternoon
        (17, 59, True),   # Just before close
        (18, 0, False),   # Market close
        (18, 30, False),  # After close
        (23, 0, False),   # Night
    ])
    def test_is_market_open_times(self, hour, minute, expected):
        """Test market open detection at various times."""
        # Use a known trading day (Monday, Nov 11, 2024)
        test_dt = datetime(2024, 11, 11, hour, minute, tzinfo=BISTMarketHours.TURKEY_TZ)

        if test_dt.date() not in BISTMarketHours.TURKISH_HOLIDAYS:
            result = BISTMarketHours.is_market_open(test_dt)
            assert result == expected, f"Time {hour}:{minute:02d} should be {expected}"

    def test_is_market_open_with_extended_hours(self):
        """Test market open detection with extended hours."""
        # Pre-market time (9:45)
        pre_market = datetime(2024, 11, 11, 9, 45, tzinfo=BISTMarketHours.TURKEY_TZ)

        # Should be False for regular hours
        assert BISTMarketHours.is_market_open(pre_market, include_extended_hours=False) is False

        # Should be True for extended hours
        assert BISTMarketHours.is_market_open(pre_market, include_extended_hours=True) is True

    def test_get_next_market_open_same_day(self):
        """Test getting next market open on same day."""
        # Early morning before market open
        early_morning = datetime(2024, 11, 11, 8, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        next_open = BISTMarketHours.get_next_market_open(early_morning)

        # Should be same day at 10:00
        assert next_open.date() == early_morning.date()
        assert next_open.time() == time(10, 0)

    def test_get_next_market_open_next_day(self):
        """Test getting next market open on next day."""
        # After market close
        after_close = datetime(2024, 11, 11, 19, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        next_open = BISTMarketHours.get_next_market_open(after_close)

        # Should be next trading day at 10:00
        assert next_open > after_close
        assert next_open.time() == time(10, 0)

    def test_get_next_market_open_skip_weekend(self):
        """Test that next market open skips weekend."""
        # Friday after close
        friday_night = datetime(2024, 11, 15, 19, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        next_open = BISTMarketHours.get_next_market_open(friday_night)

        # Should be Monday
        assert next_open.weekday() == 0  # Monday
        assert next_open.time() == time(10, 0)

    def test_get_trading_minutes_remaining_during_market(self):
        """Test calculating remaining minutes during market hours."""
        # 2 hours before close (16:00)
        test_time = datetime(2024, 11, 11, 16, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        remaining = BISTMarketHours.get_trading_minutes_remaining(test_time)

        # Should be 120 minutes (2 hours)
        assert remaining == 120

    def test_get_trading_minutes_remaining_market_closed(self):
        """Test remaining minutes when market is closed."""
        # After market close
        test_time = datetime(2024, 11, 11, 19, 0, tzinfo=BISTMarketHours.TURKEY_TZ)
        remaining = BISTMarketHours.get_trading_minutes_remaining(test_time)

        # Should be 0
        assert remaining == 0


class TestSignalScheduler:
    """Test cases for SignalScheduler class."""

    @pytest.fixture
    def mock_signal_generator_30m(self):
        """Mock 30-minute signal generator."""
        mock = Mock(return_value={"signals": [], "count": 0})
        return mock

    @pytest.fixture
    def mock_signal_generator_1h(self):
        """Mock hourly signal generator."""
        mock = Mock(return_value={"signals": [], "count": 0})
        return mock

    @pytest.fixture
    def scheduler(self, mock_signal_generator_30m, mock_signal_generator_1h):
        """Create a scheduler instance for testing."""
        sched = SignalScheduler(
            signal_generator_30m=mock_signal_generator_30m,
            signal_generator_1h=mock_signal_generator_1h,
            run_on_start=False
        )
        yield sched
        # Clean up
        if sched.scheduler.running:
            sched.stop(wait=False)

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initializes correctly."""
        assert scheduler.scheduler is not None
        assert scheduler.signal_generator_30m is not None
        assert scheduler.signal_generator_1h is not None
        assert isinstance(scheduler.job_history, list)

    def test_add_30m_schedule(self, scheduler):
        """Test adding 30-minute schedule."""
        scheduler.add_30m_schedule()

        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert 'signal_30m' in job_ids

    def test_add_hourly_schedule(self, scheduler):
        """Test adding hourly schedule."""
        scheduler.add_hourly_schedule()

        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert 'signal_hourly' in job_ids

    def test_start_stop_scheduler(self, scheduler):
        """Test starting and stopping scheduler."""
        assert not scheduler.scheduler.running

        scheduler.start()
        assert scheduler.scheduler.running

        scheduler.stop(wait=False)
        assert not scheduler.scheduler.running

    def test_pause_resume_scheduler(self, scheduler):
        """Test pausing and resuming scheduler."""
        scheduler.start()

        scheduler.pause()
        assert scheduler.scheduler.state == 2  # PAUSED state

        scheduler.resume()
        assert scheduler.scheduler.state == 1  # RUNNING state

        scheduler.stop(wait=False)

    @patch('src.signals.scheduler.BISTMarketHours.is_market_open')
    def test_validate_market_hours_open(self, mock_is_open, scheduler):
        """Test market hours validation when market is open."""
        mock_is_open.return_value = True

        result = scheduler._validate_market_hours()
        assert result is True

    @patch('src.signals.scheduler.BISTMarketHours.is_market_open')
    def test_validate_market_hours_closed(self, mock_is_open, scheduler):
        """Test market hours validation when market is closed."""
        mock_is_open.return_value = False

        result = scheduler._validate_market_hours()
        assert result is False

    @patch('src.signals.scheduler.BISTMarketHours.is_market_open')
    def test_run_30m_signals_during_market_hours(
        self, mock_is_open, scheduler, mock_signal_generator_30m
    ):
        """Test 30-minute signal generation during market hours."""
        mock_is_open.return_value = True

        result = scheduler._run_30m_signals()

        # Verify signal generator was called
        mock_signal_generator_30m.assert_called_once()
        assert result is not None

    @patch('src.signals.scheduler.BISTMarketHours.is_market_open')
    def test_run_30m_signals_market_closed(
        self, mock_is_open, scheduler, mock_signal_generator_30m
    ):
        """Test 30-minute signal generation when market is closed."""
        mock_is_open.return_value = False

        result = scheduler._run_30m_signals()

        # Verify signal generator was NOT called
        mock_signal_generator_30m.assert_not_called()
        assert result is None

    @patch('src.signals.scheduler.BISTMarketHours.is_market_open')
    def test_run_hourly_signals_during_market_hours(
        self, mock_is_open, scheduler, mock_signal_generator_1h
    ):
        """Test hourly signal generation during market hours."""
        mock_is_open.return_value = True

        result = scheduler._run_hourly_signals()

        # Verify signal generator was called
        mock_signal_generator_1h.assert_called_once()
        assert result is not None

    def test_trigger_manual_run_30m(self, scheduler, mock_signal_generator_30m):
        """Test manual trigger of 30-minute signals."""
        with patch('src.signals.scheduler.BISTMarketHours.is_market_open', return_value=True):
            scheduler.trigger_manual_run('30m')
            mock_signal_generator_30m.assert_called_once()

    def test_trigger_manual_run_1h(self, scheduler, mock_signal_generator_1h):
        """Test manual trigger of hourly signals."""
        with patch('src.signals.scheduler.BISTMarketHours.is_market_open', return_value=True):
            scheduler.trigger_manual_run('1h')
            mock_signal_generator_1h.assert_called_once()

    def test_trigger_manual_run_invalid_interval(self, scheduler):
        """Test manual trigger with invalid interval."""
        with pytest.raises(ValueError):
            scheduler.trigger_manual_run('5m')

    def test_get_market_status(self, scheduler):
        """Test getting market status."""
        status = scheduler.get_market_status()

        assert 'current_time' in status
        assert 'is_trading_day' in status
        assert 'is_market_open' in status
        assert 'market_open_time' in status
        assert 'market_close_time' in status

    def test_job_history_tracking(self, scheduler):
        """Test job history is tracked."""
        initial_count = len(scheduler.job_history)

        # Simulate job execution event
        from apscheduler.events import JobExecutionEvent

        event = MagicMock(spec=JobExecutionEvent)
        event.job_id = 'test_job'
        event.scheduled_run_time = datetime.now()
        event.exception = None

        scheduler._job_executed_listener(event)

        assert len(scheduler.job_history) == initial_count + 1

    def test_get_job_history_dataframe(self, scheduler):
        """Test getting job history as DataFrame."""
        import pandas as pd

        history_df = scheduler.get_job_history(limit=10)
        assert isinstance(history_df, pd.DataFrame)

    def test_add_market_open_job(self, scheduler):
        """Test adding market open job."""
        mock_func = Mock()
        scheduler.add_market_open_job(mock_func)

        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert 'market_open' in job_ids

    def test_add_market_close_job(self, scheduler):
        """Test adding market close job."""
        mock_func = Mock()
        scheduler.add_market_close_job(mock_func)

        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert 'market_close' in job_ids


class TestCreateDefaultScheduler:
    """Test cases for create_default_scheduler helper function."""

    def test_create_default_scheduler_basic(self):
        """Test creating scheduler with basic configuration."""
        mock_30m = Mock()
        mock_1h = Mock()

        scheduler = create_default_scheduler(
            signal_generator_30m=mock_30m,
            signal_generator_1h=mock_1h
        )

        assert scheduler is not None
        assert scheduler.signal_generator_30m == mock_30m
        assert scheduler.signal_generator_1h == mock_1h

        # Clean up
        if scheduler.scheduler.running:
            scheduler.stop(wait=False)

    def test_create_default_scheduler_with_schedules(self):
        """Test that default scheduler creates appropriate jobs."""
        mock_30m = Mock()
        mock_1h = Mock()

        scheduler = create_default_scheduler(
            signal_generator_30m=mock_30m,
            signal_generator_1h=mock_1h
        )

        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        # Should have both 30m and hourly jobs
        assert 'signal_30m' in job_ids
        assert 'signal_hourly' in job_ids

        # Clean up
        if scheduler.scheduler.running:
            scheduler.stop(wait=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
