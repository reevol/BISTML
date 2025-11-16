"""
Portfolio Alert System - Signal-Based Alerts for User Holdings

This module provides a comprehensive alert system that cross-references new trading
signals with user portfolio holdings and generates actionable alerts via multiple
channels (email, SMS, Telegram).

Key Features:
- Cross-reference signals with portfolio positions
- Generate contextual alerts based on signal type and holdings
- Multi-channel notification support (Email, SMS, Telegram)
- Alert priority and urgency classification
- Alert history and tracking
- Customizable alert rules and thresholds
- Template-based alert messages
- Batch alert processing
- Alert deduplication and throttling

Alert Types:
- Position signals: Alerts for stocks you currently hold
- Watchlist signals: Alerts for stocks on your watchlist
- Strong signal alerts: High-confidence signals
- Risk alerts: Alerts for positions with sell signals
- Opportunity alerts: Buy signals for stocks not in portfolio

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
import smtplib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import warnings

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class AlertType(Enum):
    """Types of alerts"""
    POSITION_SIGNAL = "POSITION_SIGNAL"  # Signal for current holding
    STRONG_SELL_HOLDING = "STRONG_SELL_HOLDING"  # Strong sell for holding
    SELL_HOLDING = "SELL_HOLDING"  # Sell signal for holding
    STRONG_BUY_OPPORTUNITY = "STRONG_BUY_OPPORTUNITY"  # Strong buy not in portfolio
    BUY_OPPORTUNITY = "BUY_OPPORTUNITY"  # Buy signal not in portfolio
    WATCHLIST_SIGNAL = "WATCHLIST_SIGNAL"  # Signal for watchlist stock
    RISK_WARNING = "RISK_WARNING"  # High risk warning
    PRICE_TARGET_REACHED = "PRICE_TARGET_REACHED"  # Price target reached


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "CRITICAL"  # Immediate action required
    HIGH = "HIGH"  # Action recommended soon
    MEDIUM = "MEDIUM"  # Review when convenient
    LOW = "LOW"  # Informational


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "EMAIL"
    SMS = "SMS"
    TELEGRAM = "TELEGRAM"
    PUSH = "PUSH"
    IN_APP = "IN_APP"


class SignalDirection(Enum):
    """Signal direction (simplified)"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Alert:
    """
    Represents a single alert to be sent to the user

    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of alert
        priority: Alert priority level
        symbol: Stock symbol
        signal_direction: Trading signal direction
        confidence_score: Signal confidence (0-100)
        title: Alert title/subject
        message: Alert message body
        timestamp: When alert was generated
        position_size: Current position size if holding (shares)
        position_value: Current position value if holding
        unrealized_pnl: Unrealized P&L if holding
        target_price: Predicted target price
        current_price: Current market price
        expected_return: Expected return percentage
        metadata: Additional alert metadata
    """
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    symbol: str
    signal_direction: SignalDirection
    confidence_score: float
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    position_size: Optional[float] = None
    position_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    expected_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['alert_type'] = self.alert_type.value
        d['priority'] = self.priority.value
        d['signal_direction'] = self.signal_direction.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    def to_short_summary(self) -> str:
        """Generate short summary for SMS/push"""
        return f"{self.symbol}: {self.signal_direction.value} ({self.confidence_score:.0f}% confidence)"

    def to_email_html(self) -> str:
        """Generate HTML formatted email"""
        # Priority badge color
        priority_colors = {
            AlertPriority.CRITICAL: "#dc3545",
            AlertPriority.HIGH: "#fd7e14",
            AlertPriority.MEDIUM: "#ffc107",
            AlertPriority.LOW: "#17a2b8"
        }

        # Signal badge color
        signal_colors = {
            SignalDirection.STRONG_BUY: "#28a745",
            SignalDirection.BUY: "#20c997",
            SignalDirection.HOLD: "#6c757d",
            SignalDirection.SELL: "#fd7e14",
            SignalDirection.STRONG_SELL: "#dc3545"
        }

        priority_color = priority_colors.get(self.priority, "#6c757d")
        signal_color = signal_colors.get(self.signal_direction, "#6c757d")

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f8f9fa;">
            <div style="background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: #333; margin-top: 0;">{self.title}</h2>

                <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                    <span style="background-color: {priority_color}; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        {self.priority.value}
                    </span>
                    <span style="background-color: {signal_color}; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        {self.signal_direction.value}
                    </span>
                </div>

                <p style="color: #555; line-height: 1.6;">{self.message}</p>

                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 6px; margin-top: 20px;">
                    <h3 style="margin-top: 0; color: #333; font-size: 16px;">Signal Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Symbol:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.symbol}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Confidence:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.confidence_score:.1f}%</td>
                        </tr>
        """

        if self.current_price is not None:
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Current Price:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.current_price:.2f} TRY</td>
                        </tr>
            """

        if self.target_price is not None:
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Target Price:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.target_price:.2f} TRY</td>
                        </tr>
            """

        if self.expected_return is not None:
            return_color = "#28a745" if self.expected_return > 0 else "#dc3545"
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Expected Return:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: {return_color};">
                                {self.expected_return:+.2f}%
                            </td>
                        </tr>
            """

        if self.position_size is not None:
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Your Position:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.position_size:.0f} shares</td>
                        </tr>
            """

        if self.position_value is not None:
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Position Value:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: #333;">{self.position_value:,.2f} TRY</td>
                        </tr>
            """

        if self.unrealized_pnl is not None:
            pnl_color = "#28a745" if self.unrealized_pnl > 0 else "#dc3545"
            html += f"""
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Unrealized P&L:</td>
                            <td style="padding: 8px 0; font-weight: bold; color: {pnl_color};">
                                {self.unrealized_pnl:+,.2f} TRY ({self.unrealized_pnl_pct:+.2f}%)
                            </td>
                        </tr>
            """

        html += """
                    </table>
                </div>

                <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px;">
                    <p style="margin: 0;">Generated at: {timestamp}</p>
                    <p style="margin: 5px 0 0 0;">BIST AI Trading System</p>
                </div>
            </div>
        </div>
        """.format(timestamp=self.timestamp.strftime('%Y-%m-%d %H:%M:%S'))

        return html


@dataclass
class AlertRule:
    """
    Rule for generating alerts based on conditions

    Attributes:
        rule_id: Unique rule identifier
        name: Rule name
        enabled: Whether rule is active
        alert_type: Type of alert to generate
        min_confidence: Minimum confidence score (0-100)
        signal_types: List of signal types to alert on
        only_holdings: Only alert for holdings
        channels: Notification channels to use
        priority: Alert priority
        throttle_hours: Hours to wait before re-alerting same symbol
    """
    rule_id: str
    name: str
    enabled: bool = True
    alert_type: AlertType = AlertType.POSITION_SIGNAL
    min_confidence: float = 50.0
    signal_types: List[SignalDirection] = field(default_factory=lambda: [
        SignalDirection.STRONG_BUY, SignalDirection.STRONG_SELL
    ])
    only_holdings: bool = False
    channels: List[NotificationChannel] = field(default_factory=lambda: [
        NotificationChannel.EMAIL
    ])
    priority: AlertPriority = AlertPriority.MEDIUM
    throttle_hours: int = 24  # Prevent alert spam


# ============================================================================
# Portfolio Alert Manager
# ============================================================================

class PortfolioAlertManager:
    """
    Manages alert generation and notification for portfolio positions

    This class cross-references trading signals with user holdings and
    generates contextual alerts via multiple channels.
    """

    def __init__(
        self,
        portfolio_manager=None,
        watchlist: Optional[List[str]] = None,
        alert_rules: Optional[List[AlertRule]] = None,
        smtp_config: Optional[Dict] = None,
        sms_config: Optional[Dict] = None,
        telegram_config: Optional[Dict] = None
    ):
        """
        Initialize Portfolio Alert Manager

        Args:
            portfolio_manager: PortfolioManager instance (optional)
            watchlist: List of symbols to watch (optional)
            alert_rules: Custom alert rules (optional)
            smtp_config: Email configuration (optional)
            sms_config: SMS configuration (optional)
            telegram_config: Telegram bot configuration (optional)
        """
        self.portfolio_manager = portfolio_manager
        self.watchlist = set(watchlist or [])

        # Alert rules
        self.alert_rules = alert_rules or self._get_default_rules()

        # Notification configurations
        self.smtp_config = smtp_config or self._load_smtp_config()
        self.sms_config = sms_config or {}
        self.telegram_config = telegram_config or self._load_telegram_config()

        # Alert history for throttling
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}

        logger.info("PortfolioAlertManager initialized")

    def _get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules"""
        return [
            # Critical: Strong sell signal for holdings
            AlertRule(
                rule_id="strong_sell_holding",
                name="Strong Sell Signal for Your Holdings",
                alert_type=AlertType.STRONG_SELL_HOLDING,
                min_confidence=60.0,
                signal_types=[SignalDirection.STRONG_SELL],
                only_holdings=True,
                channels=[NotificationChannel.EMAIL, NotificationChannel.TELEGRAM],
                priority=AlertPriority.CRITICAL,
                throttle_hours=12
            ),

            # High: Sell signal for holdings
            AlertRule(
                rule_id="sell_holding",
                name="Sell Signal for Your Holdings",
                alert_type=AlertType.SELL_HOLDING,
                min_confidence=65.0,
                signal_types=[SignalDirection.SELL],
                only_holdings=True,
                channels=[NotificationChannel.EMAIL],
                priority=AlertPriority.HIGH,
                throttle_hours=24
            ),

            # High: Strong buy opportunity
            AlertRule(
                rule_id="strong_buy_opportunity",
                name="Strong Buy Opportunity",
                alert_type=AlertType.STRONG_BUY_OPPORTUNITY,
                min_confidence=75.0,
                signal_types=[SignalDirection.STRONG_BUY],
                only_holdings=False,
                channels=[NotificationChannel.EMAIL],
                priority=AlertPriority.HIGH,
                throttle_hours=24
            ),

            # Medium: Any signal for holdings
            AlertRule(
                rule_id="position_signal",
                name="Signal for Your Holdings",
                alert_type=AlertType.POSITION_SIGNAL,
                min_confidence=60.0,
                signal_types=[SignalDirection.STRONG_BUY, SignalDirection.BUY,
                             SignalDirection.SELL, SignalDirection.STRONG_SELL],
                only_holdings=True,
                channels=[NotificationChannel.EMAIL],
                priority=AlertPriority.MEDIUM,
                throttle_hours=24
            )
        ]

    def _load_smtp_config(self) -> Dict:
        """Load SMTP configuration from environment"""
        return {
            'host': os.getenv('SMTP_HOST', ''),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'user': os.getenv('SMTP_USER', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('SMTP_USER', ''),
            'to_email': os.getenv('NOTIFICATION_EMAIL', '')
        }

    def _load_telegram_config(self) -> Dict:
        """Load Telegram configuration from environment"""
        return {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
        }

    def process_signals(
        self,
        signals: List[Dict],
        current_prices: Optional[Dict[str, float]] = None
    ) -> List[Alert]:
        """
        Process new signals and generate alerts

        Args:
            signals: List of trading signals (dict or TradingSignal objects)
            current_prices: Current market prices (optional)

        Returns:
            List of generated alerts
        """
        alerts = []

        # Get current holdings
        holdings = set()
        if self.portfolio_manager:
            holdings = set(self.portfolio_manager.positions.keys())

        for signal in signals:
            # Convert to dict if needed
            if hasattr(signal, 'to_dict'):
                signal_dict = signal.to_dict()
            else:
                signal_dict = signal

            symbol = signal_dict.get('stock_code') or signal_dict.get('symbol')
            signal_type = signal_dict.get('signal') or signal_dict.get('signal_type')
            confidence = signal_dict.get('confidence_score') or signal_dict.get('overall_confidence', 0)

            # Convert signal type to enum
            if isinstance(signal_type, str):
                try:
                    signal_direction = SignalDirection[signal_type.upper().replace(' ', '_')]
                except KeyError:
                    logger.warning(f"Unknown signal type: {signal_type}")
                    continue
            else:
                signal_direction = signal_type

            # Check if symbol is in holdings or watchlist
            is_holding = symbol in holdings
            is_watchlist = symbol in self.watchlist

            # Check each rule
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue

                # Check if rule applies
                if rule.only_holdings and not is_holding:
                    continue

                if signal_direction not in rule.signal_types:
                    continue

                if confidence < rule.min_confidence:
                    continue

                # Check throttling
                if self._is_throttled(symbol, rule):
                    logger.debug(f"Alert throttled for {symbol} (rule: {rule.name})")
                    continue

                # Generate alert
                alert = self._create_alert(
                    signal_dict=signal_dict,
                    symbol=symbol,
                    signal_direction=signal_direction,
                    confidence=confidence,
                    rule=rule,
                    is_holding=is_holding,
                    is_watchlist=is_watchlist,
                    current_prices=current_prices
                )

                if alert:
                    alerts.append(alert)
                    self._record_alert(symbol, rule)

        # Store alerts in history
        self.alert_history.extend(alerts)

        logger.info(f"Generated {len(alerts)} alerts from {len(signals)} signals")

        return alerts

    def _create_alert(
        self,
        signal_dict: Dict,
        symbol: str,
        signal_direction: SignalDirection,
        confidence: float,
        rule: AlertRule,
        is_holding: bool,
        is_watchlist: bool,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Optional[Alert]:
        """Create an alert from signal and rule"""

        # Get position details if holding
        position_size = None
        position_value = None
        unrealized_pnl = None
        unrealized_pnl_pct = None

        if is_holding and self.portfolio_manager:
            position = self.portfolio_manager.get_position(symbol)
            if position:
                position_size = position.shares
                current_price = signal_dict.get('current_price')

                if current_price:
                    pnl_metrics = position.calculate_unrealized_pnl(current_price)
                    position_value = pnl_metrics['market_value']
                    unrealized_pnl = pnl_metrics['unrealized_pnl']
                    unrealized_pnl_pct = pnl_metrics['unrealized_pnl_pct']

        # Generate alert title and message
        title, message = self._generate_alert_content(
            symbol=symbol,
            signal_direction=signal_direction,
            confidence=confidence,
            is_holding=is_holding,
            is_watchlist=is_watchlist,
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            signal_dict=signal_dict
        )

        # Create alert
        alert = Alert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{symbol}",
            alert_type=rule.alert_type,
            priority=rule.priority,
            symbol=symbol,
            signal_direction=signal_direction,
            confidence_score=confidence,
            title=title,
            message=message,
            position_size=position_size,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            target_price=signal_dict.get('target_price'),
            current_price=signal_dict.get('current_price'),
            expected_return=signal_dict.get('expected_return'),
            stop_loss=signal_dict.get('stop_loss'),
            take_profit=signal_dict.get('take_profit'),
            risk_score=signal_dict.get('risk_score'),
            metadata={
                'rule_id': rule.rule_id,
                'is_holding': is_holding,
                'is_watchlist': is_watchlist,
                'rationale': signal_dict.get('rationale', '')
            }
        )

        return alert

    def _generate_alert_content(
        self,
        symbol: str,
        signal_direction: SignalDirection,
        confidence: float,
        is_holding: bool,
        is_watchlist: bool,
        position_size: Optional[float],
        unrealized_pnl: Optional[float],
        unrealized_pnl_pct: Optional[float],
        signal_dict: Dict
    ) -> Tuple[str, str]:
        """Generate alert title and message"""

        # Title
        if is_holding:
            if signal_direction == SignalDirection.STRONG_SELL:
                title = f"⚠️ URGENT: Strong Sell Signal for Your Holding - {symbol}"
            elif signal_direction == SignalDirection.SELL:
                title = f"⚠️ Sell Signal for Your Holding - {symbol}"
            elif signal_direction == SignalDirection.STRONG_BUY:
                title = f"📈 Strong Buy Signal for Your Holding - {symbol}"
            elif signal_direction == SignalDirection.BUY:
                title = f"📊 Buy Signal for Your Holding - {symbol}"
            else:
                title = f"📋 {signal_direction.value} Signal for {symbol}"
        else:
            if signal_direction == SignalDirection.STRONG_BUY:
                title = f"🎯 Strong Buy Opportunity - {symbol}"
            elif signal_direction == SignalDirection.BUY:
                title = f"📈 Buy Opportunity - {symbol}"
            else:
                title = f"📊 {signal_direction.value} Signal - {symbol}"

        # Message
        message_parts = []

        # Opening
        if is_holding:
            message_parts.append(
                f"A {signal_direction.value} signal has been generated for {symbol}, "
                f"which you currently hold in your portfolio."
            )
        else:
            message_parts.append(
                f"A {signal_direction.value} signal has been generated for {symbol}."
            )

        # Confidence
        message_parts.append(
            f"\nSignal Confidence: {confidence:.1f}%"
        )

        # Position details
        if is_holding and position_size:
            message_parts.append(
                f"\nYour Position: {position_size:.0f} shares"
            )

            if unrealized_pnl is not None:
                pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
                message_parts.append(
                    f"Current P&L: {pnl_emoji} {unrealized_pnl:+,.2f} TRY ({unrealized_pnl_pct:+.2f}%)"
                )

        # Price information
        current_price = signal_dict.get('current_price')
        target_price = signal_dict.get('target_price')
        expected_return = signal_dict.get('expected_return')

        if current_price and target_price:
            message_parts.append(
                f"\nCurrent Price: {current_price:.2f} TRY"
            )
            message_parts.append(
                f"Target Price: {target_price:.2f} TRY"
            )

            if expected_return is not None:
                message_parts.append(
                    f"Expected Return: {expected_return*100:+.2f}%"
                )

        # Risk levels
        stop_loss = signal_dict.get('stop_loss')
        take_profit = signal_dict.get('take_profit')

        if stop_loss and take_profit:
            message_parts.append(
                f"\nSuggested Stop Loss: {stop_loss:.2f} TRY"
            )
            message_parts.append(
                f"Suggested Take Profit: {take_profit:.2f} TRY"
            )

        # Recommendation
        if is_holding:
            if signal_direction == SignalDirection.STRONG_SELL:
                message_parts.append(
                    f"\n⚠️ Recommendation: Consider selling your position to protect gains/minimize losses."
                )
            elif signal_direction == SignalDirection.SELL:
                message_parts.append(
                    f"\n⚠️ Recommendation: Review your position. Consider reducing exposure."
                )
            elif signal_direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                message_parts.append(
                    f"\n✅ Recommendation: Consider adding to your position if risk tolerance allows."
                )
        else:
            if signal_direction == SignalDirection.STRONG_BUY:
                message_parts.append(
                    f"\n✅ Recommendation: Strong buying opportunity. Consider opening a position."
                )
            elif signal_direction == SignalDirection.BUY:
                message_parts.append(
                    f"\n✅ Recommendation: Potential buying opportunity. Review fundamentals before entry."
                )

        # Rationale
        rationale = signal_dict.get('rationale')
        if rationale:
            message_parts.append(f"\nAnalysis: {rationale}")

        message = "\n".join(message_parts)

        return title, message

    def _is_throttled(self, symbol: str, rule: AlertRule) -> bool:
        """Check if alert should be throttled"""
        key = f"{symbol}_{rule.rule_id}"

        if key in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[key]
            if time_since_last < timedelta(hours=rule.throttle_hours):
                return True

        return False

    def _record_alert(self, symbol: str, rule: AlertRule):
        """Record alert time for throttling"""
        key = f"{symbol}_{rule.rule_id}"
        self.last_alert_time[key] = datetime.now()

    # ========================================================================
    # Notification Methods
    # ========================================================================

    def send_alerts(
        self,
        alerts: List[Alert],
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, int]:
        """
        Send alerts via specified channels

        Args:
            alerts: List of alerts to send
            channels: Channels to use (default: from alert rules)

        Returns:
            Dictionary with counts per channel
        """
        results = {
            'email': 0,
            'sms': 0,
            'telegram': 0,
            'total': len(alerts)
        }

        for alert in alerts:
            # Determine channels to use
            alert_channels = channels or [NotificationChannel.EMAIL]

            # Send via each channel
            for channel in alert_channels:
                try:
                    if channel == NotificationChannel.EMAIL:
                        if self._send_email(alert):
                            results['email'] += 1

                    elif channel == NotificationChannel.TELEGRAM:
                        if self._send_telegram(alert):
                            results['telegram'] += 1

                    elif channel == NotificationChannel.SMS:
                        if self._send_sms(alert):
                            results['sms'] += 1

                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")

        logger.info(f"Sent alerts - Email: {results['email']}, "
                   f"Telegram: {results['telegram']}, SMS: {results['sms']}")

        return results

    def _send_email(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.smtp_config or not self.smtp_config.get('host'):
            logger.warning("SMTP not configured, skipping email")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = alert.title
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = self.smtp_config['to_email']

            # Plain text version
            text_part = MIMEText(alert.message, 'plain')
            msg.attach(text_part)

            # HTML version
            html_part = MIMEText(alert.to_email_html(), 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                server.starttls()
                if self.smtp_config.get('user') and self.smtp_config.get('password'):
                    server.login(self.smtp_config['user'], self.smtp_config['password'])
                server.send_message(msg)

            logger.info(f"Email sent for {alert.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _send_telegram(self, alert: Alert) -> bool:
        """Send Telegram notification"""
        if not self.telegram_config or not self.telegram_config.get('bot_token'):
            logger.warning("Telegram not configured, skipping")
            return False

        try:
            import requests

            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_config['chat_id']

            # Format message for Telegram
            message = f"*{alert.title}*\n\n{alert.message}"

            # Send via Telegram API
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()

            logger.info(f"Telegram message sent for {alert.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _send_sms(self, alert: Alert) -> bool:
        """Send SMS notification (via Twilio or similar)"""
        if not self.sms_config or not self.sms_config.get('enabled'):
            logger.warning("SMS not configured, skipping")
            return False

        try:
            # This would integrate with Twilio or another SMS provider
            # Example implementation:
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=alert.to_short_summary(),
            #     from_=from_number,
            #     to=to_number
            # )

            logger.info(f"SMS would be sent for {alert.symbol} (not implemented)")
            return False

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        self.watchlist.add(symbol.upper())
        logger.info(f"Added {symbol} to watchlist")

    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        self.watchlist.discard(symbol.upper())
        logger.info(f"Removed {symbol} from watchlist")

    def get_alert_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alert history"""
        alerts = self.alert_history

        # Filter by symbol
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]

        # Filter by date
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]

        # Sort by timestamp (newest first)
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)

        return alerts[:limit]

    def get_alert_summary(self) -> Dict:
        """Get summary of alert history"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'alerts_by_priority': {},
                'alerts_by_type': {},
                'most_alerted_symbols': []
            }

        # Count by priority
        by_priority = {}
        for alert in self.alert_history:
            priority = alert.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1

        # Count by type
        by_type = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            by_type[alert_type] = by_type.get(alert_type, 0) + 1

        # Most alerted symbols
        symbol_counts = {}
        for alert in self.alert_history:
            symbol_counts[alert.symbol] = symbol_counts.get(alert.symbol, 0) + 1

        most_alerted = sorted(
            symbol_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_priority': by_priority,
            'alerts_by_type': by_type,
            'most_alerted_symbols': most_alerted
        }

    def clear_alert_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        self.last_alert_time.clear()
        logger.info("Alert history cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_alert_manager(
    portfolio_manager=None,
    watchlist: Optional[List[str]] = None,
    **kwargs
) -> PortfolioAlertManager:
    """
    Create a portfolio alert manager

    Args:
        portfolio_manager: PortfolioManager instance
        watchlist: List of symbols to watch
        **kwargs: Additional configuration

    Returns:
        PortfolioAlertManager instance
    """
    return PortfolioAlertManager(
        portfolio_manager=portfolio_manager,
        watchlist=watchlist,
        **kwargs
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Portfolio Alert System - Example Usage")
    print("=" * 80)

    # Example: Create alert manager
    print("\n1. Creating Alert Manager")
    print("-" * 80)

    alert_manager = create_alert_manager(
        watchlist=['THYAO', 'GARAN', 'AKBNK']
    )

    print(f"Watchlist: {alert_manager.watchlist}")
    print(f"Alert rules: {len(alert_manager.alert_rules)}")

    # Example: Process signals
    print("\n2. Processing Signals")
    print("-" * 80)

    example_signals = [
        {
            'stock_code': 'THYAO',
            'signal': 'STRONG_SELL',
            'confidence_score': 85.0,
            'current_price': 265.0,
            'target_price': 245.0,
            'expected_return': -0.075,
            'rationale': 'Multiple models indicate downward trend with high confidence.'
        },
        {
            'stock_code': 'AKBNK',
            'signal': 'STRONG_BUY',
            'confidence_score': 78.0,
            'current_price': 50.0,
            'target_price': 55.0,
            'expected_return': 0.10,
            'rationale': 'Strong fundamentals and positive sentiment.'
        }
    ]

    alerts = alert_manager.process_signals(example_signals)

    print(f"Generated {len(alerts)} alerts")

    for alert in alerts:
        print(f"\n  Alert: {alert.title}")
        print(f"  Priority: {alert.priority.value}")
        print(f"  Symbol: {alert.symbol}")
        print(f"  Signal: {alert.signal_direction.value}")
        print(f"  Confidence: {alert.confidence_score:.1f}%")

    # Example: Get alert summary
    print("\n3. Alert Summary")
    print("-" * 80)

    summary = alert_manager.get_alert_summary()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("Alert System Ready!")
    print("=" * 80)
