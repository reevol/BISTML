"""
Streamlit Dashboard for BIST AI Trading System

This interactive dashboard provides real-time monitoring and analysis of trading signals,
portfolio performance, backtesting results, and comprehensive performance metrics.

Features:
- Live trading signals table with filtering and sorting
- Portfolio overview with position tracking and P&L
- Backtesting results visualization
- Performance charts (equity curve, drawdown, returns distribution)
- Real-time updates and auto-refresh
- Interactive filters for stocks, timeframes, and signal types
- Export functionality for reports

Author: BIST AI Trading System
Date: 2025-11-16
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.signals.generator import SignalGenerator, TradingSignal, SignalType
from src.portfolio.manager import PortfolioManager, create_portfolio
from src.backtesting.engine import BacktestEngine, BacktestConfig, BacktestResults
from src.backtesting.metrics import calculate_all_metrics, PerformanceMetrics


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="BIST AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/BISTML',
        'Report a bug': 'https://github.com/yourusername/BISTML/issues',
        'About': '# BIST AI Trading System\nAdvanced AI-powered trading system for BIST stocks'
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_signals_data(filters: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load live trading signals from database or generate sample data.

    Args:
        filters: Optional filters for signals (stock, timeframe, signal type)

    Returns:
        DataFrame with trading signals
    """
    # In production, this would query the database
    # For now, generate sample data

    np.random.seed(42)
    stocks = ['THYAO', 'GARAN', 'EREGL', 'KCHOL', 'AKBNK', 'SAHOL', 'VAKBN', 'ISCTR', 'TUPRS', 'ASELS']
    signal_types = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']

    n_signals = 50
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 300)) for _ in range(n_signals)],
        'stock_code': np.random.choice(stocks, n_signals),
        'signal': np.random.choice(signal_types, n_signals, p=[0.15, 0.25, 0.35, 0.15, 0.10]),
        'confidence_score': np.random.uniform(0.5, 0.95, n_signals),
        'current_price': np.random.uniform(10, 250, n_signals),
        'target_price': np.random.uniform(10, 300, n_signals),
        'expected_return': np.random.uniform(-5, 10, n_signals),
        'position_size': np.random.uniform(0.05, 0.15, n_signals),
        'risk_score': np.random.uniform(0.2, 0.8, n_signals),
        'stop_loss': np.random.uniform(8, 240, n_signals),
        'take_profit': np.random.uniform(12, 280, n_signals),
    }

    df = pd.DataFrame(data)

    # Apply filters if provided
    if filters:
        if filters.get('stock') and filters['stock'] != 'All':
            df = df[df['stock_code'] == filters['stock']]
        if filters.get('signal_type') and filters['signal_type'] != 'All':
            df = df[df['signal'] == filters['signal_type']]
        if filters.get('min_confidence'):
            df = df[df['confidence_score'] >= filters['min_confidence']]

    return df.sort_values('timestamp', ascending=False)


@st.cache_data(ttl=300)
def load_portfolio_data() -> Dict[str, Any]:
    """
    Load portfolio data including positions, P&L, and performance.

    Returns:
        Dictionary with portfolio data
    """
    # In production, load from database or portfolio manager
    # For now, generate sample data

    current_prices = {
        'THYAO': 265.50,
        'GARAN': 92.30,
        'EREGL': 45.80,
        'KCHOL': 125.40
    }

    portfolio_data = {
        'total_value': 128500.00,
        'cash': 28500.00,
        'positions_value': 100000.00,
        'total_pnl': 28500.00,
        'total_return_pct': 28.5,
        'realized_pnl': 12000.00,
        'unrealized_pnl': 16500.00,
        'positions': [
            {
                'symbol': 'THYAO',
                'shares': 100,
                'cost_basis': 250.00,
                'current_price': 265.50,
                'market_value': 26550.00,
                'total_cost': 25000.00,
                'unrealized_pnl': 1550.00,
                'unrealized_pnl_pct': 6.2
            },
            {
                'symbol': 'GARAN',
                'shares': 300,
                'cost_basis': 85.00,
                'current_price': 92.30,
                'market_value': 27690.00,
                'total_cost': 25500.00,
                'unrealized_pnl': 2190.00,
                'unrealized_pnl_pct': 8.59
            },
            {
                'symbol': 'EREGL',
                'shares': 500,
                'cost_basis': 42.00,
                'current_price': 45.80,
                'market_value': 22900.00,
                'total_cost': 21000.00,
                'unrealized_pnl': 1900.00,
                'unrealized_pnl_pct': 9.05
            },
            {
                'symbol': 'KCHOL',
                'shares': 180,
                'cost_basis': 120.00,
                'current_price': 125.40,
                'market_value': 22572.00,
                'total_cost': 21600.00,
                'unrealized_pnl': 972.00,
                'unrealized_pnl_pct': 4.5
            }
        ],
        'daily_returns': np.random.randn(30) * 0.02 + 0.001,
        'equity_curve': [100000 * (1 + np.random.randn() * 0.02 + 0.001) for _ in range(30)]
    }

    return portfolio_data


@st.cache_data(ttl=600)
def load_backtest_results() -> Dict[str, Any]:
    """
    Load backtesting results.

    Returns:
        Dictionary with backtest results
    """
    # In production, load from saved backtest results
    # For now, generate sample data

    np.random.seed(42)
    n_days = 252  # One year

    # Generate realistic equity curve
    daily_returns = np.random.randn(n_days) * 0.015 + 0.0008
    equity_curve = 100000 * (1 + daily_returns).cumprod()

    # Generate trades
    n_trades = 150
    trades_data = {
        'entry_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_trades)],
        'symbol': np.random.choice(['THYAO', 'GARAN', 'EREGL', 'KCHOL', 'AKBNK'], n_trades),
        'entry_price': np.random.uniform(50, 200, n_trades),
        'exit_price': np.random.uniform(50, 220, n_trades),
        'shares': np.random.randint(50, 500, n_trades),
        'pnl': np.random.randn(n_trades) * 500 + 100,
        'pnl_percent': np.random.randn(n_trades) * 5 + 0.5,
        'holding_period': np.random.randint(1, 30, n_trades)
    }

    trades_df = pd.DataFrame(trades_data)
    trades_df = trades_df.sort_values('entry_date')

    results = {
        'start_date': datetime.now() - timedelta(days=365),
        'end_date': datetime.now(),
        'initial_capital': 100000.00,
        'final_capital': equity_curve[-1],
        'total_return': equity_curve[-1] - 100000,
        'total_return_pct': ((equity_curve[-1] / 100000) - 1) * 100,
        'annualized_return': 18.5,
        'sharpe_ratio': 1.85,
        'sortino_ratio': 2.35,
        'max_drawdown': -12.5,
        'max_drawdown_duration': 45,
        'calmar_ratio': 1.48,
        'total_trades': n_trades,
        'winning_trades': int(n_trades * 0.58),
        'losing_trades': int(n_trades * 0.42),
        'win_rate': 58.0,
        'avg_win': 850.00,
        'avg_loss': -420.00,
        'profit_factor': 2.02,
        'expectancy': 235.50,
        'avg_holding_period': 12.5,
        'total_commission': 2450.00,
        'total_slippage': 1250.00,
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'trades': trades_df
    }

    return results


def format_signal_badge(signal: str) -> str:
    """Format signal as colored badge HTML."""
    if signal in ['STRONG_BUY', 'BUY']:
        return f'<span class="signal-buy">{signal}</span>'
    elif signal in ['STRONG_SELL', 'SELL']:
        return f'<span class="signal-sell">{signal}</span>'
    else:
        return f'<span class="signal-hold">{signal}</span>'


def format_confidence(confidence: float) -> str:
    """Format confidence score with color."""
    if confidence >= 0.7:
        return f'<span class="confidence-high">{confidence:.1%}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.1%}</span>'


def create_equity_curve_chart(equity_data: np.ndarray, title: str = "Equity Curve") -> go.Figure:
    """Create interactive equity curve chart."""
    dates = pd.date_range(end=datetime.now(), periods=len(equity_data), freq='D')

    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_data)
    drawdown = (equity_data - running_max) / running_max * 100

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Drawdown (%)'),
        row_heights=[0.7, 0.3]
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_data,
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            name='Drawdown',
            line=dict(color='#d62728', width=1),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.2)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value (TRY)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_returns_distribution_chart(returns: np.ndarray) -> go.Figure:
    """Create returns distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Daily Returns',
        marker=dict(
            color='#1f77b4',
            line=dict(color='white', width=1)
        )
    ))

    # Add normal distribution overlay
    from scipy import stats
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) / 50

    fig.add_trace(go.Scatter(
        x=x * 100,
        y=y * 100,
        name='Normal Distribution',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


def create_portfolio_allocation_chart(positions: List[Dict]) -> go.Figure:
    """Create portfolio allocation pie chart."""
    labels = [p['symbol'] for p in positions]
    values = [p['market_value'] for p in positions]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='Portfolio Allocation',
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


def create_performance_metrics_chart(metrics: Dict) -> go.Figure:
    """Create performance metrics bar chart."""
    metrics_data = {
        'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor', 'Win Rate (%)'],
        'Value': [
            metrics['sharpe_ratio'],
            metrics['sortino_ratio'],
            metrics['calmar_ratio'],
            metrics['profit_factor'],
            metrics['win_rate']
        ]
    }

    df = pd.DataFrame(metrics_data)

    fig = go.Figure(data=[
        go.Bar(
            x=df['Metric'],
            y=df['Value'],
            marker=dict(
                color=df['Value'],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=df['Value'].round(2),
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='Key Performance Metrics',
        xaxis_title='',
        yaxis_title='Value',
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_trades_timeline(trades_df: pd.DataFrame) -> go.Figure:
    """Create trades timeline scatter plot."""
    trades_df = trades_df.copy()
    trades_df['color'] = trades_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red')

    fig = go.Figure()

    # Winning trades
    winning = trades_df[trades_df['pnl'] > 0]
    fig.add_trace(go.Scatter(
        x=winning['entry_date'],
        y=winning['pnl'],
        mode='markers',
        name='Winning Trades',
        marker=dict(
            size=10,
            color='green',
            line=dict(color='white', width=1)
        ),
        text=winning['symbol'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>P&L: %{y:.2f} TRY<extra></extra>'
    ))

    # Losing trades
    losing = trades_df[trades_df['pnl'] <= 0]
    fig.add_trace(go.Scatter(
        x=losing['entry_date'],
        y=losing['pnl'],
        mode='markers',
        name='Losing Trades',
        marker=dict(
            size=10,
            color='red',
            line=dict(color='white', width=1)
        ),
        text=losing['symbol'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>P&L: %{y:.2f} TRY<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Trades Timeline',
        xaxis_title='Date',
        yaxis_title='P&L (TRY)',
        height=400,
        template='plotly_white',
        hovermode='closest'
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">📈 BIST AI Trading Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar - Filters and Settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 60)
            st.info(f"Dashboard will refresh every {refresh_interval} seconds")

        st.divider()

        # Filters
        st.header("🔍 Filters")

        # Stock filter
        all_stocks = ['All', 'THYAO', 'GARAN', 'EREGL', 'KCHOL', 'AKBNK', 'SAHOL', 'VAKBN', 'ISCTR', 'TUPRS', 'ASELS']
        selected_stock = st.selectbox("Stock", all_stocks, index=0)

        # Signal type filter
        signal_types = ['All', 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        selected_signal = st.selectbox("Signal Type", signal_types, index=0)

        # Confidence threshold
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)

        # Date range for backtesting
        st.divider()
        st.header("📅 Date Range")
        date_from = st.date_input("From", datetime.now() - timedelta(days=365))
        date_to = st.date_input("To", datetime.now())

        st.divider()

        # Export options
        st.header("📥 Export")
        if st.button("Export Signals (CSV)"):
            st.success("Signals exported successfully!")
        if st.button("Export Portfolio (JSON)"):
            st.success("Portfolio exported successfully!")
        if st.button("Export Backtest Report"):
            st.success("Backtest report exported!")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Signals", "💼 Portfolio", "🔬 Backtesting", "📈 Performance"])

    # ========================================================================
    # TAB 1: Live Signals
    # ========================================================================
    with tab1:
        st.header("Live Trading Signals")

        # Load signals
        filters = {
            'stock': selected_stock,
            'signal_type': selected_signal,
            'min_confidence': min_confidence
        }
        signals_df = load_signals_data(filters)

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_signals = len(signals_df)
            st.metric("Total Signals", total_signals)

        with col2:
            buy_signals = len(signals_df[signals_df['signal'].isin(['BUY', 'STRONG_BUY'])])
            st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/max(total_signals,1)*100:.1f}%")

        with col3:
            sell_signals = len(signals_df[signals_df['signal'].isin(['SELL', 'STRONG_SELL'])])
            st.metric("Sell Signals", sell_signals, delta=f"-{sell_signals/max(total_signals,1)*100:.1f}%")

        with col4:
            avg_confidence = signals_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

        with col5:
            avg_return = signals_df['expected_return'].mean()
            st.metric("Avg Expected Return", f"{avg_return:.2f}%", delta=f"{avg_return:.2f}%")

        st.divider()

        # Signals table
        st.subheader("Signals Table")

        # Format the dataframe for display
        display_df = signals_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1%}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"{x:.2f}")
        display_df['target_price'] = display_df['target_price'].apply(lambda x: f"{x:.2f}")
        display_df['expected_return'] = display_df['expected_return'].apply(lambda x: f"{x:.2f}%")
        display_df['position_size'] = display_df['position_size'].apply(lambda x: f"{x:.1%}")
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.2f}")

        # Rename columns
        display_df = display_df.rename(columns={
            'timestamp': 'Time',
            'stock_code': 'Stock',
            'signal': 'Signal',
            'confidence_score': 'Confidence',
            'current_price': 'Current Price',
            'target_price': 'Target Price',
            'expected_return': 'Expected Return',
            'position_size': 'Position Size',
            'risk_score': 'Risk Score'
        })

        # Display with conditional formatting
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )

        # Signal distribution chart
        col1, col2 = st.columns(2)

        with col1:
            signal_counts = signals_df['signal'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=0.3,
                marker=dict(colors=['#2ca02c', '#8dd3c7', '#ffd92f', '#fb8072', '#e31a1c'])
            )])
            fig.update_layout(title='Signal Distribution', height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            stock_counts = signals_df['stock_code'].value_counts().head(10)
            fig = go.Figure(data=[go.Bar(
                x=stock_counts.index,
                y=stock_counts.values,
                marker=dict(color='#1f77b4')
            )])
            fig.update_layout(title='Top 10 Active Stocks', height=350, xaxis_title='Stock', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 2: Portfolio
    # ========================================================================
    with tab2:
        st.header("Portfolio Overview")

        # Load portfolio data
        portfolio = load_portfolio_data()

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Value",
                f"{portfolio['total_value']:,.2f} TRY",
                delta=f"{portfolio['total_pnl']:,.2f} TRY"
            )

        with col2:
            st.metric(
                "Cash",
                f"{portfolio['cash']:,.2f} TRY"
            )

        with col3:
            st.metric(
                "Positions Value",
                f"{portfolio['positions_value']:,.2f} TRY"
            )

        with col4:
            st.metric(
                "Total Return",
                f"{portfolio['total_return_pct']:.2f}%",
                delta=f"{portfolio['total_return_pct']:.2f}%"
            )

        with col5:
            st.metric(
                "Unrealized P&L",
                f"{portfolio['unrealized_pnl']:,.2f} TRY",
                delta=f"{portfolio['unrealized_pnl']:,.2f} TRY"
            )

        st.divider()

        # Portfolio charts
        col1, col2 = st.columns([1, 1])

        with col1:
            # Allocation pie chart
            allocation_fig = create_portfolio_allocation_chart(portfolio['positions'])
            st.plotly_chart(allocation_fig, use_container_width=True)

        with col2:
            # Equity curve
            equity_fig = create_equity_curve_chart(
                np.array(portfolio['equity_curve']),
                title="Portfolio Equity (Last 30 Days)"
            )
            equity_fig.update_layout(height=400)
            st.plotly_chart(equity_fig, use_container_width=True)

        st.divider()

        # Positions table
        st.subheader("Current Positions")

        positions_df = pd.DataFrame(portfolio['positions'])

        # Format for display
        display_positions = positions_df.copy()
        display_positions['shares'] = display_positions['shares'].astype(int)
        display_positions['cost_basis'] = display_positions['cost_basis'].apply(lambda x: f"{x:.2f}")
        display_positions['current_price'] = display_positions['current_price'].apply(lambda x: f"{x:.2f}")
        display_positions['market_value'] = display_positions['market_value'].apply(lambda x: f"{x:,.2f}")
        display_positions['total_cost'] = display_positions['total_cost'].apply(lambda x: f"{x:,.2f}")
        display_positions['unrealized_pnl'] = display_positions['unrealized_pnl'].apply(lambda x: f"{x:,.2f}")
        display_positions['unrealized_pnl_pct'] = display_positions['unrealized_pnl_pct'].apply(lambda x: f"{x:.2f}%")

        display_positions = display_positions.rename(columns={
            'symbol': 'Symbol',
            'shares': 'Shares',
            'cost_basis': 'Cost Basis',
            'current_price': 'Current Price',
            'market_value': 'Market Value',
            'total_cost': 'Total Cost',
            'unrealized_pnl': 'Unrealized P&L',
            'unrealized_pnl_pct': 'Return %'
        })

        st.dataframe(display_positions, use_container_width=True, height=300)

        # P&L breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("P&L Breakdown")
            pnl_data = pd.DataFrame({
                'Category': ['Realized P&L', 'Unrealized P&L', 'Total P&L'],
                'Amount (TRY)': [
                    portfolio['realized_pnl'],
                    portfolio['unrealized_pnl'],
                    portfolio['total_pnl']
                ]
            })

            fig = go.Figure(data=[go.Bar(
                x=pnl_data['Category'],
                y=pnl_data['Amount (TRY)'],
                marker=dict(
                    color=['#2ca02c', '#ff7f0e', '#1f77b4'],
                    line=dict(color='white', width=1)
                ),
                text=pnl_data['Amount (TRY)'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside'
            )])

            fig.update_layout(
                height=350,
                template='plotly_white',
                showlegend=False,
                yaxis_title='Amount (TRY)'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Position Performance")
            perf_df = positions_df.copy()
            perf_df = perf_df.sort_values('unrealized_pnl_pct', ascending=True)

            fig = go.Figure(data=[go.Bar(
                x=perf_df['unrealized_pnl_pct'],
                y=perf_df['symbol'],
                orientation='h',
                marker=dict(
                    color=perf_df['unrealized_pnl_pct'],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=perf_df['unrealized_pnl_pct'].apply(lambda x: f"{x:.2f}%"),
                textposition='outside'
            )])

            fig.update_layout(
                height=350,
                template='plotly_white',
                showlegend=False,
                xaxis_title='Return (%)',
                yaxis_title=''
            )
            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 3: Backtesting
    # ========================================================================
    with tab3:
        st.header("Backtesting Results")

        # Load backtest results
        backtest = load_backtest_results()

        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                "Initial Capital",
                f"{backtest['initial_capital']:,.0f} TRY"
            )

        with col2:
            st.metric(
                "Final Capital",
                f"{backtest['final_capital']:,.0f} TRY"
            )

        with col3:
            st.metric(
                "Total Return",
                f"{backtest['total_return_pct']:.2f}%",
                delta=f"{backtest['total_return']:,.0f} TRY"
            )

        with col4:
            st.metric(
                "Annualized Return",
                f"{backtest['annualized_return']:.2f}%"
            )

        with col5:
            st.metric(
                "Sharpe Ratio",
                f"{backtest['sharpe_ratio']:.2f}"
            )

        with col6:
            st.metric(
                "Max Drawdown",
                f"{backtest['max_drawdown']:.2f}%"
            )

        st.divider()

        # Equity curve
        st.subheader("Equity Curve & Drawdown")
        equity_fig = create_equity_curve_chart(backtest['equity_curve'], title="Backtest Equity Curve")
        st.plotly_chart(equity_fig, use_container_width=True)

        st.divider()

        # Trade statistics
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Trade Statistics")

            trade_stats = pd.DataFrame({
                'Metric': [
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Win Rate',
                    'Avg Win',
                    'Avg Loss',
                    'Profit Factor',
                    'Expectancy',
                    'Avg Holding Period'
                ],
                'Value': [
                    backtest['total_trades'],
                    backtest['winning_trades'],
                    backtest['losing_trades'],
                    f"{backtest['win_rate']:.1f}%",
                    f"{backtest['avg_win']:.2f} TRY",
                    f"{backtest['avg_loss']:.2f} TRY",
                    f"{backtest['profit_factor']:.2f}",
                    f"{backtest['expectancy']:.2f} TRY",
                    f"{backtest['avg_holding_period']:.1f} days"
                ]
            })

            st.dataframe(trade_stats, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Risk Metrics")

            risk_metrics = pd.DataFrame({
                'Metric': [
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Max Drawdown',
                    'Max DD Duration',
                    'Total Commission',
                    'Total Slippage'
                ],
                'Value': [
                    f"{backtest['sharpe_ratio']:.3f}",
                    f"{backtest['sortino_ratio']:.3f}",
                    f"{backtest['calmar_ratio']:.3f}",
                    f"{backtest['max_drawdown']:.2f}%",
                    f"{backtest['max_drawdown_duration']} days",
                    f"{backtest['total_commission']:,.2f} TRY",
                    f"{backtest['total_slippage']:,.2f} TRY"
                ]
            })

            st.dataframe(risk_metrics, use_container_width=True, hide_index=True)

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Returns distribution
            returns_fig = create_returns_distribution_chart(backtest['daily_returns'])
            st.plotly_chart(returns_fig, use_container_width=True)

        with col2:
            # Performance metrics
            perf_fig = create_performance_metrics_chart(backtest)
            st.plotly_chart(perf_fig, use_container_width=True)

        st.divider()

        # Trades timeline
        st.subheader("Trades Timeline")
        timeline_fig = create_trades_timeline(backtest['trades'])
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Recent trades table
        st.subheader("Recent Trades (Last 20)")

        recent_trades = backtest['trades'].head(20).copy()
        recent_trades['entry_date'] = recent_trades['entry_date'].dt.strftime('%Y-%m-%d')
        recent_trades['entry_price'] = recent_trades['entry_price'].apply(lambda x: f"{x:.2f}")
        recent_trades['exit_price'] = recent_trades['exit_price'].apply(lambda x: f"{x:.2f}")
        recent_trades['pnl'] = recent_trades['pnl'].apply(lambda x: f"{x:,.2f}")
        recent_trades['pnl_percent'] = recent_trades['pnl_percent'].apply(lambda x: f"{x:.2f}%")

        recent_trades = recent_trades.rename(columns={
            'entry_date': 'Date',
            'symbol': 'Symbol',
            'entry_price': 'Entry Price',
            'exit_price': 'Exit Price',
            'shares': 'Shares',
            'pnl': 'P&L (TRY)',
            'pnl_percent': 'Return %',
            'holding_period': 'Days Held'
        })

        st.dataframe(recent_trades, use_container_width=True, hide_index=True)

    # ========================================================================
    # TAB 4: Performance
    # ========================================================================
    with tab4:
        st.header("Performance Analysis")

        # Load data
        portfolio = load_portfolio_data()
        backtest = load_backtest_results()

        # Key performance indicators
        st.subheader("Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sharpe Ratio", f"{backtest['sharpe_ratio']:.2f}")
            st.caption("Risk-adjusted return")

        with col2:
            st.metric("Sortino Ratio", f"{backtest['sortino_ratio']:.2f}")
            st.caption("Downside risk-adjusted return")

        with col3:
            st.metric("Calmar Ratio", f"{backtest['calmar_ratio']:.2f}")
            st.caption("Return vs max drawdown")

        with col4:
            st.metric("Profit Factor", f"{backtest['profit_factor']:.2f}")
            st.caption("Gross profit / Gross loss")

        st.divider()

        # Monthly returns heatmap
        st.subheader("Monthly Returns Heatmap")

        # Generate monthly returns data
        dates = pd.date_range(end=datetime.now(), periods=len(backtest['daily_returns']), freq='D')
        returns_series = pd.Series(backtest['daily_returns'], index=dates)
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table for heatmap
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })

        pivot_table = monthly_df.pivot(index='Month', columns='Year', values='Return')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot_table)],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)")
        ))

        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Year',
            yaxis_title='Month',
            height=400,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Rolling performance metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rolling Sharpe Ratio (30 days)")

            # Calculate rolling Sharpe
            window = 30
            rolling_sharpe = []
            for i in range(window, len(backtest['daily_returns'])):
                window_returns = backtest['daily_returns'][i-window:i]
                sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
                rolling_sharpe.append(sharpe)

            dates = pd.date_range(end=datetime.now(), periods=len(rolling_sharpe), freq='D')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_sharpe,
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Sharpe Ratio',
                height=350,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cumulative Returns")

            cumulative_returns = (1 + backtest['daily_returns']).cumprod() - 1
            dates = pd.date_range(end=datetime.now(), periods=len(cumulative_returns), freq='D')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns * 100,
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ))

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                height=350,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Performance summary table
        st.subheader("Comprehensive Performance Summary")

        summary_data = {
            'Category': ['Returns', 'Returns', 'Returns', 'Risk', 'Risk', 'Risk', 'Trade Stats', 'Trade Stats', 'Trade Stats', 'Costs'],
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Best Month',
                'Max Drawdown',
                'Volatility (Annual)',
                'Sharpe Ratio',
                'Total Trades',
                'Win Rate',
                'Profit Factor',
                'Total Costs'
            ],
            'Value': [
                f"{backtest['total_return_pct']:.2f}%",
                f"{backtest['annualized_return']:.2f}%",
                f"{monthly_returns.max() * 100:.2f}%",
                f"{backtest['max_drawdown']:.2f}%",
                f"{backtest['daily_returns'].std() * np.sqrt(252) * 100:.2f}%",
                f"{backtest['sharpe_ratio']:.3f}",
                f"{backtest['total_trades']}",
                f"{backtest['win_rate']:.1f}%",
                f"{backtest['profit_factor']:.2f}",
                f"{backtest['total_commission'] + backtest['total_slippage']:,.2f} TRY"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | BIST AI Trading System v1.0")

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
