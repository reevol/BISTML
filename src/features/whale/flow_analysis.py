"""
Flow Analysis - Trading Flow Pattern Analysis for BIST

This module provides comprehensive analysis of trading flow patterns including:
- Foreign vs Local flow analysis
- Institutional vs Retail flow analysis
- Broker concentration analysis
- Flow momentum and trend detection
- Cross-sectional flow comparisons

These analyses help identify smart money movements, market manipulation patterns,
and institutional trading strategies in the BIST market.

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import entropy
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlowAnalyzer:
    """
    Comprehensive trading flow analysis for BIST stocks

    This class provides advanced analytics for understanding institutional
    trading patterns, foreign investor behavior, and broker concentration
    in the Turkish stock market.
    """

    def __init__(self, min_sample_size: int = 5):
        """
        Initialize FlowAnalyzer

        Parameters:
        -----------
        min_sample_size : int, default=5
            Minimum number of data points required for analysis
        """
        self.min_sample_size = min_sample_size
        logger.info("FlowAnalyzer initialized")

    # ========================================================================
    # FOREIGN VS LOCAL FLOW ANALYSIS
    # ========================================================================

    def analyze_foreign_local_flow(self,
                                   ownership_data: pd.DataFrame,
                                   brokerage_data: Optional[pd.DataFrame] = None,
                                   window: int = 20) -> pd.DataFrame:
        """
        Analyze foreign vs local trading flows

        Calculates flow metrics, trends, and relative positioning to identify
        foreign investor sentiment and capital flows.

        Parameters:
        -----------
        ownership_data : pd.DataFrame
            Ownership data with columns: date, symbol, foreign_institutional,
            local_institutional, foreign_individual, local_individual
        brokerage_data : pd.DataFrame, optional
            Brokerage distribution data for enhanced analysis
        window : int, default=20
            Rolling window for trend calculations

        Returns:
        --------
        pd.DataFrame
            DataFrame with foreign/local flow analysis metrics
        """
        if ownership_data.empty:
            logger.warning("Empty ownership data provided")
            return pd.DataFrame()

        logger.info("Analyzing foreign vs local flows")

        df = ownership_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])

        results = []

        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()

            if len(symbol_data) < self.min_sample_size:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Calculate total foreign and local ownership
            symbol_data['foreign_total'] = (
                symbol_data['foreign_institutional'] +
                symbol_data['foreign_individual']
            )
            symbol_data['local_total'] = (
                symbol_data['local_institutional'] +
                symbol_data['local_individual']
            )

            # Flow metrics (changes over time)
            symbol_data['foreign_flow'] = symbol_data['foreign_total'].diff()
            symbol_data['local_flow'] = symbol_data['local_total'].diff()
            symbol_data['foreign_institutional_flow'] = (
                symbol_data['foreign_institutional'].diff()
            )
            symbol_data['local_institutional_flow'] = (
                symbol_data['local_institutional'].diff()
            )

            # Net flow (foreign - local)
            symbol_data['net_foreign_flow'] = (
                symbol_data['foreign_flow'] - symbol_data['local_flow']
            )

            # Rolling metrics for trend detection
            symbol_data['foreign_flow_ma'] = (
                symbol_data['foreign_flow'].rolling(window).mean()
            )
            symbol_data['foreign_flow_std'] = (
                symbol_data['foreign_flow'].rolling(window).std()
            )

            # Z-score for foreign flow (standardized measure)
            symbol_data['foreign_flow_zscore'] = (
                (symbol_data['foreign_flow'] - symbol_data['foreign_flow_ma']) /
                symbol_data['foreign_flow_std'].replace(0, 1)
            )

            # Foreign ownership momentum
            symbol_data['foreign_momentum'] = (
                symbol_data['foreign_total'].pct_change(window) * 100
            )

            # Flow direction and strength
            symbol_data['foreign_flow_direction'] = np.where(
                symbol_data['foreign_flow'] > 0, 'INFLOW',
                np.where(symbol_data['foreign_flow'] < 0, 'OUTFLOW', 'NEUTRAL')
            )

            # Cumulative flow (net accumulation/distribution)
            symbol_data['cumulative_foreign_flow'] = (
                symbol_data['foreign_flow'].cumsum()
            )
            symbol_data['cumulative_local_flow'] = (
                symbol_data['local_flow'].cumsum()
            )

            # Foreign dominance ratio
            symbol_data['foreign_dominance'] = (
                symbol_data['foreign_total'] /
                (symbol_data['foreign_total'] + symbol_data['local_total']) * 100
            )

            # Flow acceleration (2nd derivative)
            symbol_data['foreign_flow_acceleration'] = (
                symbol_data['foreign_flow'].diff()
            )

            # Relative strength of foreign vs local flow
            symbol_data['foreign_local_flow_ratio'] = (
                symbol_data['foreign_flow'] /
                symbol_data['local_flow'].replace(0, np.nan)
            )

            results.append(symbol_data)

        if not results:
            return pd.DataFrame()

        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"Foreign/local flow analysis completed for {len(results)} symbols")

        return result_df

    def detect_foreign_accumulation_phases(self,
                                          flow_data: pd.DataFrame,
                                          symbol: str,
                                          threshold_pct: float = 1.0,
                                          min_duration: int = 5) -> pd.DataFrame:
        """
        Detect sustained foreign accumulation or distribution phases

        Parameters:
        -----------
        flow_data : pd.DataFrame
            Output from analyze_foreign_local_flow
        symbol : str
            Stock symbol to analyze
        threshold_pct : float, default=1.0
            Minimum average flow percentage to qualify as accumulation/distribution
        min_duration : int, default=5
            Minimum consecutive days to identify a phase

        Returns:
        --------
        pd.DataFrame
            DataFrame with identified accumulation/distribution phases
        """
        if flow_data.empty:
            return pd.DataFrame()

        symbol_data = flow_data[flow_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')

        # Identify accumulation periods (consecutive positive flows)
        symbol_data['is_accumulation'] = symbol_data['foreign_flow'] > threshold_pct
        symbol_data['is_distribution'] = symbol_data['foreign_flow'] < -threshold_pct

        # Find consecutive periods
        symbol_data['accumulation_group'] = (
            symbol_data['is_accumulation'] !=
            symbol_data['is_accumulation'].shift()
        ).cumsum()

        symbol_data['distribution_group'] = (
            symbol_data['is_distribution'] !=
            symbol_data['is_distribution'].shift()
        ).cumsum()

        # Aggregate phases
        accumulation_phases = symbol_data[symbol_data['is_accumulation']].groupby(
            'accumulation_group'
        ).agg({
            'date': ['min', 'max', 'count'],
            'foreign_flow': ['sum', 'mean'],
            'foreign_total': ['first', 'last']
        })

        accumulation_phases.columns = [
            'start_date', 'end_date', 'duration',
            'total_flow', 'avg_flow', 'start_ownership', 'end_ownership'
        ]

        # Filter by minimum duration
        accumulation_phases = accumulation_phases[
            accumulation_phases['duration'] >= min_duration
        ]
        accumulation_phases['phase_type'] = 'ACCUMULATION'
        accumulation_phases['symbol'] = symbol

        # Same for distribution
        distribution_phases = symbol_data[symbol_data['is_distribution']].groupby(
            'distribution_group'
        ).agg({
            'date': ['min', 'max', 'count'],
            'foreign_flow': ['sum', 'mean'],
            'foreign_total': ['first', 'last']
        })

        distribution_phases.columns = [
            'start_date', 'end_date', 'duration',
            'total_flow', 'avg_flow', 'start_ownership', 'end_ownership'
        ]

        distribution_phases = distribution_phases[
            distribution_phases['duration'] >= min_duration
        ]
        distribution_phases['phase_type'] = 'DISTRIBUTION'
        distribution_phases['symbol'] = symbol

        # Combine phases
        if accumulation_phases.empty and distribution_phases.empty:
            return pd.DataFrame()
        elif accumulation_phases.empty:
            phases = distribution_phases
        elif distribution_phases.empty:
            phases = accumulation_phases
        else:
            phases = pd.concat([accumulation_phases, distribution_phases])

        phases = phases.sort_values('start_date').reset_index(drop=True)

        return phases

    # ========================================================================
    # INSTITUTIONAL VS RETAIL FLOW ANALYSIS
    # ========================================================================

    def analyze_institutional_retail_flow(self,
                                         ownership_data: pd.DataFrame,
                                         brokerage_data: Optional[pd.DataFrame] = None,
                                         window: int = 20) -> pd.DataFrame:
        """
        Analyze institutional vs retail trading flows

        Identifies smart money movements and retail investor behavior patterns.

        Parameters:
        -----------
        ownership_data : pd.DataFrame
            Ownership data with institutional and individual breakdowns
        brokerage_data : pd.DataFrame, optional
            Brokerage data for enhanced analysis
        window : int, default=20
            Rolling window for calculations

        Returns:
        --------
        pd.DataFrame
            DataFrame with institutional/retail flow metrics
        """
        if ownership_data.empty:
            logger.warning("Empty ownership data provided")
            return pd.DataFrame()

        logger.info("Analyzing institutional vs retail flows")

        df = ownership_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])

        results = []

        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()

            if len(symbol_data) < self.min_sample_size:
                continue

            # Calculate total institutional and retail
            symbol_data['institutional_total'] = (
                symbol_data['foreign_institutional'] +
                symbol_data['local_institutional']
            )
            symbol_data['retail_total'] = (
                symbol_data['foreign_individual'] +
                symbol_data['local_individual']
            )

            # Flow metrics
            symbol_data['institutional_flow'] = (
                symbol_data['institutional_total'].diff()
            )
            symbol_data['retail_flow'] = symbol_data['retail_total'].diff()

            # Net institutional flow (institutional - retail)
            symbol_data['net_institutional_flow'] = (
                symbol_data['institutional_flow'] - symbol_data['retail_flow']
            )

            # Smart money indicator (when institutional and retail diverge)
            symbol_data['smart_money_divergence'] = (
                symbol_data['institutional_flow'] * -symbol_data['retail_flow']
            )
            symbol_data['smart_money_signal'] = np.where(
                symbol_data['smart_money_divergence'] > 0,
                np.where(symbol_data['institutional_flow'] > 0,
                        'INSTITUTIONAL_BUY', 'INSTITUTIONAL_SELL'),
                'ALIGNED'
            )

            # Rolling metrics
            symbol_data['institutional_flow_ma'] = (
                symbol_data['institutional_flow'].rolling(window).mean()
            )
            symbol_data['retail_flow_ma'] = (
                symbol_data['retail_flow'].rolling(window).mean()
            )

            # Institutional dominance
            symbol_data['institutional_dominance'] = (
                symbol_data['institutional_total'] /
                (symbol_data['institutional_total'] + symbol_data['retail_total']) * 100
            )

            # Flow momentum
            symbol_data['institutional_momentum'] = (
                symbol_data['institutional_total'].pct_change(window) * 100
            )
            symbol_data['retail_momentum'] = (
                symbol_data['retail_total'].pct_change(window) * 100
            )

            # Momentum divergence
            symbol_data['momentum_divergence'] = (
                symbol_data['institutional_momentum'] -
                symbol_data['retail_momentum']
            )

            # Cumulative flows
            symbol_data['cumulative_institutional_flow'] = (
                symbol_data['institutional_flow'].cumsum()
            )
            symbol_data['cumulative_retail_flow'] = (
                symbol_data['retail_flow'].cumsum()
            )

            # Z-scores for outlier detection
            inst_std = symbol_data['institutional_flow'].rolling(window).std()
            symbol_data['institutional_flow_zscore'] = (
                (symbol_data['institutional_flow'] -
                 symbol_data['institutional_flow_ma']) /
                inst_std.replace(0, 1)
            )

            retail_std = symbol_data['retail_flow'].rolling(window).std()
            symbol_data['retail_flow_zscore'] = (
                (symbol_data['retail_flow'] - symbol_data['retail_flow_ma']) /
                retail_std.replace(0, 1)
            )

            # Institutional conviction (large moves relative to history)
            symbol_data['institutional_conviction'] = (
                abs(symbol_data['institutional_flow_zscore'])
            )

            results.append(symbol_data)

        if not results:
            return pd.DataFrame()

        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"Institutional/retail analysis completed for {len(results)} symbols")

        return result_df

    def identify_smart_money_signals(self,
                                    flow_data: pd.DataFrame,
                                    conviction_threshold: float = 2.0,
                                    divergence_threshold: float = 1.0) -> pd.DataFrame:
        """
        Identify high-conviction smart money signals

        Parameters:
        -----------
        flow_data : pd.DataFrame
            Output from analyze_institutional_retail_flow
        conviction_threshold : float, default=2.0
            Minimum Z-score for institutional conviction
        divergence_threshold : float, default=1.0
            Minimum momentum divergence percentage

        Returns:
        --------
        pd.DataFrame
            DataFrame with smart money signals
        """
        if flow_data.empty:
            return pd.DataFrame()

        # Filter for high conviction institutional moves
        signals = flow_data[
            (abs(flow_data['institutional_flow_zscore']) >= conviction_threshold) &
            (abs(flow_data['momentum_divergence']) >= divergence_threshold)
        ].copy()

        # Add signal classification
        signals['signal_type'] = np.where(
            (signals['institutional_flow'] > 0) & (signals['retail_flow'] < 0),
            'STRONG_INSTITUTIONAL_BUY',
            np.where(
                (signals['institutional_flow'] < 0) & (signals['retail_flow'] > 0),
                'STRONG_INSTITUTIONAL_SELL',
                np.where(
                    signals['institutional_flow'] > 0,
                    'INSTITUTIONAL_BUY',
                    'INSTITUTIONAL_SELL'
                )
            )
        )

        # Signal strength
        signals['signal_strength'] = (
            (abs(signals['institutional_flow_zscore']) * 0.6 +
             abs(signals['momentum_divergence']) / 10 * 0.4)
        )

        return signals[['date', 'symbol', 'signal_type', 'signal_strength',
                       'institutional_flow', 'retail_flow',
                       'institutional_flow_zscore', 'momentum_divergence']]

    # ========================================================================
    # BROKER CONCENTRATION ANALYSIS
    # ========================================================================

    def analyze_broker_concentration(self,
                                    brokerage_data: pd.DataFrame,
                                    window: int = 20) -> pd.DataFrame:
        """
        Analyze broker concentration and market impact

        Calculates concentration indices, identifies dominant brokers,
        and measures market fragmentation.

        Parameters:
        -----------
        brokerage_data : pd.DataFrame
            Brokerage distribution data
        window : int, default=20
            Rolling window for time-series metrics

        Returns:
        --------
        pd.DataFrame
            DataFrame with broker concentration metrics by symbol and date
        """
        if brokerage_data.empty:
            logger.warning("Empty brokerage data provided")
            return pd.DataFrame()

        logger.info("Analyzing broker concentration")

        df = brokerage_data.copy()
        df['date'] = pd.to_datetime(df['date'])

        results = []

        # Group by symbol and date
        for (symbol, date), group in df.groupby(['symbol', 'date']):
            if len(group) < 2:  # Need at least 2 brokers
                continue

            metrics = {
                'symbol': symbol,
                'date': date,
                'num_brokers': len(group),
                'total_volume': group['buy_volume'].sum() + group['sell_volume'].sum(),
                'total_value': group['buy_value'].sum() + group['sell_value'].sum()
            }

            # Calculate market shares
            buy_shares = group['buy_volume'] / group['buy_volume'].sum()
            sell_shares = group['sell_volume'] / group['sell_volume'].sum()
            total_shares = (group['buy_volume'] + group['sell_volume']) / (
                group['buy_volume'].sum() + group['sell_volume'].sum()
            )

            # Herfindahl-Hirschman Index (HHI)
            # Measures market concentration (0 to 10000)
            # < 1500: Competitive, 1500-2500: Moderate, > 2500: High concentration
            metrics['hhi_buy'] = (buy_shares ** 2).sum() * 10000
            metrics['hhi_sell'] = (sell_shares ** 2).sum() * 10000
            metrics['hhi_total'] = (total_shares ** 2).sum() * 10000

            # Concentration ratios (CR)
            sorted_buy = group.nlargest(min(5, len(group)), 'buy_volume')
            sorted_sell = group.nlargest(min(5, len(group)), 'sell_volume')
            sorted_total = group.assign(
                total_vol=lambda x: x['buy_volume'] + x['sell_volume']
            ).nlargest(min(5, len(group)), 'total_vol')

            # CR3 (top 3 brokers' market share)
            metrics['cr3_buy'] = (
                sorted_buy.head(3)['buy_volume'].sum() /
                group['buy_volume'].sum() * 100
            )
            metrics['cr3_sell'] = (
                sorted_sell.head(3)['sell_volume'].sum() /
                group['sell_volume'].sum() * 100
            )

            # CR5 (top 5 brokers' market share)
            metrics['cr5_buy'] = (
                sorted_buy.head(5)['buy_volume'].sum() /
                group['buy_volume'].sum() * 100
            )
            metrics['cr5_sell'] = (
                sorted_sell.head(5)['sell_volume'].sum() /
                group['sell_volume'].sum() * 100
            )

            # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
            metrics['gini_coefficient'] = self._calculate_gini(
                group['buy_volume'] + group['sell_volume']
            )

            # Entropy (higher = more distributed)
            metrics['shannon_entropy'] = entropy(total_shares + 1e-10)
            metrics['normalized_entropy'] = (
                metrics['shannon_entropy'] / np.log(len(group))
            )

            # Dominant broker analysis
            dominant = sorted_total.iloc[0]
            metrics['dominant_broker'] = dominant['broker_code']
            metrics['dominant_broker_share'] = (
                (dominant['buy_volume'] + dominant['sell_volume']) /
                (group['buy_volume'].sum() + group['sell_volume'].sum()) * 100
            )
            metrics['dominant_broker_net'] = (
                dominant['buy_volume'] - dominant['sell_volume']
            )

            # Top 3 brokers net positions
            top3_net = sorted_total.head(3)['net_volume'].sum()
            metrics['top3_net_position'] = top3_net
            metrics['top3_net_pct'] = (
                abs(top3_net) / metrics['total_volume'] * 100
            )

            # Concentration trend indicator
            # High HHI + Low entropy = Very concentrated market
            metrics['concentration_score'] = (
                metrics['hhi_total'] / 10000 * 0.6 +
                (1 - metrics['normalized_entropy']) * 0.4
            ) * 100

            results.append(metrics)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(['symbol', 'date'])

        # Add rolling metrics for trends
        for symbol in result_df['symbol'].unique():
            mask = result_df['symbol'] == symbol
            result_df.loc[mask, 'concentration_ma'] = (
                result_df.loc[mask, 'concentration_score'].rolling(window).mean()
            )
            result_df.loc[mask, 'concentration_trend'] = (
                result_df.loc[mask, 'concentration_score'] -
                result_df.loc[mask, 'concentration_ma']
            )
            result_df.loc[mask, 'hhi_trend'] = (
                result_df.loc[mask, 'hhi_total'].diff()
            )

        logger.info(f"Broker concentration analysis completed")

        return result_df

    def _calculate_gini(self, values: pd.Series) -> float:
        """
        Calculate Gini coefficient

        Parameters:
        -----------
        values : pd.Series
            Values to calculate Gini coefficient for

        Returns:
        --------
        float
            Gini coefficient (0 to 1)
        """
        values = values.sort_values()
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0

        cumsum = values.cumsum()
        return (n + 1 - 2 * cumsum.sum() / cumsum.iloc[-1]) / n

    def identify_broker_coordination(self,
                                    brokerage_data: pd.DataFrame,
                                    symbol: str,
                                    top_n: int = 5,
                                    correlation_threshold: float = 0.7,
                                    window: int = 20) -> Dict:
        """
        Identify potential coordination among top brokers

        Detects patterns suggesting coordinated buying or selling among
        major brokers, which may indicate manipulation or coordinated
        institutional activity.

        Parameters:
        -----------
        brokerage_data : pd.DataFrame
            Brokerage distribution data
        symbol : str
            Stock symbol to analyze
        top_n : int, default=5
            Number of top brokers to analyze
        correlation_threshold : float, default=0.7
            Minimum correlation to flag as coordinated
        window : int, default=20
            Rolling window for correlation analysis

        Returns:
        --------
        Dict
            Dictionary with coordination analysis results
        """
        if brokerage_data.empty:
            return {}

        symbol_data = brokerage_data[brokerage_data['symbol'] == symbol].copy()

        if symbol_data.empty:
            return {}

        # Get top brokers by total volume
        broker_totals = symbol_data.groupby('broker_code').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum'
        })
        broker_totals['total_volume'] = (
            broker_totals['buy_volume'] + broker_totals['sell_volume']
        )
        top_brokers = broker_totals.nlargest(top_n, 'total_volume').index.tolist()

        # Create pivot table for time series analysis
        pivot = symbol_data[symbol_data['broker_code'].isin(top_brokers)].pivot_table(
            index='date',
            columns='broker_code',
            values='net_volume',
            fill_value=0
        )

        if len(pivot) < window:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {window} days of data'
            }

        # Calculate rolling correlations
        correlations = pivot.rolling(window).corr()

        # Find highly correlated pairs
        coordinated_pairs = []
        for broker1 in top_brokers:
            for broker2 in top_brokers:
                if broker1 >= broker2:  # Avoid duplicates and self-correlation
                    continue

                try:
                    corr_series = correlations.loc[
                        (slice(None), broker1), broker2
                    ]
                    avg_corr = corr_series.mean()

                    if abs(avg_corr) >= correlation_threshold:
                        coordinated_pairs.append({
                            'broker_1': broker1,
                            'broker_2': broker2,
                            'correlation': avg_corr,
                            'coordination_type': 'ALIGNED' if avg_corr > 0 else 'OPPOSING'
                        })
                except:
                    continue

        # Detect synchronized large moves
        # When multiple top brokers make large moves in the same direction
        symbol_data['date'] = pd.to_datetime(symbol_data['date'])

        synchronized_events = []
        for date in symbol_data['date'].unique():
            day_data = symbol_data[
                (symbol_data['date'] == date) &
                (symbol_data['broker_code'].isin(top_brokers))
            ]

            if len(day_data) < 2:
                continue

            # Check if majority are buying or selling strongly
            strong_buyers = (day_data['net_volume'] >
                           day_data['net_volume'].quantile(0.75)).sum()
            strong_sellers = (day_data['net_volume'] <
                            day_data['net_volume'].quantile(0.25)).sum()

            total_brokers = len(day_data)

            if strong_buyers >= total_brokers * 0.6:
                synchronized_events.append({
                    'date': date,
                    'type': 'COORDINATED_BUYING',
                    'num_brokers': strong_buyers,
                    'total_net_volume': day_data['net_volume'].sum()
                })
            elif strong_sellers >= total_brokers * 0.6:
                synchronized_events.append({
                    'date': date,
                    'type': 'COORDINATED_SELLING',
                    'num_brokers': strong_sellers,
                    'total_net_volume': day_data['net_volume'].sum()
                })

        return {
            'symbol': symbol,
            'analyzed_brokers': top_brokers,
            'coordinated_pairs': coordinated_pairs,
            'num_coordinated_pairs': len(coordinated_pairs),
            'synchronized_events': synchronized_events,
            'num_synchronized_events': len(synchronized_events),
            'coordination_score': len(coordinated_pairs) / (top_n * (top_n - 1) / 2) * 100
            if top_n > 1 else 0
        }

    def detect_broker_rotation(self,
                              brokerage_data: pd.DataFrame,
                              symbol: str,
                              window: int = 5) -> pd.DataFrame:
        """
        Detect broker rotation patterns (changing dominant brokers)

        Identifies periods where dominant brokers change, which may
        indicate shifting institutional interest or passing of positions.

        Parameters:
        -----------
        brokerage_data : pd.DataFrame
            Brokerage distribution data
        symbol : str
            Stock symbol to analyze
        window : int, default=5
            Window to identify dominant brokers

        Returns:
        --------
        pd.DataFrame
            DataFrame with broker rotation events
        """
        if brokerage_data.empty:
            return pd.DataFrame()

        symbol_data = brokerage_data[brokerage_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')

        # Find dominant broker for each date
        daily_dominant = []
        for date, group in symbol_data.groupby('date'):
            dominant = group.nlargest(1, 'net_volume').iloc[0]
            daily_dominant.append({
                'date': date,
                'dominant_broker': dominant['broker_code'],
                'net_volume': dominant['net_volume'],
                'market_share': (
                    abs(dominant['net_volume']) /
                    group['net_volume'].abs().sum() * 100
                )
            })

        dominant_df = pd.DataFrame(daily_dominant)
        dominant_df = dominant_df.sort_values('date')

        # Detect changes in dominant broker
        dominant_df['broker_changed'] = (
            dominant_df['dominant_broker'] !=
            dominant_df['dominant_broker'].shift()
        )

        # Calculate rotation frequency
        dominant_df['rotation_frequency'] = (
            dominant_df['broker_changed'].rolling(window).sum()
        )

        # High rotation = unstable/competitive market
        # Low rotation = stable dominant player
        dominant_df['market_stability'] = np.where(
            dominant_df['rotation_frequency'] <= 1,
            'STABLE',
            np.where(dominant_df['rotation_frequency'] <= 2,
                    'MODERATE', 'VOLATILE')
        )

        return dominant_df

    # ========================================================================
    # FLOW MOMENTUM AND TREND DETECTION
    # ========================================================================

    def calculate_flow_momentum(self,
                               flow_data: pd.DataFrame,
                               periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate flow momentum across multiple time periods

        Parameters:
        -----------
        flow_data : pd.DataFrame
            Flow analysis data (from any of the analysis methods)
        periods : List[int], default=[5, 10, 20]
            Periods for momentum calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum indicators
        """
        if flow_data.empty:
            return pd.DataFrame()

        df = flow_data.copy()
        df = df.sort_values(['symbol', 'date'])

        # Determine which flow columns are available
        flow_cols = []
        if 'foreign_flow' in df.columns:
            flow_cols.append('foreign_flow')
        if 'institutional_flow' in df.columns:
            flow_cols.append('institutional_flow')
        if 'net_volume' in df.columns:
            flow_cols.append('net_volume')

        if not flow_cols:
            logger.warning("No flow columns found for momentum calculation")
            return df

        results = []
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()

            for col in flow_cols:
                for period in periods:
                    # Rate of change
                    symbol_data[f'{col}_roc_{period}'] = (
                        symbol_data[col].pct_change(period) * 100
                    )

                    # Momentum (sum of flows)
                    symbol_data[f'{col}_momentum_{period}'] = (
                        symbol_data[col].rolling(period).sum()
                    )

                    # Acceleration (change in momentum)
                    symbol_data[f'{col}_acceleration_{period}'] = (
                        symbol_data[f'{col}_momentum_{period}'].diff()
                    )

            results.append(symbol_data)

        return pd.concat(results, ignore_index=True)

    # ========================================================================
    # COMPREHENSIVE FLOW REPORT
    # ========================================================================

    def generate_comprehensive_flow_report(self,
                                          ownership_data: pd.DataFrame,
                                          brokerage_data: pd.DataFrame,
                                          symbols: Optional[List[str]] = None,
                                          analysis_window: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive flow analysis report

        Combines all flow analysis methods into a single comprehensive report.

        Parameters:
        -----------
        ownership_data : pd.DataFrame
            Ownership data
        brokerage_data : pd.DataFrame
            Brokerage distribution data
        symbols : List[str], optional
            List of symbols to analyze (if None, analyze all)
        analysis_window : int, default=20
            Window for rolling calculations

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing multiple analysis results
        """
        logger.info("Generating comprehensive flow report")

        report = {}

        # Foreign vs Local Analysis
        if not ownership_data.empty:
            report['foreign_local_flow'] = self.analyze_foreign_local_flow(
                ownership_data, brokerage_data, analysis_window
            )

            # Institutional vs Retail Analysis
            report['institutional_retail_flow'] = (
                self.analyze_institutional_retail_flow(
                    ownership_data, brokerage_data, analysis_window
                )
            )

            # Smart money signals
            if not report['institutional_retail_flow'].empty:
                report['smart_money_signals'] = self.identify_smart_money_signals(
                    report['institutional_retail_flow']
                )

        # Broker Concentration Analysis
        if not brokerage_data.empty:
            report['broker_concentration'] = self.analyze_broker_concentration(
                brokerage_data, analysis_window
            )

            # Analyze each symbol for coordination
            if symbols is None:
                symbols = brokerage_data['symbol'].unique()

            coordination_results = {}
            for symbol in symbols:
                coordination_results[symbol] = self.identify_broker_coordination(
                    brokerage_data, symbol, top_n=5, window=analysis_window
                )

            report['broker_coordination'] = coordination_results

        logger.info("Comprehensive flow report generated")

        return report


# Convenience functions

def analyze_foreign_flow(ownership_data: pd.DataFrame,
                        window: int = 20) -> pd.DataFrame:
    """
    Quick function to analyze foreign vs local flows

    Parameters:
    -----------
    ownership_data : pd.DataFrame
        Ownership data
    window : int, default=20
        Analysis window

    Returns:
    --------
    pd.DataFrame
        Foreign/local flow analysis
    """
    analyzer = FlowAnalyzer()
    return analyzer.analyze_foreign_local_flow(ownership_data, window=window)


def analyze_institutional_flow(ownership_data: pd.DataFrame,
                               window: int = 20) -> pd.DataFrame:
    """
    Quick function to analyze institutional vs retail flows

    Parameters:
    -----------
    ownership_data : pd.DataFrame
        Ownership data
    window : int, default=20
        Analysis window

    Returns:
    --------
    pd.DataFrame
        Institutional/retail flow analysis
    """
    analyzer = FlowAnalyzer()
    return analyzer.analyze_institutional_retail_flow(ownership_data, window=window)


def analyze_broker_concentration(brokerage_data: pd.DataFrame,
                                window: int = 20) -> pd.DataFrame:
    """
    Quick function to analyze broker concentration

    Parameters:
    -----------
    brokerage_data : pd.DataFrame
        Brokerage distribution data
    window : int, default=20
        Analysis window

    Returns:
    --------
    pd.DataFrame
        Broker concentration metrics
    """
    analyzer = FlowAnalyzer()
    return analyzer.analyze_broker_concentration(brokerage_data, window)


if __name__ == "__main__":
    # Example usage
    print("Flow Analysis Module for BIST AI Trading System")
    print("=" * 60)
    print("\nAvailable Analysis Methods:")
    print("1. Foreign vs Local Flow Analysis")
    print("   - Track foreign investor flows and trends")
    print("   - Detect accumulation/distribution phases")
    print("   - Calculate foreign ownership momentum")
    print()
    print("2. Institutional vs Retail Flow Analysis")
    print("   - Identify smart money movements")
    print("   - Detect institutional conviction")
    print("   - Measure momentum divergence")
    print()
    print("3. Broker Concentration Analysis")
    print("   - Calculate HHI and concentration ratios")
    print("   - Identify dominant brokers")
    print("   - Detect broker coordination patterns")
    print("   - Measure market fragmentation")
    print()
    print("4. Flow Momentum & Trends")
    print("   - Multi-period momentum indicators")
    print("   - Flow acceleration metrics")
    print("   - Trend detection algorithms")
    print()
    print("=" * 60)

    # Example initialization
    analyzer = FlowAnalyzer(min_sample_size=5)
    print(f"\nFlowAnalyzer initialized with minimum sample size: 5")
    print("\nReady for analysis!")
