"""
Report Generation Module for BIST AI Trading System

This module generates comprehensive PDF and HTML reports including:
- Trading signal analysis with detailed metrics
- Backtest results with performance statistics
- Portfolio performance summary with P&L breakdown
- Visualizations and charts

The module supports:
- PDF generation using ReportLab
- HTML generation using Jinja2 templates
- Multiple report types and formats
- Customizable styling and branding
- Chart integration with matplotlib
- Export to various formats

Features:
- Professional report layouts
- Interactive HTML reports
- Static PDF reports for archival
- Signal prioritization tables
- Performance metrics visualization
- Risk analysis charts
- Portfolio allocation breakdown

Author: BIST AI Trading System
Date: 2025-11-16
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, PageBreak, Image, Frame, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF generation will be disabled.")

# HTML generation
try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available. HTML template generation will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# Report Data Classes
# ============================================================================

class ReportConfig:
    """Configuration for report generation"""

    def __init__(
        self,
        title: str = "BIST AI Trading System Report",
        author: str = "BIST AI Trading System",
        subject: str = "Trading Analysis Report",
        page_size: str = "A4",
        include_charts: bool = True,
        include_metadata: bool = True,
        logo_path: Optional[str] = None,
        custom_css: Optional[str] = None
    ):
        self.title = title
        self.author = author
        self.subject = subject
        self.page_size = A4 if page_size == "A4" else letter
        self.include_charts = include_charts
        self.include_metadata = include_metadata
        self.logo_path = logo_path
        self.custom_css = custom_css


# ============================================================================
# Base Report Generator
# ============================================================================

class ReportGenerator:
    """
    Base class for report generation

    Provides common functionality for both PDF and HTML reports.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.generated_charts = []

    def _create_chart_signal_distribution(
        self,
        signals: List[Dict[str, Any]]
    ) -> str:
        """
        Create signal distribution chart

        Args:
            signals: List of signal dictionaries

        Returns:
            Base64 encoded image string
        """
        if not signals:
            return ""

        # Count signals by type
        signal_counts = {}
        for signal in signals:
            signal_type = signal.get('signal', 'UNKNOWN')
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

        # Create chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_map = {
            'STRONG_BUY': '#00AA00',
            'BUY': '#66CC66',
            'HOLD': '#CCCCCC',
            'SELL': '#CC6666',
            'STRONG_SELL': '#AA0000'
        }

        signal_types = list(signal_counts.keys())
        counts = list(signal_counts.values())
        bar_colors = [colors_map.get(s, '#888888') for s in signal_types]

        ax.bar(signal_types, counts, color=bar_colors, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Signal Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Signal Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str

    def _create_chart_confidence_distribution(
        self,
        signals: List[Dict[str, Any]]
    ) -> str:
        """
        Create confidence score distribution chart

        Args:
            signals: List of signal dictionaries

        Returns:
            Base64 encoded image string
        """
        if not signals:
            return ""

        confidence_scores = [s.get('confidence_score', 0) for s in signals]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(confidence_scores, bins=20, color='#4CAF50', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_conf = np.mean(confidence_scores)
        ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.2f}')
        ax.legend()

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str

    def _create_chart_equity_curve(
        self,
        backtest_results: Dict[str, Any]
    ) -> str:
        """
        Create equity curve chart

        Args:
            backtest_results: Backtest results dictionary

        Returns:
            Base64 encoded image string
        """
        equity_curve = backtest_results.get('equity_curve', [])
        if not equity_curve:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(equity_curve, color='#2196F3', linewidth=2)
        ax.fill_between(range(len(equity_curve)), equity_curve, alpha=0.3, color='#2196F3')
        ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value', fontsize=12, fontweight='bold')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add drawdown shading
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str

    def _create_chart_drawdown(
        self,
        backtest_results: Dict[str, Any]
    ) -> str:
        """
        Create drawdown chart

        Args:
            backtest_results: Backtest results dictionary

        Returns:
            Base64 encoded image string
        """
        equity_curve = backtest_results.get('equity_curve', [])
        if not equity_curve:
            return ""

        # Calculate drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(range(len(drawdown)), drawdown, 0,
                        where=(drawdown < 0), color='#F44336', alpha=0.5)
        ax.plot(drawdown, color='#D32F2F', linewidth=2)
        ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str

    def _create_chart_portfolio_allocation(
        self,
        portfolio_data: Dict[str, Any]
    ) -> str:
        """
        Create portfolio allocation pie chart

        Args:
            portfolio_data: Portfolio summary data

        Returns:
            Base64 encoded image string
        """
        positions = portfolio_data.get('positions', [])
        if not positions:
            return ""

        # Extract symbols and values
        symbols = [p['symbol'] for p in positions[:10]]  # Top 10
        values = [p.get('market_value', 0) for p in positions[:10]]

        fig, ax = plt.subplots(figsize=(8, 8))
        colors_palette = sns.color_palette("husl", len(symbols))

        wedges, texts, autotexts = ax.pie(
            values,
            labels=symbols,
            autopct='%1.1f%%',
            colors=colors_palette,
            startangle=90,
            textprops={'fontsize': 10}
        )

        ax.set_title('Portfolio Allocation (Top 10 Positions)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str

    def _create_chart_returns_distribution(
        self,
        backtest_results: Dict[str, Any]
    ) -> str:
        """
        Create returns distribution chart

        Args:
            backtest_results: Backtest results dictionary

        Returns:
            Base64 encoded image string
        """
        returns = backtest_results.get('returns', [])
        if not returns:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(returns, bins=30, color='#9C27B0', edgecolor='black', alpha=0.7)

        # Add normal distribution curve
        mu, sigma = np.mean(returns), np.std(returns)
        x = np.linspace(min(returns), max(returns), 100)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mu, sigma) * len(returns) * (max(returns) - min(returns)) / 30,
                'r-', linewidth=2, label='Normal Distribution')

        ax.set_xlabel('Return', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

        # Add statistics
        textstr = f'Mean: {mu:.4f}\nStd: {sigma:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_str


# ============================================================================
# PDF Report Generator
# ============================================================================

class PDFReportGenerator(ReportGenerator):
    """
    PDF report generator using ReportLab

    Generates professional PDF reports with tables, charts, and formatting.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize PDF report generator"""
        super().__init__(config)

        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install it with: pip install reportlab"
            )

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=12,
            spaceBefore=12,
            borderWidth=1,
            borderColor=colors.HexColor('#1976D2'),
            borderPadding=5,
            backColor=colors.HexColor('#E3F2FD')
        ))

        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#424242'),
            spaceAfter=10,
            spaceBefore=10
        ))

    def generate_signal_report(
        self,
        signals: List[Dict[str, Any]],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Generate PDF report for trading signals

        Args:
            signals: List of signal dictionaries
            output_path: Path to save PDF
            metadata: Optional metadata to include
        """
        logger.info(f"Generating signal PDF report: {output_path}")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.config.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
            title=self.config.title,
            author=self.config.author
        )

        story = []

        # Title
        story.append(Paragraph(
            "Trading Signals Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 12))

        # Metadata
        if metadata or self.config.include_metadata:
            meta_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            meta_text += f"Total Signals: {len(signals)}<br/>"
            if metadata:
                for key, value in metadata.items():
                    meta_text += f"{key}: {value}<br/>"

            story.append(Paragraph(meta_text, self.styles['Normal']))
            story.append(Spacer(1, 20))

        # Summary statistics
        story.append(Paragraph("Summary Statistics", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        summary_data = self._calculate_signal_summary(signals)
        summary_table = self._create_summary_table(summary_data)
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Charts
        if self.config.include_charts:
            story.append(Paragraph("Signal Distribution", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_signal_distribution(signals)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))

            story.append(Paragraph("Confidence Distribution", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_confidence_distribution(signals)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=5*inch, height=3.5*inch)
                story.append(img)
                story.append(PageBreak())

        # Detailed signals table
        story.append(Paragraph("Detailed Signals", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        signals_table = self._create_signals_table(signals)
        story.append(signals_table)

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report saved to {output_path}")

    def generate_backtest_report(
        self,
        backtest_results: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Generate PDF report for backtest results

        Args:
            backtest_results: Backtest results dictionary
            output_path: Path to save PDF
            metadata: Optional metadata
        """
        logger.info(f"Generating backtest PDF report: {output_path}")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.config.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
            title="Backtest Results Report",
            author=self.config.author
        )

        story = []

        # Title
        story.append(Paragraph(
            "Backtest Results Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 12))

        # Metadata
        if metadata or self.config.include_metadata:
            meta_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            if metadata:
                for key, value in metadata.items():
                    meta_text += f"{key}: {value}<br/>"

            story.append(Paragraph(meta_text, self.styles['Normal']))
            story.append(Spacer(1, 20))

        # Performance metrics
        story.append(Paragraph("Performance Metrics", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        metrics = backtest_results.get('metrics', {})
        metrics_table = self._create_metrics_table(metrics)
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Charts
        if self.config.include_charts:
            # Equity curve
            story.append(Paragraph("Equity Curve", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_equity_curve(backtest_results)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))

            # Drawdown
            story.append(Paragraph("Drawdown Analysis", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_drawdown(backtest_results)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(PageBreak())

            # Returns distribution
            story.append(Paragraph("Returns Distribution", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_returns_distribution(backtest_results)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=6*inch, height=3.5*inch)
                story.append(img)

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report saved to {output_path}")

    def generate_portfolio_report(
        self,
        portfolio_data: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Generate PDF report for portfolio performance

        Args:
            portfolio_data: Portfolio summary data
            output_path: Path to save PDF
            metadata: Optional metadata
        """
        logger.info(f"Generating portfolio PDF report: {output_path}")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.config.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
            title="Portfolio Performance Report",
            author=self.config.author
        )

        story = []

        # Title
        portfolio_name = portfolio_data.get('portfolio_name', 'Portfolio')
        story.append(Paragraph(
            f"{portfolio_name} - Performance Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 12))

        # Metadata
        if metadata or self.config.include_metadata:
            meta_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            meta_text += f"Portfolio: {portfolio_name}<br/>"
            meta_text += f"Last Updated: {portfolio_data.get('last_updated', 'N/A')}<br/>"
            if metadata:
                for key, value in metadata.items():
                    meta_text += f"{key}: {value}<br/>"

            story.append(Paragraph(meta_text, self.styles['Normal']))
            story.append(Spacer(1, 20))

        # Portfolio summary
        story.append(Paragraph("Portfolio Summary", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        summary_table = self._create_portfolio_summary_table(portfolio_data)
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Charts
        if self.config.include_charts:
            story.append(Paragraph("Portfolio Allocation", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            chart_img = self._create_chart_portfolio_allocation(portfolio_data)
            if chart_img:
                img_data = base64.b64decode(chart_img)
                img = Image(io.BytesIO(img_data), width=5*inch, height=5*inch)
                story.append(img)
                story.append(PageBreak())

        # Positions detail
        story.append(Paragraph("Position Details", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        positions_table = self._create_positions_table(portfolio_data.get('positions', []))
        story.append(positions_table)

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report saved to {output_path}")

    def _calculate_signal_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for signals"""
        if not signals:
            return {}

        signal_types = [s.get('signal', 'UNKNOWN') for s in signals]
        confidence_scores = [s.get('confidence_score', 0) for s in signals]

        return {
            'Total Signals': len(signals),
            'Strong Buy': signal_types.count('STRONG_BUY'),
            'Buy': signal_types.count('BUY'),
            'Hold': signal_types.count('HOLD'),
            'Sell': signal_types.count('SELL'),
            'Strong Sell': signal_types.count('STRONG_SELL'),
            'Avg Confidence': f"{np.mean(confidence_scores):.2%}",
            'Min Confidence': f"{np.min(confidence_scores):.2%}",
            'Max Confidence': f"{np.max(confidence_scores):.2%}"
        }

    def _create_summary_table(self, summary_data: Dict[str, Any]) -> Table:
        """Create summary statistics table"""
        data = [['Metric', 'Value']]
        for key, value in summary_data.items():
            data.append([key, str(value)])

        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        return table

    def _create_signals_table(self, signals: List[Dict[str, Any]]) -> Table:
        """Create detailed signals table"""
        # Header
        data = [['Stock', 'Signal', 'Confidence', 'Target Price', 'Expected Return']]

        # Add signal rows (limit to top 20)
        for signal in signals[:20]:
            data.append([
                signal.get('stock_code', 'N/A'),
                signal.get('signal', 'N/A'),
                f"{signal.get('confidence_score', 0):.1%}",
                f"{signal.get('target_price', 0):.2f}" if signal.get('target_price') else 'N/A',
                f"{signal.get('expected_return', 0):.2%}" if signal.get('expected_return') else 'N/A'
            ])

        table = Table(data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        return table

    def _create_metrics_table(self, metrics: Dict[str, Any]) -> Table:
        """Create backtest metrics table"""
        data = [['Metric', 'Value']]

        # Format metrics nicely
        metric_formats = {
            'win_rate': lambda x: f"{x:.2f}%",
            'total_trades': lambda x: str(int(x)),
            'winning_trades': lambda x: str(int(x)),
            'losing_trades': lambda x: str(int(x)),
            'sharpe_ratio': lambda x: f"{x:.3f}",
            'sortino_ratio': lambda x: f"{x:.3f}",
            'max_drawdown_pct': lambda x: f"{x:.2f}%",
            'total_return_pct': lambda x: f"{x:.2f}%",
            'annualized_return': lambda x: f"{x*100:.2f}%",
            'profit_factor': lambda x: f"{x:.3f}",
        }

        for key, value in metrics.items():
            if key in metric_formats:
                formatted_value = metric_formats[key](value)
            else:
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)

            # Make key more readable
            readable_key = key.replace('_', ' ').title()
            data.append([readable_key, formatted_value])

        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        return table

    def _create_portfolio_summary_table(self, portfolio_data: Dict[str, Any]) -> Table:
        """Create portfolio summary table"""
        data = [['Metric', 'Value']]

        summary_items = [
            ('Total Value', f"{portfolio_data.get('total_value', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Cash', f"{portfolio_data.get('cash', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Positions Value', f"{portfolio_data.get('positions_value', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Number of Positions', str(portfolio_data.get('num_positions', 0))),
            ('Unrealized P&L', f"{portfolio_data.get('unrealized_pnl', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Realized P&L', f"{portfolio_data.get('realized_pnl', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Total P&L', f"{portfolio_data.get('total_pnl', 0):,.2f} {portfolio_data.get('currency', 'TRY')}"),
            ('Total Return', f"{portfolio_data.get('total_return_pct', 0):.2f}%"),
        ]

        for key, value in summary_items:
            data.append([key, value])

        table = Table(data, colWidths=[3*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        return table

    def _create_positions_table(self, positions: List[Dict[str, Any]]) -> Table:
        """Create positions detail table"""
        data = [['Symbol', 'Shares', 'Cost Basis', 'Current Price', 'Market Value', 'P&L', 'P&L %']]

        for pos in positions[:15]:  # Limit to 15 positions
            data.append([
                pos.get('symbol', 'N/A'),
                f"{pos.get('shares', 0):.2f}",
                f"{pos.get('cost_basis', 0):.2f}",
                f"{pos.get('current_price', 0):.2f}",
                f"{pos.get('market_value', 0):,.2f}",
                f"{pos.get('unrealized_pnl', 0):,.2f}",
                f"{pos.get('unrealized_pnl_pct', 0):.2f}%"
            ])

        table = Table(data, colWidths=[0.8*inch, 0.8*inch, 0.9*inch, 1*inch, 1.1*inch, 0.9*inch, 0.9*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        return table


# ============================================================================
# HTML Report Generator
# ============================================================================

class HTMLReportGenerator(ReportGenerator):
    """
    HTML report generator using Jinja2 templates

    Generates interactive HTML reports with embedded charts and styling.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize HTML report generator"""
        super().__init__(config)

        if not JINJA2_AVAILABLE:
            logger.warning(
                "Jinja2 is not available. Using basic HTML generation. "
                "Install it with: pip install jinja2 for better templates."
            )

    def generate_signal_report(
        self,
        signals: List[Dict[str, Any]],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Generate HTML report for trading signals"""
        logger.info(f"Generating signal HTML report: {output_path}")

        # Generate charts
        chart_distribution = self._create_chart_signal_distribution(signals)
        chart_confidence = self._create_chart_confidence_distribution(signals)

        # Calculate summary
        summary = self._calculate_signal_summary_dict(signals)

        # Create HTML
        html = self._create_signal_html(
            signals,
            summary,
            chart_distribution,
            chart_confidence,
            metadata
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")

    def generate_backtest_report(
        self,
        backtest_results: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Generate HTML report for backtest results"""
        logger.info(f"Generating backtest HTML report: {output_path}")

        # Generate charts
        chart_equity = self._create_chart_equity_curve(backtest_results)
        chart_drawdown = self._create_chart_drawdown(backtest_results)
        chart_returns = self._create_chart_returns_distribution(backtest_results)

        # Create HTML
        html = self._create_backtest_html(
            backtest_results,
            chart_equity,
            chart_drawdown,
            chart_returns,
            metadata
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")

    def generate_portfolio_report(
        self,
        portfolio_data: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Generate HTML report for portfolio performance"""
        logger.info(f"Generating portfolio HTML report: {output_path}")

        # Generate charts
        chart_allocation = self._create_chart_portfolio_allocation(portfolio_data)

        # Create HTML
        html = self._create_portfolio_html(
            portfolio_data,
            chart_allocation,
            metadata
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")

    def _calculate_signal_summary_dict(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for signals"""
        if not signals:
            return {}

        signal_types = [s.get('signal', 'UNKNOWN') for s in signals]
        confidence_scores = [s.get('confidence_score', 0) for s in signals]

        return {
            'total': len(signals),
            'strong_buy': signal_types.count('STRONG_BUY'),
            'buy': signal_types.count('BUY'),
            'hold': signal_types.count('HOLD'),
            'sell': signal_types.count('SELL'),
            'strong_sell': signal_types.count('STRONG_SELL'),
            'avg_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }

    def _create_signal_html(
        self,
        signals: List[Dict[str, Any]],
        summary: Dict[str, Any],
        chart_distribution: str,
        chart_confidence: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create HTML for signal report"""

        css = self.config.custom_css or """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }
            h1 {
                color: #1976D2;
                border-bottom: 3px solid #1976D2;
                padding-bottom: 10px;
            }
            h2 {
                color: #424242;
                margin-top: 30px;
                border-left: 4px solid #1976D2;
                padding-left: 15px;
            }
            .metadata {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th {
                background-color: #1976D2;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .chart {
                text-align: center;
                margin: 20px 0;
            }
            .chart img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .summary-card {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #1976D2;
            }
            .summary-card h3 {
                margin: 0 0 10px 0;
                color: #424242;
                font-size: 14px;
            }
            .summary-card p {
                margin: 0;
                font-size: 24px;
                font-weight: bold;
                color: #1976D2;
            }
            .signal-STRONG_BUY { color: #00AA00; font-weight: bold; }
            .signal-BUY { color: #66CC66; font-weight: bold; }
            .signal-HOLD { color: #888888; font-weight: bold; }
            .signal-SELL { color: #CC6666; font-weight: bold; }
            .signal-STRONG_SELL { color: #AA0000; font-weight: bold; }
        </style>
        """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Trading Signals Report</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>Trading Signals Report</h1>

                <div class="metadata">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>Total Signals:</strong> {len(signals)}
        """

        if metadata:
            for key, value in metadata.items():
                html += f"<br><strong>{key}:</strong> {value}"

        html += """
                </div>

                <h2>Summary Statistics</h2>
                <div class="summary-grid">
        """

        # Add summary cards
        summary_cards = [
            ('Total Signals', summary.get('total', 0)),
            ('Strong Buy', summary.get('strong_buy', 0)),
            ('Buy', summary.get('buy', 0)),
            ('Hold', summary.get('hold', 0)),
            ('Sell', summary.get('sell', 0)),
            ('Strong Sell', summary.get('strong_sell', 0)),
            ('Avg Confidence', f"{summary.get('avg_confidence', 0):.1%}"),
        ]

        for title, value in summary_cards:
            html += f"""
                    <div class="summary-card">
                        <h3>{title}</h3>
                        <p>{value}</p>
                    </div>
            """

        html += """
                </div>
        """

        # Add charts
        if self.config.include_charts:
            if chart_distribution:
                html += f"""
                <h2>Signal Distribution</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_distribution}" alt="Signal Distribution">
                </div>
                """

            if chart_confidence:
                html += f"""
                <h2>Confidence Distribution</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_confidence}" alt="Confidence Distribution">
                </div>
                """

        # Add signals table
        html += """
                <h2>Detailed Signals</h2>
                <table>
                    <tr>
                        <th>Stock</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                        <th>Target Price</th>
                        <th>Expected Return</th>
                        <th>Rationale</th>
                    </tr>
        """

        for signal in signals[:50]:  # Limit to 50 signals
            signal_type = signal.get('signal', 'UNKNOWN')
            html += f"""
                    <tr>
                        <td>{signal.get('stock_code', 'N/A')}</td>
                        <td class="signal-{signal_type}">{signal_type}</td>
                        <td>{signal.get('confidence_score', 0):.1%}</td>
                        <td>{signal.get('target_price', 0):.2f}</td>
                        <td>{signal.get('expected_return', 0):.2%}</td>
                        <td>{signal.get('rationale', 'N/A')[:100]}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html

    def _create_backtest_html(
        self,
        backtest_results: Dict[str, Any],
        chart_equity: str,
        chart_drawdown: str,
        chart_returns: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create HTML for backtest report"""

        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }
            h1 {
                color: #1976D2;
                border-bottom: 3px solid #1976D2;
                padding-bottom: 10px;
            }
            h2 {
                color: #424242;
                margin-top: 30px;
                border-left: 4px solid #1976D2;
                padding-left: 15px;
            }
            .metadata {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th {
                background-color: #1976D2;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .chart {
                text-align: center;
                margin: 20px 0;
            }
            .chart img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #1976D2;
            }
            .metric-card h3 {
                margin: 0 0 10px 0;
                color: #424242;
                font-size: 14px;
            }
            .metric-card p {
                margin: 0;
                font-size: 24px;
                font-weight: bold;
                color: #1976D2;
            }
        </style>
        """

        metrics = backtest_results.get('metrics', {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Backtest Results Report</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>Backtest Results Report</h1>

                <div class="metadata">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if metadata:
            for key, value in metadata.items():
                html += f"<br><strong>{key}:</strong> {value}"

        html += """
                </div>

                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
        """

        # Add metric cards
        metric_cards = [
            ('Total Return', f"{metrics.get('total_return_pct', 0):.2f}%"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0)*100:.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.3f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.3f}"),
            ('Total Trades', metrics.get('total_trades', 0)),
        ]

        for title, value in metric_cards:
            html += f"""
                    <div class="metric-card">
                        <h3>{title}</h3>
                        <p>{value}</p>
                    </div>
            """

        html += """
                </div>
        """

        # Add charts
        if chart_equity:
            html += f"""
                <h2>Equity Curve</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_equity}" alt="Equity Curve">
                </div>
            """

        if chart_drawdown:
            html += f"""
                <h2>Drawdown Analysis</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_drawdown}" alt="Drawdown">
                </div>
            """

        if chart_returns:
            html += f"""
                <h2>Returns Distribution</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_returns}" alt="Returns Distribution">
                </div>
            """

        # Add detailed metrics table
        html += """
                <h2>Detailed Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """

        for key, value in metrics.items():
            readable_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            html += f"""
                    <tr>
                        <td>{readable_key}</td>
                        <td>{formatted_value}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html

    def _create_portfolio_html(
        self,
        portfolio_data: Dict[str, Any],
        chart_allocation: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create HTML for portfolio report"""

        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }
            h1 {
                color: #1976D2;
                border-bottom: 3px solid #1976D2;
                padding-bottom: 10px;
            }
            h2 {
                color: #424242;
                margin-top: 30px;
                border-left: 4px solid #1976D2;
                padding-left: 15px;
            }
            .metadata {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th {
                background-color: #1976D2;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .chart {
                text-align: center;
                margin: 20px 0;
            }
            .chart img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .summary-card {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #1976D2;
            }
            .summary-card h3 {
                margin: 0 0 10px 0;
                color: #424242;
                font-size: 14px;
            }
            .summary-card p {
                margin: 0;
                font-size: 20px;
                font-weight: bold;
                color: #1976D2;
            }
            .positive { color: #00AA00; }
            .negative { color: #AA0000; }
        </style>
        """

        portfolio_name = portfolio_data.get('portfolio_name', 'Portfolio')
        currency = portfolio_data.get('currency', 'TRY')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{portfolio_name} - Performance Report</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>{portfolio_name} - Performance Report</h1>

                <div class="metadata">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>Last Updated:</strong> {portfolio_data.get('last_updated', 'N/A')}
        """

        if metadata:
            for key, value in metadata.items():
                html += f"<br><strong>{key}:</strong> {value}"

        html += """
                </div>

                <h2>Portfolio Summary</h2>
                <div class="summary-grid">
        """

        # Add summary cards
        summary_cards = [
            ('Total Value', f"{portfolio_data.get('total_value', 0):,.2f} {currency}"),
            ('Cash', f"{portfolio_data.get('cash', 0):,.2f} {currency}"),
            ('Positions Value', f"{portfolio_data.get('positions_value', 0):,.2f} {currency}"),
            ('Number of Positions', portfolio_data.get('num_positions', 0)),
            ('Total P&L', f"{portfolio_data.get('total_pnl', 0):,.2f} {currency}"),
            ('Total Return', f"{portfolio_data.get('total_return_pct', 0):.2f}%"),
        ]

        for title, value in summary_cards:
            html += f"""
                    <div class="summary-card">
                        <h3>{title}</h3>
                        <p>{value}</p>
                    </div>
            """

        html += """
                </div>
        """

        # Add allocation chart
        if chart_allocation:
            html += f"""
                <h2>Portfolio Allocation</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{chart_allocation}" alt="Portfolio Allocation">
                </div>
            """

        # Add positions table
        positions = portfolio_data.get('positions', [])
        if positions:
            html += """
                <h2>Position Details</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Cost Basis</th>
                        <th>Current Price</th>
                        <th>Market Value</th>
                        <th>Unrealized P&L</th>
                        <th>P&L %</th>
                    </tr>
            """

            for pos in positions:
                pnl_class = 'positive' if pos.get('unrealized_pnl', 0) >= 0 else 'negative'
                html += f"""
                    <tr>
                        <td>{pos.get('symbol', 'N/A')}</td>
                        <td>{pos.get('shares', 0):.2f}</td>
                        <td>{pos.get('cost_basis', 0):.2f}</td>
                        <td>{pos.get('current_price', 0):.2f}</td>
                        <td>{pos.get('market_value', 0):,.2f}</td>
                        <td class="{pnl_class}">{pos.get('unrealized_pnl', 0):,.2f}</td>
                        <td class="{pnl_class}">{pos.get('unrealized_pnl_pct', 0):.2f}%</td>
                    </tr>
                """

            html += """
                </table>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_pdf_signal_report(
    signals: List[Dict[str, Any]],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate PDF signal report

    Args:
        signals: List of signal dictionaries
        output_path: Path to save PDF
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = PDFReportGenerator(config)
    generator.generate_signal_report(signals, output_path, metadata)


def generate_html_signal_report(
    signals: List[Dict[str, Any]],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate HTML signal report

    Args:
        signals: List of signal dictionaries
        output_path: Path to save HTML
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = HTMLReportGenerator(config)
    generator.generate_signal_report(signals, output_path, metadata)


def generate_pdf_backtest_report(
    backtest_results: Dict[str, Any],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate PDF backtest report

    Args:
        backtest_results: Backtest results dictionary
        output_path: Path to save PDF
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = PDFReportGenerator(config)
    generator.generate_backtest_report(backtest_results, output_path, metadata)


def generate_html_backtest_report(
    backtest_results: Dict[str, Any],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate HTML backtest report

    Args:
        backtest_results: Backtest results dictionary
        output_path: Path to save HTML
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = HTMLReportGenerator(config)
    generator.generate_backtest_report(backtest_results, output_path, metadata)


def generate_pdf_portfolio_report(
    portfolio_data: Dict[str, Any],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate PDF portfolio report

    Args:
        portfolio_data: Portfolio summary data
        output_path: Path to save PDF
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = PDFReportGenerator(config)
    generator.generate_portfolio_report(portfolio_data, output_path, metadata)


def generate_html_portfolio_report(
    portfolio_data: Dict[str, Any],
    output_path: str,
    config: Optional[ReportConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to generate HTML portfolio report

    Args:
        portfolio_data: Portfolio summary data
        output_path: Path to save HTML
        config: Optional report configuration
        metadata: Optional metadata
    """
    generator = HTMLReportGenerator(config)
    generator.generate_portfolio_report(portfolio_data, output_path, metadata)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BIST Report Generator - Example Usage")
    print("=" * 80)

    # Example 1: Signal Report
    print("\n1. Generating Signal Report")
    print("-" * 80)

    sample_signals = [
        {
            'stock_code': 'THYAO',
            'signal': 'STRONG_BUY',
            'confidence_score': 0.85,
            'target_price': 275.0,
            'expected_return': 0.05,
            'rationale': 'Strong institutional buying with positive sentiment'
        },
        {
            'stock_code': 'GARAN',
            'signal': 'BUY',
            'confidence_score': 0.72,
            'target_price': 92.0,
            'expected_return': 0.03,
            'rationale': 'Technical breakout with good fundamentals'
        },
        {
            'stock_code': 'EREGL',
            'signal': 'HOLD',
            'confidence_score': 0.55,
            'target_price': 45.0,
            'expected_return': 0.01,
            'rationale': 'Mixed signals, consolidation expected'
        }
    ]

    # Generate HTML report
    generate_html_signal_report(
        signals=sample_signals,
        output_path='signal_report.html',
        metadata={'timeframe': '30min', 'date': '2025-11-16'}
    )
    print("HTML signal report generated: signal_report.html")

    # Generate PDF report (if reportlab available)
    if REPORTLAB_AVAILABLE:
        generate_pdf_signal_report(
            signals=sample_signals,
            output_path='signal_report.pdf',
            metadata={'timeframe': '30min', 'date': '2025-11-16'}
        )
        print("PDF signal report generated: signal_report.pdf")

    # Example 2: Backtest Report
    print("\n2. Generating Backtest Report")
    print("-" * 80)

    sample_backtest = {
        'metrics': {
            'win_rate': 65.5,
            'total_trades': 100,
            'winning_trades': 65,
            'losing_trades': 35,
            'sharpe_ratio': 1.85,
            'sortino_ratio': 2.15,
            'max_drawdown_pct': 12.5,
            'total_return_pct': 45.2,
            'annualized_return': 0.352,
            'profit_factor': 2.15
        },
        'equity_curve': list(np.cumsum(np.random.randn(100) * 100 + 50) + 100000),
        'returns': list(np.random.randn(100) * 0.02 + 0.005)
    }

    generate_html_backtest_report(
        backtest_results=sample_backtest,
        output_path='backtest_report.html',
        metadata={'strategy': 'ML Signal Following', 'period': '2023-2025'}
    )
    print("HTML backtest report generated: backtest_report.html")

    # Example 3: Portfolio Report
    print("\n3. Generating Portfolio Report")
    print("-" * 80)

    sample_portfolio = {
        'portfolio_name': 'BIST Trading Portfolio',
        'currency': 'TRY',
        'total_value': 125000.0,
        'cash': 15000.0,
        'positions_value': 110000.0,
        'num_positions': 5,
        'unrealized_pnl': 10000.0,
        'realized_pnl': 5000.0,
        'total_pnl': 15000.0,
        'total_return_pct': 15.0,
        'last_updated': '2025-11-16T10:30:00',
        'positions': [
            {
                'symbol': 'THYAO',
                'shares': 100,
                'cost_basis': 250.0,
                'current_price': 265.0,
                'market_value': 26500.0,
                'unrealized_pnl': 1500.0,
                'unrealized_pnl_pct': 6.0
            },
            {
                'symbol': 'GARAN',
                'shares': 200,
                'cost_basis': 85.0,
                'current_price': 90.0,
                'market_value': 18000.0,
                'unrealized_pnl': 1000.0,
                'unrealized_pnl_pct': 5.88
            }
        ]
    }

    generate_html_portfolio_report(
        portfolio_data=sample_portfolio,
        output_path='portfolio_report.html',
        metadata={'as_of': '2025-11-16'}
    )
    print("HTML portfolio report generated: portfolio_report.html")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("Generated files: signal_report.html, backtest_report.html, portfolio_report.html")
    if REPORTLAB_AVAILABLE:
        print("               signal_report.pdf")
    print("=" * 80)
