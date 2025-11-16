"""
Signal Processing and Prioritization Module

This module provides signal generation, prioritization, and ranking capabilities
for the BIST AI Trading System. It combines multiple signal sources (ML models,
whale activity, sentiment analysis) into actionable trading recommendations.

Key Components:
- SignalGenerator: Generate trading signals from model outputs
- SignalPrioritizer: Multi-factor signal ranking engine
- SignalInput: Input data structure for signals
- PrioritizedSignal: Output with ranking and component scores
- Multiple prioritization strategies (balanced, whale-focused, etc.)

Author: BIST AI Trading System
Date: 2025-11-16
"""

# Signal generation
from .generator import (
    SignalGenerator,
    SignalType,
    SignalConfidence,
    ModelOutput,
    TradingSignal,
    create_signal_generator,
    create_model_output
)

# Signal prioritization
from .prioritizer import (
    SignalPrioritizer,
    SignalInput,
    PrioritizedSignal,
    SignalDirection,
    PrioritizationStrategy,
    create_signal_input,
    prioritize_signals
)

# Signal scheduling
from .scheduler import (
    SignalScheduler,
    BISTMarketHours,
    create_default_scheduler
)

__all__ = [
    # Generator
    'SignalGenerator',
    'SignalType',
    'SignalConfidence',
    'ModelOutput',
    'TradingSignal',
    'create_signal_generator',
    'create_model_output',
    # Prioritizer
    'SignalPrioritizer',
    'SignalInput',
    'PrioritizedSignal',
    'SignalDirection',
    'PrioritizationStrategy',
    'create_signal_input',
    'prioritize_signals',
    # Scheduler
    'SignalScheduler',
    'BISTMarketHours',
    'create_default_scheduler'
]

__version__ = '1.0.0'
