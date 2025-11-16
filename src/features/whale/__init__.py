"""
Whale (Institutional) Trading Activity Analysis Module

This module provides comprehensive tools for analyzing institutional trading
activity on BIST, including whale activity indices, flow tracking, and
unusual activity detection.
"""

from .activity_index import (
    WhaleActivityIndex,
    calculate_wai,
    detect_unusual_activity,
    generate_whale_signals
)

__all__ = [
    'WhaleActivityIndex',
    'calculate_wai',
    'detect_unusual_activity',
    'generate_whale_signals'
]
