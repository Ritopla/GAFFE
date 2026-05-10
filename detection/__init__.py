"""
GAFFE Detection — Geometric Analysis of Facial Features for Emotion/Stress

Deterministic, explainable stress detection pipeline based on
facial landmark geometry. Uses MediaPipe Face Mesh landmarks to
compute psychophysiological stress metrics frame-by-frame.

This package processes video files and produces structured JSON
output with per-frame stress scores and metric breakdowns.
"""

__version__ = "0.1.0"

from detection.config import StressConfig
from detection.stress_scorer import StressScorer

__all__ = ["StressConfig", "StressScorer"]
