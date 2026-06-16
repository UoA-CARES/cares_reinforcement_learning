"""Public package surface for CARES RL statistics analysis.

Import run/metric dataclasses from data and validation configuration/types
from validation, then import module-specific helpers from their owning modules
when you need lower-level behavior.
"""

from .data import AlgorithmRun, MetricDirection, MetricSpec, SeedMetricSummary, SeedRun
from .task_analysis import run_task_analysis
from .validation import (
    AnalysisOptions,
    ValidationIssue,
    ValidationMode,
    ValidationPolicy,
    ValidationSeverity,
)

__all__ = [
    "AlgorithmRun",
    "AnalysisOptions",
    "MetricDirection",
    "MetricSpec",
    "SeedMetricSummary",
    "SeedRun",
    "ValidationIssue",
    "ValidationMode",
    "ValidationPolicy",
    "ValidationSeverity",
    "run_task_analysis",
]
