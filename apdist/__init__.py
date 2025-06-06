"""
Amplitude-Phase Distance Package

A light-weight repository to compute Amplitude Phase distance between two functions.
"""

from .distances import AmplitudePhaseDistance, _amplitude_distance, _phase_distance
from .geometry import SquareRootSlopeFramework, WarpingManifold

__version__ = "1.0"
__author__ = "Kiran Vaddi"
__email__ = "kiranvad@uw.edu"

__all__ = [
    "AmplitudePhaseDistance",
    "_amplitude_distance",
    "_phase_distance",
    "SquareRootSlopeFramework",
    "WarpingManifold"
]