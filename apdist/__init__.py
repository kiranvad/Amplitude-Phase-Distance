"""
Amplitude-Phase Distance Package

A light-weight repository to compute Amplitude Phase distance between two functions.
"""

from .distances import AmplitudePhaseDistance, _amplitude_distance, _phase_distance
from .geometry import SquareRootSlopeFramework, WarpingManifold
from ._version import __version__, __author__, __email__, __description__, __url__

__all__ = [
    "AmplitudePhaseDistance",
    "_amplitude_distance",
    "_phase_distance",
    "SquareRootSlopeFramework",
    "WarpingManifold"
]