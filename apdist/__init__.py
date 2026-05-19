"""
Amplitude-Phase Distance Package

A light-weight repository to compute Amplitude Phase distance between two functions.
"""

import warnings

from ._version import __version__, __author__, __email__, __description__, __url__

import apdist.distances
import apdist.geometry
import apdist.utils

try:
    import apdist.torch
except ImportError as exc:
    warnings.warn(
        f"Optional torch support is unavailable: {exc}",
        ImportWarning,
    )
