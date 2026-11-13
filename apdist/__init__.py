"""
Amplitude-Phase Distance Package

A light-weight repository to compute Amplitude Phase distance between two functions.
"""

import warnings

try:
    # Generated at build/install time by setuptools-scm from git tags
    from ._version import __version__, __version_tuple__
except ImportError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "unknown")

__author__ = "Kiran Vaddi"
__email__ = "kiranvad@uw.edu"
__description__ = "Amplitude-Phase distance between functions"
__url__ = "https://github.com/kiranvad/Amplitude-Phase-Distance"

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
