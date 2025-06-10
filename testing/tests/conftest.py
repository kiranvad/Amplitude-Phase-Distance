"""Pytest configuration and shared fixtures for apdist tests."""

import pytest
import numpy as np
from apdist.geometry import SquareRootSlopeFramework, WarpingManifold


@pytest.fixture
def time_domain():
    """Standard time domain for testing."""
    return np.linspace(0, 1, 101)


@pytest.fixture
def simple_functions(time_domain):
    """Simple test functions for basic testing."""
    t = time_domain
    f1 = np.sin(2 * np.pi * t)
    f2 = np.sin(2 * np.pi * t + np.pi/4)  # Phase shifted
    return f1, f2


@pytest.fixture
def identical_functions(time_domain):
    """Identical functions for zero distance testing."""
    t = time_domain
    f = np.sin(2 * np.pi * t)
    return f, f.copy()


@pytest.fixture
def polynomial_functions(time_domain):
    """Polynomial test functions."""
    t = time_domain
    f1 = t**2
    f2 = (t - 0.2)**2 + 0.1
    return f1, f2


@pytest.fixture
def srsf_framework(time_domain):
    """SRSF framework instance."""
    return SquareRootSlopeFramework(time_domain)


@pytest.fixture
def warping_manifold(time_domain):
    """Warping manifold instance."""
    return WarpingManifold(time_domain)


@pytest.fixture
def identity_warping(time_domain):
    """Identity warping function."""
    return time_domain.copy()


@pytest.fixture
def simple_warping(time_domain):
    """Simple non-identity warping function."""
    t = time_domain
    # Smooth monotonic warping
    return t + 0.1 * np.sin(2 * np.pi * t) * t * (1 - t)
