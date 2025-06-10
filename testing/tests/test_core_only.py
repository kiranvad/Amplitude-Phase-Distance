"""
Tests for core functionality without optional dependencies.
These tests should pass with only numpy, scipy, matplotlib installed.
"""

import pytest
import numpy as np
import warnings
from apdist.distances import AmplitudePhaseDistance
from apdist.geometry import SquareRootSlopeFramework, WarpingManifold


class TestCoreWithoutOptionalDeps:
    """Test core functionality without optional dependencies."""
    
    def test_basic_distance_computation(self):
        """Test basic distance computation works without warping package."""
        t = np.linspace(0, 1, 51)  # Smaller for faster testing
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore fallback warnings
            da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert dp <= np.pi  # Phase distance should be bounded by pi
    
    def test_identical_functions_zero_distance(self):
        """Test that identical functions give zero distance."""
        t = np.linspace(0, 1, 51)
        f = np.sin(2 * np.pi * t)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da, dp = AmplitudePhaseDistance(t, f, f.copy())
        
        assert da == 0.0
        assert dp == 0.0
    
    def test_srsf_basic_functionality(self):
        """Test SRSF framework basic functionality."""
        t = np.linspace(0, 1, 51)
        f = np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        
        # Test SRSF conversion
        q = srsf.to_srsf(f)
        assert q.shape == f.shape
        assert np.all(np.isfinite(q))
        
        # Test reconstruction
        f_rec = srsf.from_srsf(q, f0=f[0])
        np.testing.assert_allclose(f_rec, f, rtol=1e-1, atol=1e-5)
    
    def test_srsf_constant_function(self):
        """Test SRSF of constant function."""
        t = np.linspace(0, 1, 51)
        f = np.ones_like(t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        # SRSF of constant function should be zero
        np.testing.assert_allclose(q, 0, atol=1e-10)
    
    def test_warping_manifold_basic(self):
        """Test warping manifold basic operations."""
        t = np.linspace(0, 1, 51)
        v1 = np.sin(2 * np.pi * t)
        v2 = np.cos(2 * np.pi * t)
        
        wm = WarpingManifold(t)
        
        # Test inner product
        ip = wm.inner_product(v1, v2)
        assert np.isfinite(ip)
        
        # Test norm
        norm1 = wm.norm(v1)
        norm2 = wm.norm(v2)
        assert norm1 >= 0
        assert norm2 >= 0
        
        # Test Cauchy-Schwarz inequality
        assert abs(ip) <= norm1 * norm2 + 1e-10
    
    def test_fallback_gamma_computation(self):
        """Test that gamma computation falls back gracefully."""
        t = np.linspace(0, 1, 51)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)
        
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        
        # This should work even without optimum_reparamN2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma = srsf.get_gamma(q1, q2)
        
        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))
        # Without warping package, should return identity or close to it
        assert gamma[0] == t[0]  # Should preserve boundaries
        assert gamma[-1] == t[-1]
    
    def test_warping_functions_identity(self):
        """Test warping functions with identity warping."""
        t = np.linspace(0, 1, 51)
        f = np.sin(2 * np.pi * t)
        gamma = t.copy()  # Identity warping
        
        srsf = SquareRootSlopeFramework(t)
        
        # Test function warping
        f_warped = srsf.warp_f_gamma(f, gamma)
        np.testing.assert_allclose(f_warped, f, rtol=1e-10)
        
        # Test SRSF warping
        q = srsf.to_srsf(f)
        q_warped = srsf.warp_q_gamma(q, gamma)
        np.testing.assert_allclose(q_warped, q, rtol=1e-10)
    
    def test_multiple_function_types(self):
        """Test with different types of functions."""
        t = np.linspace(0, 1, 51)
        
        functions = [
            np.sin(2 * np.pi * t),           # Sinusoidal
            t**2,                            # Polynomial
            np.exp(-5 * t),                  # Exponential
            np.where(t < 0.5, 0, 1),        # Step function
        ]
        
        for i, f1 in enumerate(functions):
            for j, f2 in enumerate(functions):
                if i != j:  # Don't test identical functions
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        da, dp = AmplitudePhaseDistance(t, f1, f2)
                    
                    assert da >= 0, f"Negative amplitude distance for functions {i}, {j}"
                    assert dp >= 0, f"Negative phase distance for functions {i}, {j}"
                    assert np.isfinite(da), f"Non-finite amplitude distance for functions {i}, {j}"
                    assert np.isfinite(dp), f"Non-finite phase distance for functions {i}, {j}"

class TestNumericalStability:
    """Test numerical stability with core functionality only."""
    
    def test_small_values(self):
        """Test with very small function values."""
        t = np.linspace(0, 1, 51)
        f1 = 1e-10 * np.sin(2 * np.pi * t)
        f2 = 1e-10 * np.sin(2 * np.pi * t + np.pi/4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_large_values(self):
        """Test with large function values."""
        t = np.linspace(0, 1, 51)
        f1 = 1e6 * np.sin(2 * np.pi * t)
        f2 = 1e6 * np.sin(2 * np.pi * t + np.pi/4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_edge_case_functions(self):
        """Test edge case functions."""
        t = np.linspace(0, 1, 51)
        
        # Test with functions that might cause numerical issues
        test_cases = [
            (np.ones_like(t), np.ones_like(t) * 2),  # Constants
            (t, 2 * t + 1),                          # Linear
            (np.zeros_like(t), np.ones_like(t)),     # Zero vs constant
        ]
        
        for f1, f2 in test_cases:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                da, dp = AmplitudePhaseDistance(t, f1, f2)
            
            assert np.isfinite(da), f"Non-finite amplitude distance for edge case"
            assert np.isfinite(dp), f"Non-finite phase distance for edge case"
            assert da >= 0, f"Negative amplitude distance for edge case"
            assert dp >= 0, f"Negative phase distance for edge case"
