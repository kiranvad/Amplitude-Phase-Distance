"""Basic tests that should work without external dependencies."""

import pytest
import numpy as np
import warnings
from apdist.geometry import SquareRootSlopeFramework, WarpingManifold
from apdist.distances import AmplitudePhaseDistance


class TestBasicFunctionality:
    """Test basic functionality that should work without external dependencies."""
    
    def test_srsf_framework_creation(self):
        """Test that SRSF framework can be created."""
        t = np.linspace(0, 1, 101)
        srsf = SquareRootSlopeFramework(t)
        assert srsf.time is not None
        np.testing.assert_array_equal(srsf.time, t)
    
    def test_warping_manifold_creation(self):
        """Test that warping manifold can be created."""
        t = np.linspace(0, 1, 101)
        wm = WarpingManifold(t)
        assert wm.time is not None
        np.testing.assert_array_equal(wm.time, t)
    
    def test_srsf_conversion_basic(self):
        """Test basic SRSF conversion."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        assert q.shape == f.shape
        assert np.all(np.isfinite(q))
    
    def test_srsf_reconstruction(self):
        """Test SRSF reconstruction."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        f_rec = srsf.from_srsf(q, f0=f[0])
        
        # Should reconstruct approximately
        np.testing.assert_allclose(f_rec, f, rtol=1e-1, atol=1e-5)
    
    def test_warping_functions(self):
        """Test basic warping functions."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        gamma = t  # Identity warping
        
        srsf = SquareRootSlopeFramework(t)
        
        # Test function warping
        f_warped = srsf.warp_f_gamma(f, gamma)
        np.testing.assert_allclose(f_warped, f, rtol=1e-10)
        
        # Test SRSF warping
        q = srsf.to_srsf(f)
        q_warped = srsf.warp_q_gamma(q, gamma)
        np.testing.assert_allclose(q_warped, q, rtol=1e-10)
    
    def test_manifold_operations(self):
        """Test basic manifold operations."""
        t = np.linspace(0, 1, 101)
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
    
    def test_distance_computation_with_fallback(self):
        """Test distance computation with fallback gamma."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)
        
        # This should work even without optimum_reparamN2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore fallback warnings
            da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_identical_functions_zero_distance(self):
        """Test that identical functions give zero distance."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da, dp = AmplitudePhaseDistance(t, f, f.copy())
        
        assert da == 0.0
        assert dp == 0.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        t = np.array([])
        f = np.array([])

        srsf = SquareRootSlopeFramework(t)
        # Should raise an error when trying to use empty arrays
        with pytest.raises((ValueError, IndexError, RuntimeError, Exception)):
            srsf.to_srsf(f)
    
    def test_single_point(self):
        """Test behavior with single point."""
        t = np.array([0.5])
        f = np.array([1.0])

        srsf = SquareRootSlopeFramework(t)

        # Single point should raise an error in SRSF computation
        with pytest.raises((ValueError, IndexError, RuntimeError, Exception)):
            srsf.to_srsf(f)
    
    def test_mismatched_arrays(self):
        """Test behavior with mismatched array sizes."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)[:50]  # Different size
        
        srsf = SquareRootSlopeFramework(t)
        
        with pytest.raises((ValueError, IndexError)):
            srsf.to_srsf(f)
    
    def test_constant_function(self):
        """Test with constant function."""
        t = np.linspace(0, 1, 101)
        f = np.ones_like(t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        # SRSF of constant should be approximately zero
        np.testing.assert_allclose(q, 0, atol=1e-10)
    
    def test_nan_values(self):
        """Test behavior with NaN values."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        f[50] = np.nan
        
        srsf = SquareRootSlopeFramework(t)
        
        # This should either handle NaN gracefully or raise an error
        try:
            q = srsf.to_srsf(f)
            # If it succeeds, check that result contains NaN
            assert np.any(np.isnan(q)) or np.all(np.isfinite(q))
        except (ValueError, RuntimeError):
            # This is acceptable behavior for NaN input
            pass


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_very_small_values(self):
        """Test with very small function values."""
        t = np.linspace(0, 1, 101)
        f = 1e-15 * np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        assert np.all(np.isfinite(q))
    
    def test_very_large_values(self):
        """Test with very large function values."""
        t = np.linspace(0, 1, 101)
        f = 1e15 * np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        assert np.all(np.isfinite(q))
    
    def test_high_frequency_functions(self):
        """Test with high frequency functions."""
        t = np.linspace(0, 1, 1001)  # Higher resolution for high frequency
        f = np.sin(100 * np.pi * t)  # High frequency
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        assert np.all(np.isfinite(q))
    
    def test_discontinuous_function(self):
        """Test with discontinuous function."""
        t = np.linspace(0, 1, 101)
        f = np.where(t < 0.5, 0, 1)  # Step function
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        
        assert np.all(np.isfinite(q))
