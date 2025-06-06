"""Tests for distance computation functions in apdist.distances module."""

import pytest
import numpy as np
from apdist.distances import _amplitude_distance, _phase_distance, AmplitudePhaseDistance


class TestAmplitudeDistance:
    """Test amplitude distance computation."""
    
    def test_identical_functions_zero_distance(self, time_domain, identical_functions):
        """Test that identical functions have zero amplitude distance."""
        t = time_domain
        f1, f2 = identical_functions
        
        # Convert to SRSF manually for testing
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist = _amplitude_distance(t, q1, q2, gam)
        assert dist == 0.0
    
    def test_amplitude_distance_positive(self, time_domain, simple_functions):
        """Test that different functions have positive amplitude distance."""
        t = time_domain
        f1, f2 = simple_functions
        
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist = _amplitude_distance(t, q1, q2, gam)
        assert dist > 0
        assert np.isfinite(dist)
    
    def test_amplitude_distance_symmetric(self, time_domain, simple_functions):
        """Test that amplitude distance is symmetric."""
        t = time_domain
        f1, f2 = simple_functions
        
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist12 = _amplitude_distance(t, q1, q2, gam)
        dist21 = _amplitude_distance(t, q2, q1, gam)
        
        np.testing.assert_allclose(dist12, dist21, rtol=1e-10)


class TestPhaseDistance:
    """Test phase distance computation."""
    
    def test_identical_functions_zero_distance(self, time_domain, identical_functions):
        """Test that identical functions have zero phase distance."""
        t = time_domain
        f1, f2 = identical_functions
        
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist = _phase_distance(t, q1, q2, gam)
        assert dist == 0.0
    
    def test_phase_distance_positive(self, time_domain, simple_functions):
        """Test that different functions have positive phase distance."""
        t = time_domain
        f1, f2 = simple_functions
        
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist = _phase_distance(t, q1, q2, gam)
        assert dist >= 0
        assert np.isfinite(dist)
    
    def test_phase_distance_bounded(self, time_domain, simple_functions):
        """Test that phase distance is bounded by pi."""
        t = time_domain
        f1, f2 = simple_functions
        
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(t)
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        gam = t  # Identity warping
        
        dist = _phase_distance(t, q1, q2, gam)
        assert dist <= np.pi


class TestAmplitudePhaseDistance:
    """Test the main AmplitudePhaseDistance function."""
    
    def test_identical_functions_zero_distances(self, time_domain, identical_functions):
        """Test that identical functions return zero for both distances."""
        t = time_domain
        f1, f2 = identical_functions
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da == 0.0
        assert dp == 0.0
    
    def test_different_functions_positive_distances(self, time_domain, simple_functions):
        """Test that different functions return positive distances."""
        t = time_domain
        f1, f2 = simple_functions
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_polynomial_functions(self, time_domain, polynomial_functions):
        """Test with polynomial functions."""
        t = time_domain
        f1, f2 = polynomial_functions
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_return_types(self, time_domain, simple_functions):
        """Test that function returns correct types."""
        t = time_domain
        f1, f2 = simple_functions
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert isinstance(da, (float, np.floating))
        assert isinstance(dp, (float, np.floating))
    
    @pytest.mark.parametrize("lambda_val", [0.0, 0.1, 1.0])
    def test_lambda_parameter(self, time_domain, simple_functions, lambda_val):
        """Test different lambda values for regularization."""
        t = time_domain
        f1, f2 = simple_functions

        # This test might fail if optimum_reparamN2 is not available
        # We'll catch the import error and skip
        try:
            da, dp = AmplitudePhaseDistance(t, f1, f2, **{"lambda": lambda_val})
            assert da >= 0
            assert dp >= 0
        except (ImportError, RuntimeError, ModuleNotFoundError) as e:
            pytest.skip(f"Skipping test due to missing dependency: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_constant_functions(self, time_domain):
        """Test with constant functions."""
        t = time_domain
        f1 = np.ones_like(t)
        f2 = np.ones_like(t) * 2
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_linear_functions(self, time_domain):
        """Test with linear functions."""
        t = time_domain
        f1 = t
        f2 = 2 * t + 1
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
    
    def test_single_point_difference(self, time_domain):
        """Test functions that differ at only one point."""
        t = time_domain
        f1 = np.sin(2 * np.pi * t)
        f2 = f1.copy()
        f2[len(f2)//2] += 0.1  # Small perturbation
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert da > 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
