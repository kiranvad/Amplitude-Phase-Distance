"""Integration tests for the apdist package."""

import pytest
import numpy as np
from apdist.distances import AmplitudePhaseDistance
from apdist.geometry import SquareRootSlopeFramework, WarpingManifold


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_distance_computation_workflow(self):
        """Test complete workflow from functions to distances."""
        # Create test data
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
        f2 = np.sin(2 * np.pi * (t - 0.1)) + 0.3 * np.sin(6 * np.pi * (t - 0.1))
        
        # Compute distances
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        # Verify results
        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert dp <= np.pi
    
    def test_srsf_roundtrip_workflow(self):
        """Test SRSF conversion and reconstruction workflow."""
        t = np.linspace(0, 1, 101)
        f_original = np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)
        
        # Create SRSF framework
        srsf = SquareRootSlopeFramework(t)
        
        # Convert to SRSF and back
        q = srsf.to_srsf(f_original)
        f_reconstructed = srsf.from_srsf(q, f0=f_original[0])
        
        # Verify reconstruction
        np.testing.assert_allclose(f_reconstructed, f_original, rtol=1e-2)
    
    def test_warping_workflow(self):
        """Test complete warping workflow."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)
        
        # Create SRSF framework
        srsf = SquareRootSlopeFramework(t)
        
        # Convert to SRSF
        q1 = srsf.to_srsf(f1)
        q2 = srsf.to_srsf(f2)
        
        # Apply warping
        gamma = t + 0.1 * np.sin(2 * np.pi * t) * t * (1 - t)
        f2_warped = srsf.warp_f_gamma(f2, gamma)
        q2_warped = srsf.warp_q_gamma(q2, gamma)
        
        # Verify warping properties
        assert f2_warped.shape == f2.shape
        assert q2_warped.shape == q2.shape
        assert np.all(np.isfinite(f2_warped))
        assert np.all(np.isfinite(q2_warped))
    
    def test_manifold_operations_workflow(self):
        """Test warping manifold operations workflow."""
        t = np.linspace(0, 1, 101)
        
        # Create warping manifold
        wm = WarpingManifold(t)
        
        # Create test vectors
        v1 = np.sin(2 * np.pi * t)
        v2 = np.cos(2 * np.pi * t)
        
        # Test manifold operations
        ip = wm.inner_product(v1, v2)
        norm1 = wm.norm(v1)
        norm2 = wm.norm(v2)
        
        # Verify properties
        assert np.isfinite(ip)
        assert norm1 >= 0
        assert norm2 >= 0
        
        # Test Cauchy-Schwarz inequality
        assert abs(ip) <= norm1 * norm2 + 1e-10  # Small tolerance for numerical errors


@pytest.mark.integration
class TestConsistencyChecks:
    """Test consistency between different components."""
    
    def test_distance_symmetry(self):
        """Test that distances are symmetric."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/3)
        
        da12, dp12 = AmplitudePhaseDistance(t, f1, f2)
        da21, dp21 = AmplitudePhaseDistance(t, f2, f1)
        
        # Distances should be symmetric
        np.testing.assert_allclose(da12, da21, rtol=1e-10)
        np.testing.assert_allclose(dp12, dp21, rtol=1e-10)
    
    def test_triangle_inequality_approximation(self):
        """Test approximate triangle inequality for distances."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/6)
        f3 = np.sin(2 * np.pi * t + np.pi/3)
        
        da12, dp12 = AmplitudePhaseDistance(t, f1, f2)
        da23, dp23 = AmplitudePhaseDistance(t, f2, f3)
        da13, dp13 = AmplitudePhaseDistance(t, f1, f3)
        
        # Triangle inequality should hold approximately
        # (may not be exact due to warping optimization)
        assert da13 <= da12 + da23 + 1e-1  # Relaxed tolerance
        assert dp13 <= dp12 + dp23 + 1e-1
    
    def test_zero_distance_consistency(self):
        """Test that zero distances are consistent across methods."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t) + 0.3 * np.cos(4 * np.pi * t)
        
        # Test with identical functions
        da, dp = AmplitudePhaseDistance(t, f, f.copy())
        
        assert da == 0.0
        assert dp == 0.0
    
    def test_warping_consistency(self):
        """Test consistency between function and SRSF warping."""
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        gamma = t + 0.05 * np.sin(2 * np.pi * t) * t * (1 - t)
        
        srsf = SquareRootSlopeFramework(t)
        
        # Warp function and convert to SRSF
        f_warped = srsf.warp_f_gamma(f, gamma)
        q_from_warped_f = srsf.to_srsf(f_warped)
        
        # Convert to SRSF and then warp
        q = srsf.to_srsf(f)
        q_warped = srsf.warp_q_gamma(q, gamma)
        
        # These should be approximately equal
        np.testing.assert_allclose(q_from_warped_f, q_warped, rtol=1e-1)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAndStability:
    """Test performance and numerical stability."""
    
    def test_large_domain_stability(self):
        """Test stability with large domain sizes."""
        t = np.linspace(0, 1, 1001)  # Large domain
        f1 = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
        f2 = np.sin(2 * np.pi * t + np.pi/4) + 0.1 * np.random.randn(len(t))
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
    
    def test_numerical_precision_stability(self):
        """Test stability with different numerical precisions."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + 1e-10)  # Very small difference
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
    
    def test_repeated_computation_consistency(self):
        """Test that repeated computations give consistent results."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)
        
        # Compute distances multiple times
        results = []
        for _ in range(5):
            da, dp = AmplitudePhaseDistance(t, f1, f2)
            results.append((da, dp))
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[i][0], results[0][0], rtol=1e-12)
            np.testing.assert_allclose(results[i][1], results[0][1], rtol=1e-12)


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test scenarios that might occur in real-world usage."""
    
    def test_noisy_signals(self):
        """Test with noisy signals."""
        t = np.linspace(0, 1, 101)
        np.random.seed(42)  # For reproducibility
        
        f1 = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
        f2 = np.sin(2 * np.pi * t + np.pi/4) + 0.1 * np.random.randn(len(t))
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
    
    def test_multimodal_functions(self):
        """Test with multimodal functions."""
        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t) + 0.3 * np.sin(10 * np.pi * t)
        f2 = np.sin(2 * np.pi * (t - 0.1)) + 0.5 * np.sin(6 * np.pi * (t - 0.1)) + 0.3 * np.sin(10 * np.pi * (t - 0.1))
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
    
    def test_step_functions(self):
        """Test with step functions."""
        t = np.linspace(0, 1, 101)
        f1 = np.where(t < 0.5, 0, 1)
        f2 = np.where(t < 0.6, 0, 1)
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
    
    def test_exponential_functions(self):
        """Test with exponential functions."""
        t = np.linspace(0, 1, 101)
        f1 = np.exp(-5 * t)
        f2 = np.exp(-3 * t)
        
        da, dp = AmplitudePhaseDistance(t, f1, f2)
        
        assert np.isfinite(da)
        assert np.isfinite(dp)
        assert da >= 0
        assert dp >= 0
