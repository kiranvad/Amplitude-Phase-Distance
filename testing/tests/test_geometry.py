"""Tests for geometry module including SRSF and WarpingManifold classes."""

import pytest
import numpy as np
from apdist.geometry import SquareRootSlopeFramework, WarpingManifold


class TestSquareRootSlopeFramework:
    """Test SquareRootSlopeFramework class."""
    
    def test_initialization(self, time_domain):
        """Test SRSF framework initialization."""
        srsf = SquareRootSlopeFramework(time_domain)
        np.testing.assert_array_equal(srsf.time, time_domain)
    
    def test_to_srsf_basic(self, srsf_framework, time_domain):
        """Test basic SRSF computation."""
        t = time_domain
        f = np.sin(2 * np.pi * t)
        
        q = srsf_framework.to_srsf(f)
        
        assert q.shape == f.shape
        assert np.all(np.isfinite(q))
    
    def test_to_srsf_constant_function(self, srsf_framework, time_domain):
        """Test SRSF of constant function."""
        f = np.ones_like(time_domain)
        
        q = srsf_framework.to_srsf(f)
        
        # SRSF of constant function should be zero
        np.testing.assert_allclose(q, 0, atol=1e-10)
    
    def test_to_srsf_linear_function(self, srsf_framework, time_domain):
        """Test SRSF of linear function."""
        f = time_domain  # Linear function
        
        q = srsf_framework.to_srsf(f)
        
        # SRSF of linear function should be constant
        assert np.all(np.isfinite(q))
        # Check that it's approximately constant (allowing for numerical errors)
        assert np.std(q) < 0.1
    
    def test_from_srsf_reconstruction(self, srsf_framework, time_domain):
        """Test function reconstruction from SRSF."""
        t = time_domain
        f_original = np.sin(2 * np.pi * t)
        
        q = srsf_framework.to_srsf(f_original)
        f_reconstructed = srsf_framework.from_srsf(q, f0=f_original[0])
        
        # Should reconstruct original function up to a constant
        np.testing.assert_allclose(f_reconstructed, f_original, rtol=1e-2)
    
    def test_from_srsf_with_different_f0(self, srsf_framework, time_domain):
        """Test reconstruction with different initial values."""
        t = time_domain
        f = np.sin(2 * np.pi * t)
        
        q = srsf_framework.to_srsf(f)
        f_rec1 = srsf_framework.from_srsf(q, f0=0.0)
        f_rec2 = srsf_framework.from_srsf(q, f0=1.0)
        
        # Difference should be approximately constant
        diff = f_rec2 - f_rec1
        np.testing.assert_allclose(diff, diff[0], rtol=1e-10)
    
    def test_warp_f_gamma_identity(self, srsf_framework, time_domain):
        """Test warping with identity function."""
        f = np.sin(2 * np.pi * time_domain)
        gam = time_domain  # Identity warping
        
        f_warped = srsf_framework.warp_f_gamma(f, gam)
        
        np.testing.assert_allclose(f_warped, f, rtol=1e-10)
    
    def test_warp_f_gamma_basic(self, srsf_framework, time_domain, simple_warping):
        """Test basic function warping."""
        f = np.sin(2 * np.pi * time_domain)
        gam = simple_warping
        
        f_warped = srsf_framework.warp_f_gamma(f, gam)
        
        assert f_warped.shape == f.shape
        assert np.all(np.isfinite(f_warped))
    
    def test_warp_q_gamma_identity(self, srsf_framework, time_domain):
        """Test SRSF warping with identity function."""
        f = np.sin(2 * np.pi * time_domain)
        q = srsf_framework.to_srsf(f)
        gam = time_domain  # Identity warping
        
        q_warped = srsf_framework.warp_q_gamma(q, gam)
        
        np.testing.assert_allclose(q_warped, q, rtol=1e-10)
    
    def test_warp_q_gamma_basic(self, srsf_framework, time_domain, simple_warping):
        """Test basic SRSF warping."""
        f = np.sin(2 * np.pi * time_domain)
        q = srsf_framework.to_srsf(f)
        gam = simple_warping

        q_warped = srsf_framework.warp_q_gamma(q, gam)

        assert q_warped.shape == q.shape
        assert np.all(np.isfinite(q_warped))

    def test_get_gamma_fallback(self, srsf_framework, time_domain):
        """Test get_gamma with fallback when optimum_reparamN2 is not available."""
        f1 = np.sin(2 * np.pi * time_domain)
        f2 = np.sin(2 * np.pi * time_domain + np.pi/4)
        q1 = srsf_framework.to_srsf(f1)
        q2 = srsf_framework.to_srsf(f2)

        # This should work even without optimum_reparamN2
        # We'll catch warnings but not require them since the dependency might be available
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this test
            gamma = srsf_framework.get_gamma(q1, q2)

        assert gamma.shape == time_domain.shape
        assert np.all(np.isfinite(gamma))


class TestWarpingManifold:
    """Test WarpingManifold class."""
    
    def test_initialization(self, time_domain):
        """Test warping manifold initialization."""
        wm = WarpingManifold(time_domain)
        np.testing.assert_array_equal(wm.time, time_domain)
    
    def test_inner_product_basic(self, warping_manifold, time_domain):
        """Test basic inner product computation."""
        v1 = np.sin(2 * np.pi * time_domain)
        v2 = np.cos(2 * np.pi * time_domain)
        
        ip = warping_manifold.inner_product(v1, v2)
        
        assert isinstance(ip, (float, np.floating))
        assert np.isfinite(ip)
    
    def test_inner_product_self(self, warping_manifold, time_domain):
        """Test inner product of vector with itself."""
        v = np.sin(2 * np.pi * time_domain)
        
        ip = warping_manifold.inner_product(v, v)
        
        assert ip >= 0  # Should be non-negative
    
    def test_norm_positive(self, warping_manifold, time_domain):
        """Test that norm is positive for non-zero vectors."""
        v = np.sin(2 * np.pi * time_domain)
        
        norm = warping_manifold.norm(v)
        
        assert norm >= 0
        assert np.isfinite(norm)
    
    def test_norm_zero_vector(self, warping_manifold, time_domain):
        """Test norm of zero vector."""
        v = np.zeros_like(time_domain)
        
        norm = warping_manifold.norm(v)
        
        np.testing.assert_allclose(norm, 0, atol=1e-10)
    
    def test_inverse_identity(self, warping_manifold, time_domain):
        """Test inverse of identity warping."""
        gam = time_domain
        
        gam_inv = warping_manifold.inverse(gam)
        
        np.testing.assert_allclose(gam_inv, gam, rtol=1e-10)
    
    def test_inverse_basic(self, warping_manifold, simple_warping):
        """Test basic warping inverse."""
        gam = simple_warping
        
        gam_inv = warping_manifold.inverse(gam)
        
        assert gam_inv.shape == gam.shape
        assert np.all(np.isfinite(gam_inv))
        # Check that it's monotonic
        assert np.all(np.diff(gam_inv) >= 0)
    
    def test_log_exp_consistency(self, warping_manifold, time_domain):
        """Test that log and exp are inverse operations."""
        # Create normalized vectors for the manifold
        v1 = np.sin(2 * np.pi * time_domain)
        v1 = v1 / warping_manifold.norm(v1)
        
        v2 = np.cos(2 * np.pi * time_domain)
        v2 = v2 / warping_manifold.norm(v2)
        
        # Test log-exp consistency
        log_result, _ = warping_manifold.log(v1, v2)  # theta not used in this test
        exp_result = warping_manifold.exp(v2, log_result)
        
        # Due to numerical precision, we use a relaxed tolerance
        np.testing.assert_allclose(exp_result, v1, rtol=1e-1)
    
    def test_center_single_warping(self, warping_manifold, time_domain):
        """Test center computation with single warping function."""
        gam = time_domain + 0.1 * np.sin(2 * np.pi * time_domain) * time_domain * (1 - time_domain)
        
        center = warping_manifold.center(gam)
        
        assert center.shape == gam.shape
        assert np.all(np.isfinite(center))


class TestEdgeCasesGeometry:
    """Test edge cases for geometry functions."""
    
    def test_srsf_very_small_function(self, srsf_framework, time_domain):
        """Test SRSF with very small function values."""
        f = 1e-10 * np.sin(2 * np.pi * time_domain)
        
        q = srsf_framework.to_srsf(f)
        
        assert np.all(np.isfinite(q))
    
    def test_srsf_large_function(self, srsf_framework, time_domain):
        """Test SRSF with large function values."""
        f = 1e6 * np.sin(2 * np.pi * time_domain)
        
        q = srsf_framework.to_srsf(f)
        
        assert np.all(np.isfinite(q))
    
    def test_warping_boundary_values(self, srsf_framework, time_domain):
        """Test warping with boundary-preserving gamma."""
        f = np.sin(2 * np.pi * time_domain)
        # Create a warping that preserves boundaries
        gam = time_domain**2
        gam = (gam - gam[0]) / (gam[-1] - gam[0])  # Normalize to [0,1]
        
        f_warped = srsf_framework.warp_f_gamma(f, gam)
        
        assert np.all(np.isfinite(f_warped))
        # Check boundary preservation
        np.testing.assert_allclose(f_warped[0], f[0], rtol=1e-10)
        np.testing.assert_allclose(f_warped[-1], f[-1], rtol=1e-10)
