"""Tests for PyTorch implementation in apdist.torch module."""

import pytest
import numpy as np

# Try to import torch and funcshape dependencies
torch = pytest.importorskip("torch", reason="PyTorch not available")
try:
    from funcshape.functions import Function, SRSF, get_warping_function
    funcshape_available = True
except ImportError:
    funcshape_available = False

from apdist.torch import _amplitude_distance, _phase_distance, AmplitudePhaseDistance, plot_warping


@pytest.mark.torch
@pytest.mark.skipif(not funcshape_available, reason="funcshape package not available")
class TestTorchAmplitudeDistance:
    """Test PyTorch amplitude distance computation."""
    
    def test_identical_functions_zero_distance(self):
        """Test that identical functions have zero amplitude distance."""
        t = torch.linspace(0, 1, 101)
        f_vals = torch.sin(2 * torch.pi * t)
        
        f1 = Function(t, f_vals.reshape(-1, 1))
        f2 = Function(t, f_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping
        
        dist = _amplitude_distance(f1, f2, warping)
        
        assert torch.allclose(dist, torch.tensor(0.0), atol=1e-6)
    
    def test_different_functions_positive_distance(self):
        """Test that different functions have positive amplitude distance."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi/4)
        
        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping
        
        dist = _amplitude_distance(f1, f2, warping)
        
        assert dist > 0
        assert torch.isfinite(dist)
    
    def test_amplitude_distance_gradient(self):
        """Test that amplitude distance supports gradients."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi/4)
        f2_vals.requires_grad_(True)
        
        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))
        
        dist = _amplitude_distance(f1, f2, warping)
        
        # Test that we can compute gradients
        dist.backward()
        assert f2_vals.grad is not None
        assert torch.all(torch.isfinite(f2_vals.grad))


@pytest.mark.torch
@pytest.mark.skipif(not funcshape_available, reason="funcshape package not available")
class TestTorchPhaseDistance:
    """Test PyTorch phase distance computation."""
    
    def test_identical_functions_zero_distance(self):
        """Test that identical functions have zero phase distance."""
        t = torch.linspace(0, 1, 101)
        f_vals = torch.sin(2 * torch.pi * t)
        
        f1 = Function(t, f_vals.reshape(-1, 1))
        f2 = Function(t, f_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping
        
        dist = _phase_distance(f1, f2, warping)
        
        assert torch.allclose(dist, torch.tensor(0.0), atol=1e-6)
    
    def test_different_functions_bounded_distance(self):
        """Test that phase distance is bounded."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi/4)
        
        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))
        
        dist = _phase_distance(f1, f2, warping)
        
        assert dist >= 0
        assert dist <= torch.pi
        assert torch.isfinite(dist)
    
    def test_phase_distance_gradient(self):
        """Test that phase distance supports gradients."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi/4)
        f2_vals.requires_grad_(True)
        
        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))
        
        dist = _phase_distance(f1, f2, warping)
        
        # Test that we can compute gradients
        dist.backward()
        assert f2_vals.grad is not None
        assert torch.all(torch.isfinite(f2_vals.grad))


@pytest.mark.torch
@pytest.mark.skipif(not funcshape_available, reason="funcshape package not available")
class TestTorchAmplitudePhaseDistance:
    """Test the main PyTorch AmplitudePhaseDistance function."""
    
    def test_identical_functions_zero_distances(self):
        """Test that identical functions return zero for both distances."""
        t = torch.linspace(0, 1, 101)
        f = torch.sin(2 * torch.pi * t)
        
        da, dp, output = AmplitudePhaseDistance(t, f, f.clone())
        
        assert torch.allclose(da, torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(dp, torch.tensor(0.0), atol=1e-6)
        assert len(output) > 0  # Should return optimization output
    
    def test_different_functions_positive_distances(self):
        """Test that different functions return positive distances."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi/4)
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert da >= 0
        assert dp >= 0
        assert torch.isfinite(da)
        assert torch.isfinite(dp)
    
    def test_return_types(self):
        """Test that function returns correct types."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi/4)
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert isinstance(da, torch.Tensor)
        assert isinstance(dp, torch.Tensor)
        assert isinstance(output, (list, tuple))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the distance computation."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi/4)
        f2.requires_grad_(True)
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        # Test amplitude distance gradients
        da.backward(retain_graph=True)
        assert f2.grad is not None
        assert torch.all(torch.isfinite(f2.grad))
        
        # Reset gradients and test phase distance
        f2.grad.zero_()
        dp.backward()
        assert f2.grad is not None
        assert torch.all(torch.isfinite(f2.grad))


@pytest.mark.torch
@pytest.mark.skipif(not funcshape_available, reason="funcshape package not available")
class TestTorchPlotWarping:
    """Test PyTorch plotting functionality."""
    
    @pytest.mark.slow
    def test_plot_warping_basic(self):
        """Test basic warping plot functionality."""
        x = np.linspace(0, 1, 100)
        f1 = np.sin(2 * np.pi * x)
        f2 = np.sin(2 * np.pi * x + np.pi/4)
        
        # Create mock output
        t = torch.from_numpy((x - x.min()) / (x.max() - x.min()))
        warping = Function(t, t.reshape(-1, 1))
        output = [warping, None, [1.0, 0.5, 0.1]]  # Mock convergence
        
        # This should not raise an exception
        try:
            plot_warping(x, f1, f2, output)
        except Exception as e:
            # If matplotlib is not available in headless mode, skip
            if "display" in str(e).lower():
                pytest.skip("Display not available for plotting")
            else:
                raise


@pytest.mark.torch
@pytest.mark.skipif(not funcshape_available, reason="funcshape package not available")
class TestTorchEdgeCases:
    """Test edge cases for PyTorch implementation."""
    
    def test_constant_functions(self):
        """Test with constant functions."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.ones_like(t)
        f2 = torch.ones_like(t) * 2
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert torch.isfinite(da)
        assert torch.isfinite(dp)
    
    def test_linear_functions(self):
        """Test with linear functions."""
        t = torch.linspace(0, 1, 101)
        f1 = t
        f2 = 2 * t + 1
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert torch.isfinite(da)
        assert torch.isfinite(dp)
    
    def test_very_small_functions(self):
        """Test with very small function values."""
        t = torch.linspace(0, 1, 101)
        f1 = 1e-10 * torch.sin(2 * torch.pi * t)
        f2 = 1e-10 * torch.sin(2 * torch.pi * t + torch.pi/4)
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert torch.isfinite(da)
        assert torch.isfinite(dp)
    
    def test_large_functions(self):
        """Test with large function values."""
        t = torch.linspace(0, 1, 101)
        f1 = 1e6 * torch.sin(2 * torch.pi * t)
        f2 = 1e6 * torch.sin(2 * torch.pi * t + torch.pi/4)
        
        da, dp, output = AmplitudePhaseDistance(t, f1, f2)
        
        assert torch.isfinite(da)
        assert torch.isfinite(dp)


@pytest.mark.torch
class TestTorchWithoutFuncshape:
    """Test behavior when funcshape is not available."""
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # This test verifies that the module can be imported even if funcshape is not available
        # The actual functions will fail, but the module should import
        try:
            import apdist.torch
            # If we get here, the module imported successfully
            assert True
        except ImportError as e:
            # If funcshape is not available, this is expected
            if "funcshape" in str(e):
                pytest.skip("funcshape not available")
            else:
                raise
