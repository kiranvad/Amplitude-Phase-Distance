"""Tests for PyTorch implementation in apdist.torch module."""

import torch
from funcshape.functions import Function

from apdist.torch import *

optim_kwargs = {
    "n_iters": 1,
    "n_basis": 5,
    "n_layers": 5,
    "n_restarts": 1,
    "domain_type": "linear",
    "basis_type": "palais",
    "lr": 1e-1,
    "n_domain": 101,
    "eps": 1e-2,
    "verbose": True,
}


class TestTorchAmplitudeDistance:
    """Test PyTorch amplitude distance computation."""

    def test_identical_functions_zero_distance(self):
        """Test that identical functions have zero amplitude distance."""
        t = torch.linspace(0, 1, 101)
        f_vals = torch.sin(2 * torch.pi * t)

        f1 = Function(t, f_vals.reshape(-1, 1))
        f2 = Function(t, f_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping

        dist = torch_amplitude_distance(f1, f2, warping)

        assert torch.equal(dist, torch.tensor(0.0))

    def test_different_functions_positive_distance(self):
        """Test that different functions have positive amplitude distance."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi / 4)

        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping

        dist = torch_amplitude_distance(f1, f2, warping)

        assert dist > 0
        assert torch.isfinite(dist)


class TestTorchPhaseDistance:
    """Test PyTorch phase distance computation."""

    def test_identical_functions_zero_distance(self):
        """Test that identical functions have zero phase distance."""
        t = torch.linspace(0, 1, 101)
        f_vals = torch.sin(2 * torch.pi * t)

        f1 = Function(t, f_vals.reshape(-1, 1))
        f2 = Function(t, f_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))  # Identity warping

        dist = torch_phase_distance(f1, f2, warping)

        assert torch.equal(dist, torch.tensor(0.0))

    def test_different_functions_bounded_distance(self):
        """Test that phase distance is bounded."""
        t = torch.linspace(0, 1, 101)
        f1_vals = torch.sin(2 * torch.pi * t)
        f2_vals = torch.sin(2 * torch.pi * t + torch.pi / 4)

        f1 = Function(t, f1_vals.reshape(-1, 1))
        f2 = Function(t, f2_vals.reshape(-1, 1))
        warping = Function(t, t.reshape(-1, 1))

        dist = torch_phase_distance(f1, f2, warping)

        assert dist >= 0
        assert dist <= torch.pi
        assert torch.isfinite(dist)


class TestTorchAmplitudePhaseDistance:
    """Test the main PyTorch AmplitudePhaseDistance function."""

    def test_identical_functions_zero_distances(self):
        """Test that identical functions return zero for both distances."""
        t = torch.linspace(0, 1, 101)
        f = torch.sin(2 * torch.pi * t)

        da, dp, output = TorchAmplitudePhaseDistance(t, f, f.clone(), **optim_kwargs)

        assert da == 0.0
        assert dp == 0.0
        assert len(output) > 0  # Should return optimization output

    def test_different_functions_positive_distances(self):
        """Test that different functions return positive distances."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi / 4)

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert da >= 0
        assert dp >= 0
        assert torch.isfinite(da)
        assert torch.isfinite(dp)

    def test_return_types(self):
        """Test that function returns correct types."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi / 4)

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert isinstance(da, torch.Tensor)
        assert isinstance(dp, torch.Tensor)
        assert isinstance(output, (list, tuple))


class TestTorchEdgeCases:
    """Test edge cases for PyTorch implementation."""

    def test_constant_functions(self):
        """Test with constant functions."""
        t = torch.linspace(0, 1, 101)
        f1 = torch.ones_like(t)
        f2 = torch.ones_like(t) * 2

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert torch.isfinite(da)
        assert torch.isfinite(dp)


class TestTorchDeviceHandling:
    """Ensure the device parameter controls the output tensor location."""

    def test_device_argument(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.linspace(0, 1, 101)
        f1 = torch.sin(2 * torch.pi * t)
        f2 = torch.sin(2 * torch.pi * t + torch.pi / 4)

        da, dp, _ = TorchAmplitudePhaseDistance(t, f1, f2, device=device, **optim_kwargs)

        assert da.device == device
        assert dp.device == device

    def test_linear_functions(self):
        """Test with linear functions."""
        t = torch.linspace(0, 1, 101)
        f1 = t
        f2 = 2 * t + 1

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert torch.isfinite(da)
        assert torch.isfinite(dp)

    def test_very_small_functions(self):
        """Test with very small function values."""
        t = torch.linspace(0, 1, 101)
        f1 = 1e-10 * torch.sin(2 * torch.pi * t)
        f2 = 1e-10 * torch.sin(2 * torch.pi * t + torch.pi / 4)

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert torch.isfinite(da)
        assert torch.isfinite(dp)

    def test_large_functions(self):
        """Test with large function values."""
        t = torch.linspace(0, 1, 101)
        f1 = 1e6 * torch.sin(2 * torch.pi * t)
        f2 = 1e6 * torch.sin(2 * torch.pi * t + torch.pi / 4)

        da, dp, output = TorchAmplitudePhaseDistance(t, f1, f2, **optim_kwargs)

        assert torch.isfinite(da)
        assert torch.isfinite(dp)
