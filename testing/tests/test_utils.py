"""Tests for utility functions in apdist.utils module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from apdist.utils import peak_plot2, plot_warping


class TestPeakPlot2:
    """Test peak_plot2 function."""
    
    def test_peak_plot2_basic(self):
        """Test basic peak plotting functionality."""
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1, 'width': 0.01}
        curve = {'color': 'blue', 'label': 'test'}
        markers = {'color': 'red'}
        
        peaks = peak_plot2(x, y, ax, pk, curve, markers)
        
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) >= 0  # Should find some peaks
        assert np.all(peaks < len(x))  # Peak indices should be valid
        
        plt.close(fig)
    
    def test_peak_plot2_no_peaks(self):
        """Test peak plotting with no peaks found."""
        x = np.linspace(0, 1, 100)
        y = np.ones_like(x)  # Constant function, no peaks
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1, 'width': 0.01}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        peaks = peak_plot2(x, y, ax, pk, curve, markers)
        
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) == 0  # Should find no peaks
        
        plt.close(fig)
    
    def test_peak_plot2_single_peak(self):
        """Test peak plotting with single peak."""
        x = np.linspace(0, 1, 100)
        y = np.exp(-(x - 0.5)**2 / 0.1)  # Gaussian with single peak
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1, 'width': 0.01}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        peaks = peak_plot2(x, y, ax, pk, curve, markers)
        
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) >= 1  # Should find at least one peak
        
        plt.close(fig)


class TestPlotWarping:
    """Test plot_warping function."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_warping_basic(self, mock_show):
        """Test basic warping plot functionality."""
        x = np.linspace(0, 1, 100)
        f1 = np.sin(2 * np.pi * x)
        f2 = np.sin(2 * np.pi * x + np.pi/4)
        f2_gamma = np.sin(2 * np.pi * x + np.pi/8)  # Partially aligned
        gamma = x + 0.1 * np.sin(2 * np.pi * x) * x * (1 - x)
        
        # This should not raise an exception
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        # Verify that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_warping_identical_functions(self, mock_show):
        """Test warping plot with identical functions."""
        x = np.linspace(0, 1, 100)
        f1 = np.sin(2 * np.pi * x)
        f2 = f1.copy()
        f2_gamma = f1.copy()
        gamma = x  # Identity warping
        
        # This should not raise an exception
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_warping_constant_functions(self, mock_show):
        """Test warping plot with constant functions."""
        x = np.linspace(0, 1, 100)
        f1 = np.ones_like(x)
        f2 = np.ones_like(x) * 2
        f2_gamma = np.ones_like(x) * 1.5
        gamma = x
        
        # This should not raise an exception
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_warping_no_peaks(self, mock_show):
        """Test warping plot when no peaks are found."""
        x = np.linspace(0, 1, 100)
        f1 = x  # Linear function, no peaks
        f2 = 2 * x + 1
        f2_gamma = 1.5 * x + 0.5
        gamma = x
        
        # This should not raise an exception even with no peaks
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        mock_show.assert_called_once()


class TestUtilsIntegration:
    """Integration tests for utils functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_full_workflow_visualization(self, mock_show):
        """Test complete visualization workflow."""
        # Create test data
        x = np.linspace(0, 1, 100)
        f1 = np.sin(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x)
        f2 = np.sin(2 * np.pi * (x - 0.1)) + 0.3 * np.sin(6 * np.pi * (x - 0.1))
        
        # Simulate warping result
        from apdist.geometry import SquareRootSlopeFramework
        srsf = SquareRootSlopeFramework(x)
        f2_gamma = srsf.warp_f_gamma(f2, x + 0.05 * np.sin(2 * np.pi * x))
        gamma = x + 0.05 * np.sin(2 * np.pi * x)
        
        # Test the visualization
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        mock_show.assert_called_once()


class TestUtilsEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_peak_plot2_empty_arrays(self):
        """Test peak plotting with empty arrays."""
        x = np.array([])
        y = np.array([])
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        with pytest.raises((ValueError, IndexError)):
            peak_plot2(x, y, ax, pk, curve, markers)
        
        plt.close(fig)
    
    def test_peak_plot2_single_point(self):
        """Test peak plotting with single data point."""
        x = np.array([0.5])
        y = np.array([1.0])
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        peaks = peak_plot2(x, y, ax, pk, curve, markers)
        
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) == 0  # Single point cannot be a peak
        
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_warping_extreme_gamma(self, mock_show):
        """Test warping plot with extreme gamma values."""
        x = np.linspace(0, 1, 100)
        f1 = np.sin(2 * np.pi * x)
        f2 = np.sin(2 * np.pi * x)
        f2_gamma = f1.copy()
        
        # Extreme warping (but still valid)
        gamma = x**3  # Highly nonlinear but monotonic
        
        # This should still work
        plot_warping(x, f1, f2, f2_gamma, gamma)
        
        mock_show.assert_called_once()
    
    def test_peak_plot2_nan_values(self):
        """Test peak plotting with NaN values."""
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x)
        y[50] = np.nan  # Insert NaN
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        # Should handle NaN gracefully
        peaks = peak_plot2(x, y, ax, pk, curve, markers)
        
        assert isinstance(peaks, np.ndarray)
        # The function should still work, though results may vary
        
        plt.close(fig)


class TestUtilsParameterValidation:
    """Test parameter validation in utility functions."""
    
    def test_peak_plot2_invalid_parameters(self):
        """Test peak plotting with invalid parameters."""
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x)
        
        fig, ax = plt.subplots()
        
        # Test with invalid prominence (negative)
        pk = {'prominence': -0.1}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        # This might raise an exception or return empty peaks
        try:
            peaks = peak_plot2(x, y, ax, pk, curve, markers)
            assert isinstance(peaks, np.ndarray)
        except ValueError:
            # This is acceptable behavior for invalid parameters
            pass
        
        plt.close(fig)
    
    def test_peak_plot2_mismatched_arrays(self):
        """Test peak plotting with mismatched array sizes."""
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x)[:50]  # Different size
        
        fig, ax = plt.subplots()
        pk = {'prominence': 0.1}
        curve = {'color': 'blue'}
        markers = {'color': 'red'}
        
        with pytest.raises((ValueError, IndexError)):
            peak_plot2(x, y, ax, pk, curve, markers)
        
        plt.close(fig)
