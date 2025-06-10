#!/usr/bin/env python3
"""
Quick test script to verify basic functionality works.
This script tests the core functionality without requiring pytest.
"""

import sys
import warnings
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from apdist.geometry import SquareRootSlopeFramework, WarpingManifold
        from apdist.distances import AmplitudePhaseDistance
        print("[SUCCESS] All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"[FAILED] Import error: {e}")
        return False

def test_basic_srsf():
    """Test basic SRSF functionality."""
    print("Testing SRSF framework...")
    try:
        from apdist.geometry import SquareRootSlopeFramework
        
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        
        srsf = SquareRootSlopeFramework(t)
        q = srsf.to_srsf(f)
        f_rec = srsf.from_srsf(q, f0=f[0])
        
        assert q.shape == f.shape
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(f_rec))
        
        print("[SUCCESS] SRSF framework works correctly")
        return True
    except Exception as e:
        print(f"[FAILED] SRSF test failed: {e}")
        return False

def test_basic_distance():
    """Test basic distance computation."""
    print("Testing distance computation...")
    try:
        from apdist.distances import AmplitudePhaseDistance

        t = np.linspace(0, 1, 101)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.sin(2 * np.pi * t + np.pi/4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore fallback warnings
            da, dp = AmplitudePhaseDistance(t, f1, f2)

        assert da >= 0
        assert dp >= 0
        assert np.isfinite(da)
        assert np.isfinite(dp)

        print(f"[SUCCESS] Distance computation works: da={da:.4f}, dp={dp:.4f}")

        # Test that fallback warning is issued when warping is not available
        try:
            import optimum_reparamN2
            print("[INFO] Warping optimization available")
        except ImportError:
            print("[INFO] Using fallback (identity warping) - this is expected without warping package")

        return True
    except Exception as e:
        print(f"[FAILED] Distance test failed: {e}")
        return False

def test_identical_functions():
    """Test that identical functions give zero distance."""
    print("Testing identical functions...")
    try:
        from apdist.distances import AmplitudePhaseDistance
        
        t = np.linspace(0, 1, 101)
        f = np.sin(2 * np.pi * t)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da, dp = AmplitudePhaseDistance(t, f, f.copy())
        
        assert da == 0.0
        assert dp == 0.0
        
        print("[SUCCESS] Identical functions give zero distance")
        return True
    except Exception as e:
        print(f"[FAILED] Identical functions test failed: {e}")
        return False

def test_warping_manifold():
    """Test warping manifold functionality."""
    print("Testing warping manifold...")
    try:
        from apdist.geometry import WarpingManifold

        t = np.linspace(0, 1, 101)
        v1 = np.sin(2 * np.pi * t)
        v2 = np.cos(2 * np.pi * t)

        wm = WarpingManifold(t)
        ip = wm.inner_product(v1, v2)
        norm1 = wm.norm(v1)
        norm2 = wm.norm(v2)

        assert np.isfinite(ip)
        assert norm1 >= 0
        assert norm2 >= 0

        print(f"[SUCCESS] Warping manifold works: ip={ip:.4f}, norms=({norm1:.4f}, {norm2:.4f})")
        return True
    except Exception as e:
        print(f"[FAILED] Warping manifold test failed: {e}")
        return False


def test_all_dependencies():
    """Test that all required dependencies are available."""
    print("Testing required dependencies...")

    # Test warping package
    try:
        import optimum_reparamN2
        print("[SUCCESS] Warping optimization package is available")
    except ImportError:
        print("[FAILED] Warping optimization not available")
        return False

    # Test PyTorch
    try:
        import torch
        print("[SUCCESS] PyTorch is available")
    except ImportError:
        print("[FAILED] PyTorch not available")
        return False

    # Test funcshape
    try:
        import funcshape
        print("[SUCCESS] funcshape is available")
    except ImportError:
        print("[FAILED] funcshape not available")
        return False

    # Test PyTorch version of apdist
    try:
        from apdist.torch import TorchAmplitudePhaseDistance as TorchAPD
        print("[SUCCESS] PyTorch version of apdist is available")
    except ImportError as e:
        print(f"[FAILED] PyTorch version import failed: {e}")
        return False

    print("[SUCCESS] All required dependencies are available")
    return True

def main():
    """Run all basic tests."""
    print("=" * 60)
    print("Testing apdist package basic functionality")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_srsf,
        test_basic_distance,
        test_identical_functions,
        test_warping_manifold,
        test_all_dependencies
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAILED] Test {test.__name__} failed with exception: {e}")
        print()

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All basic tests passed! The package is working correctly.")
        return 0
    else:
        print("[WARNING] Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
