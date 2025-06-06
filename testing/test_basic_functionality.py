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
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
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
        
        print("✅ SRSF framework works correctly")
        return True
    except Exception as e:
        print(f"❌ SRSF test failed: {e}")
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
        
        print(f"✅ Distance computation works: da={da:.4f}, dp={dp:.4f}")
        return True
    except Exception as e:
        print(f"❌ Distance test failed: {e}")
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
        
        print("✅ Identical functions give zero distance")
        return True
    except Exception as e:
        print(f"❌ Identical functions test failed: {e}")
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
        
        print(f"✅ Warping manifold works: ip={ip:.4f}, norms=({norm1:.4f}, {norm2:.4f})")
        return True
    except Exception as e:
        print(f"❌ Warping manifold test failed: {e}")
        return False

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
        test_warping_manifold
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The package is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
