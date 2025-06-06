# Testing Guide for apdist Package

## Quick Start

### 1. Navigate to Testing Directory
```bash
cd testing
```

### 2. Verify Basic Functionality
```bash
python test_basic_functionality.py
```
This runs a quick verification that doesn't require pytest or external dependencies.

### 3. Install Testing Dependencies
```bash
pip install -r requirements-test.txt
```

### 4. Run Full Test Suite
```bash
python run_tests.py
```

## Test Categories

### Fast Tests (Default)
Tests that run quickly and don't require optional dependencies:
```bash
python run_tests.py --fast
# or
pytest -m "not slow and not torch" tests/
```

### Slow Tests
Longer-running tests for performance and stability:
```bash
pytest -m slow tests/
```

### PyTorch Tests
Tests requiring PyTorch and funcshape dependencies:
```bash
python run_tests.py --torch
# or
pytest -m torch tests/
```

### 🔗 Integration Tests
End-to-end workflow tests:
```bash
python run_tests.py --integration
# or
pytest -m integration tests/
```

## Coverage Reports

Generate HTML coverage report:
```bash
python run_tests.py --coverage
```

View coverage in browser:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Files Overview

| File | Purpose | Dependencies |
|------|---------|--------------|
| `test_basic.py` | Core functionality without external deps | numpy, scipy |
| `test_distances.py` | Distance computation tests | numpy, scipy |
| `test_geometry.py` | SRSF and manifold tests | numpy, scipy |
| `test_utils.py` | Visualization utilities | matplotlib |
| `test_torch.py` | PyTorch implementation | torch, funcshape |
| `test_integration.py` | End-to-end workflows | All above |

## Common Issues and Solutions

### Issue: "optimum_reparamN2 not available" warning
**Solution**: This is expected. The package falls back to identity warping. To get full functionality:
```bash
pip install git+https://github.com/kiranvad/warping.git
```

### Issue: PyTorch tests skipped
**Solution**: Install PyTorch dependencies:
```bash
pip install torch
pip install git+https://github.com/kiranvad/funcshape.git
```

### Issue: Plotting tests fail in headless environment
**Solution**: This is expected in environments without display. Tests will skip automatically.

### Issue: "ModuleNotFoundError" for pytest
**Solution**: Install pytest:
```bash
pip install pytest
```

## Writing New Tests

### 1. Choose the Right File
- Core functionality → `test_basic.py`
- Distance functions → `test_distances.py`
- SRSF/Manifold → `test_geometry.py`
- Plotting → `test_utils.py`
- PyTorch → `test_torch.py`
- Workflows → `test_integration.py`

### 2. Use Appropriate Markers
```python
@pytest.mark.slow          # For long-running tests
@pytest.mark.torch         # For PyTorch-dependent tests
@pytest.mark.integration   # For end-to-end tests
```

## Continuous Integration

For CI/CD pipelines, use:

```yaml
# Basic tests (fast, no optional deps)
- name: Run basic tests
  run: cd testing && python run_tests.py --fast

# Full tests with coverage
- name: Run full test suite
  run: cd testing && python run_tests.py --coverage
```

## Performance Testing

For performance regression testing:

```bash
# Run slow tests that include performance checks
pytest -m slow tests/test_integration.py::TestPerformanceAndStability
```

## Debugging Failed Tests

### 1. Run with verbose output
```bash
pytest -v tests/test_specific.py::TestClass::test_method
```

### 2. Run single test
```bash
pytest tests/test_distances.py::TestAmplitudeDistance::test_identical_functions_zero_distance
```

### 3. Drop into debugger on failure
```bash
pytest --pdb tests/
```

### 4. Show local variables on failure
```bash
pytest -l tests/
```

## Test Data and Fixtures

Common test fixtures available in all tests:

- `time_domain`: Standard [0,1] domain with 101 points
- `simple_functions`: Basic sine wave test functions
- `identical_functions`: Same function for zero-distance testing
- `polynomial_functions`: Polynomial test functions
- `srsf_framework`: Pre-configured SRSF instance
- `warping_manifold`: Pre-configured manifold instance
- `identity_warping`: Identity warping function
- `simple_warping`: Non-identity warping function



