# Test Suite for apdist Package

This directory contains a comprehensive test suite for the `apdist` package, covering all major functionality including distance computations, SRSF framework, warping manifolds, and utilities.

## Test Structure

### Core Test Files

- **`test_distances.py`** - Tests for amplitude and phase distance computations
- **`test_geometry.py`** - Tests for SRSF framework and warping manifold classes  
- **`test_utils.py`** - Tests for visualization and utility functions
- **`test_torch.py`** - Tests for PyTorch implementation (requires optional dependencies)
- **`test_integration.py`** - End-to-end integration tests

### Configuration Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`__init__.py`** - Makes tests directory a Python package

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install pytest pytest-cov matplotlib
```

For PyTorch tests, also install:
```bash
pip install torch
pip install git+https://github.com/kiranvad/funcshape.git
```

### Basic Usage

Run all tests:
```bash
pytest tests/
```

Run with verbose output:
```bash
pytest -v tests/
```

### Test Categories

The tests are organized with markers for different categories:

- **Fast tests** (default): Quick unit tests
- **Slow tests**: Longer-running tests (marked with `@pytest.mark.slow`)
- **Integration tests**: End-to-end workflow tests (marked with `@pytest.mark.integration`)
- **PyTorch tests**: Tests requiring PyTorch dependencies (marked with `@pytest.mark.torch`)

### Running Specific Test Categories

Run only fast tests (exclude slow and torch tests):
```bash
pytest -m "not slow and not torch" tests/
```

Run only integration tests:
```bash
pytest -m integration tests/
```

Run only PyTorch tests:
```bash
pytest -m torch tests/
```

### Coverage Reports

Generate coverage report:
```bash
pytest --cov=apdist --cov-report=html --cov-report=term tests/
```

This will create an HTML coverage report in `htmlcov/index.html`.

### Using the Test Runner Script

A convenience script is provided in the root directory:

```bash
# Run all tests
python run_tests.py

# Run only fast tests
python run_tests.py --fast

# Run with coverage
python run_tests.py --coverage

# Run integration tests only
python run_tests.py --integration

# Run PyTorch tests only
python run_tests.py --torch
```

## Test Coverage

The test suite covers:

### Distance Computations (`test_distances.py`)
- Amplitude distance calculation
- Phase distance calculation  
- Main `AmplitudePhaseDistance` function
- Edge cases (identical functions, constant functions, etc.)
- Parameter validation

### Geometry Framework (`test_geometry.py`)
- SRSF conversion (to_srsf, from_srsf)
- Function and SRSF warping
- Warping manifold operations
- Numerical stability tests

### Utilities (`test_utils.py`)
- Peak detection and plotting
- Warping visualization
- Error handling for plotting functions

### PyTorch Implementation (`test_torch.py`)
- PyTorch distance computations
- Gradient flow testing
- Compatibility with funcshape package
- Performance comparisons

### Integration Tests (`test_integration.py`)
- End-to-end workflows
- Consistency checks between components
- Performance and stability tests
- Real-world scenarios

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `time_domain`: Standard time domain [0,1] with 101 points
- `simple_functions`: Basic sine wave test functions
- `identical_functions`: Identical functions for zero-distance testing
- `polynomial_functions`: Polynomial test functions
- `srsf_framework`: Pre-configured SRSF framework instance
- `warping_manifold`: Pre-configured warping manifold instance
- Various warping functions for testing

## Expected Test Results

When all dependencies are available, you should expect:

- **Core tests**: All should pass
- **PyTorch tests**: May be skipped if torch/funcshape not installed
- **Slow tests**: May take longer but should pass
- **Integration tests**: Should pass with proper numerical tolerances

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages as listed above
2. **Import errors**: Ensure the package is properly installed (`pip install -e .`)
3. **Numerical precision**: Some tests use relaxed tolerances for numerical stability
4. **Display issues**: Plotting tests may fail in headless environments

### Skipped Tests

Tests may be skipped for various reasons:
- Missing optional dependencies (PyTorch, funcshape)
- Platform-specific issues
- Display not available for plotting tests

This is normal and expected behavior.

## Contributing

When adding new functionality:

1. Add corresponding tests in the appropriate test file
2. Use appropriate markers (`@pytest.mark.slow`, `@pytest.mark.torch`, etc.)
3. Add fixtures to `conftest.py` if they'll be reused
4. Update this README if adding new test categories
5. Ensure tests pass with `python run_tests.py`
