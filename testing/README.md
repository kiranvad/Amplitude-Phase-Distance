# Testing Directory for apdist Package

This directory contains all testing-related files for the `apdist` package.

## Quick Start

### 1. Basic Verification (No Dependencies)
```bash
python test_basic_functionality.py
```

### 2. Install Testing Dependencies
```bash
pip install -r requirements-test.txt
```

### 3. Run Full Test Suite
```bash
python run_tests.py
```

## Directory Structure

```
testing/
├── README.md                    # This file
├── TESTING_GUIDE.md            # Comprehensive testing guide
├── TEST_SUITE_SUMMARY.md       # Summary of test suite and fixes
├── pytest.ini                  # Pytest configuration
├── requirements-test.txt        # Testing dependencies
├── run_tests.py                # Test runner script
├── test_basic_functionality.py # Quick verification script
└── tests/                      # Test files
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures and configuration
    ├── test_basic.py           # Basic functionality tests
    ├── test_distances.py       # Distance computation tests
    ├── test_geometry.py        # SRSF and manifold tests
    ├── test_integration.py     # End-to-end integration tests
    ├── test_torch.py           # PyTorch implementation tests
    └── test_utils.py           # Utility function tests
```

## Test Categories

- **Basic Tests**: Core functionality without external dependencies
- **Fast Tests**: Quick unit tests (default)
- **Slow Tests**: Performance and stability tests
- **Integration Tests**: End-to-end workflows
- **PyTorch Tests**: Tests requiring PyTorch and funcshape

## Running Tests

### Quick Commands
```bash
# Basic verification
python test_basic_functionality.py

# All tests
python run_tests.py

# Fast tests only
python run_tests.py --fast

# With coverage
python run_tests.py --coverage

# Integration tests
python run_tests.py --integration

# PyTorch tests
python run_tests.py --torch
```

### Using pytest directly
```bash
pytest tests/                    # All tests
pytest -v tests/                 # Verbose
pytest -m "not slow" tests/      # Exclude slow tests
pytest --cov=apdist tests/       # With coverage
```

## Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive guide for running and writing tests
- **[TEST_SUITE_SUMMARY.md](TEST_SUITE_SUMMARY.md)** - Summary of test suite structure and fixes
- **[tests/README.md](tests/README.md)** - Detailed test documentation

## Dependencies

### Required for Basic Tests
- numpy
- scipy
- matplotlib

### Required for Full Test Suite
- pytest
- pytest-cov

### Optional for Complete Coverage
- torch (for PyTorch tests)
- funcshape package (for PyTorch tests)
- warping package (for full optimization functionality)

## Getting Help

1. Check the [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed instructions
2. Run `python test_basic_functionality.py` for quick verification
3. Review test output for specific error messages
4. Ensure all dependencies are installed

## Contributing

When adding new tests:
1. Place them in the appropriate test file in `tests/`
2. Use fixtures from `conftest.py`
3. Add appropriate markers (`@pytest.mark.slow`, etc.)
4. Update documentation if needed

The test suite is designed to be robust, comprehensive, and easy to use!
