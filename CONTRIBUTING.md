# Contributing to apdist

Thank you for your interest in contributing to the apdist package! This document provides guidelines and information for contributors.

## Getting Started

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/your-username/Amplitude-Phase-Distance.git
cd Amplitude-Phase-Distance
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install testing dependencies
pip install -r testing/requirements-test.txt
```

### 3. Verify Installation

```bash
cd testing
python test_basic_functionality.py
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Follow the existing code style and conventions
- Add docstrings to new functions and classes
- Include type hints where appropriate
- Keep functions focused and well-documented

### 3. Add Tests

- Add tests for new functionality in the appropriate test file:
  - Core functionality → `testing/tests/test_basic.py`
  - Distance functions → `testing/tests/test_distances.py`
  - SRSF/Manifold → `testing/tests/test_geometry.py`
  - Utilities → `testing/tests/test_utils.py`
  - PyTorch → `testing/tests/test_torch.py`
  - Integration → `testing/tests/test_integration.py`

### 4. Run Tests

```bash
cd testing

# Quick verification
python test_basic_functionality.py

# Fast tests
python run_tests.py --fast

# Full test suite
python run_tests.py

# With coverage
python run_tests.py --coverage
```

### 5. Code Quality Checks

```bash
# Install code quality tools
pip install black isort flake8

# Format code
black apdist/ testing/tests/

# Sort imports
isort apdist/ testing/tests/

# Check linting
flake8 apdist/
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "feat: add new functionality for X"
# or
git commit -m "fix: resolve issue with Y"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `perf:` for performance improvements

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Maximum line length: 127 characters
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes

### Docstring Format

Use NumPy-style docstrings:

```python
def example_function(param1: np.ndarray, param2: float) -> tuple:
    """Brief description of the function.
    
    Parameters
    ----------
    param1 : numpy.ndarray
        Description of param1
    param2 : float
        Description of param2
        
    Returns
    -------
    tuple
        Description of return value
        
    Examples
    --------
    >>> result = example_function(data, 0.5)
    >>> print(result)
    (1.0, 2.0)
    """
    pass
```

### Test Guidelines

- Write clear, descriptive test names
- Test both normal cases and edge cases
- Use appropriate fixtures from `conftest.py`
- Add markers for slow tests: `@pytest.mark.slow`
- Handle optional dependencies gracefully

```python
def test_descriptive_name(self, fixture_name):
    """Test description explaining what is being tested."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert np.all(np.isfinite(result))
```

## Types of Contributions

### Bug Fixes

- Check existing issues before creating a new one
- Provide a minimal reproducible example
- Include tests that verify the fix
- Update documentation if necessary

### New Features

- Discuss major features in an issue first
- Ensure the feature fits the package's scope
- Add comprehensive tests
- Update documentation and examples
- Consider performance implications

### Documentation

- Fix typos and improve clarity
- Add examples and use cases
- Update API documentation
- Improve installation and usage instructions

### Performance Improvements

- Benchmark before and after changes
- Ensure improvements don't break existing functionality
- Document performance gains
- Consider memory usage implications

## Testing Guidelines

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical functions
- **Edge Case Tests**: Test boundary conditions and error cases

### Test Markers

Use appropriate pytest markers:

```python
@pytest.mark.slow          # For long-running tests
@pytest.mark.integration   # For integration tests
@pytest.mark.torch         # For PyTorch-dependent tests
```

### Handling Dependencies

For optional dependencies:

```python
torch = pytest.importorskip("torch", reason="PyTorch not available")

@pytest.mark.skipif(not has_dependency, reason="dependency not available")
def test_with_optional_dependency():
    pass
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation (will be automated in CI)
cd docs
sphinx-build -b html source build/html
```

### Adding Examples

- Include practical examples in docstrings
- Add Jupyter notebooks for complex use cases
- Ensure examples are tested and work correctly

## Release Process

Releases are handled by maintainers:

1. Update version in `setup.py` and `apdist/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. GitHub Actions will automatically create a release

## Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join the community chat (if available)
- Contact maintainers for complex issues

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow GitHub's community guidelines

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Documentation credits

Thank you for contributing to apdist! 🎉
