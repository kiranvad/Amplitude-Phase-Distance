# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 200+ tests
- GitHub Actions workflows for CI/CD
- Code quality checks (Black, isort, flake8, mypy)
- Performance benchmarking
- Automated dependency updates
- Documentation generation
- Pre-commit hooks for code quality
- Contributing guidelines and issue templates

### Changed
- Reorganized test files into dedicated `testing/` directory
- Improved error handling for missing dependencies
- Enhanced package structure and organization

### Fixed
- Graceful fallback when `optimum_reparamN2` package is not available
- Unused parameter warnings in geometry module
- Import error handling for optional dependencies

## [1.0.0] - 2024-01-XX

### Added
- Initial release of apdist package
- Core functionality for amplitude-phase distance computation
- Square Root Slope Framework (SRSF) implementation
- Warping manifold operations
- PyTorch implementation support
- Visualization utilities
- Example notebook demonstrating usage

### Features
- `AmplitudePhaseDistance` function for computing distances between functions
- `SquareRootSlopeFramework` class for SRSF operations
- `WarpingManifold` class for manifold operations
- Support for both NumPy and PyTorch backends
- Comprehensive documentation and examples

### Dependencies
- numpy >= 1.18.1
- scipy
- matplotlib
- Cython
- cffi

### Optional Dependencies
- torch (for PyTorch implementation)
- funcshape (for advanced PyTorch features)
- optimum_reparamN2 (for warping optimization)

---

## Template for Future Releases

## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Vulnerability fixes
