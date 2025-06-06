# Development Workflows and Automation

This document describes the GitHub workflows and development tools set up for the apdist package to make development easier and more efficient.

## 🚀 GitHub Actions Workflows

### 1. Continuous Integration (`.github/workflows/ci.yml`)

**Triggers:** Push/PR to main/develop branches

**Features:**
- ✅ Multi-platform testing (Ubuntu, Windows, macOS)
- ✅ Multi-Python version support (3.8, 3.9, 3.10, 3.11)
- ✅ Dependency caching for faster builds
- ✅ Basic functionality verification
- ✅ Fast test suite execution
- ✅ Full test suite with coverage (Ubuntu + Python 3.10)
- ✅ Codecov integration for coverage reporting

**Matrix Strategy:** Optimized to reduce CI time while maintaining coverage

### 2. Code Quality (`.github/workflows/code-quality.yml`)

**Triggers:** Push/PR to main/develop branches

**Features:**
- ✅ **Black** - Code formatting checks
- ✅ **isort** - Import sorting verification
- ✅ **flake8** - Linting and style checks
- ✅ **mypy** - Type checking (informational)
- ✅ **bandit** - Security vulnerability scanning
- ✅ Artifact upload for security reports

### 3. Documentation (`.github/workflows/docs.yml`)

**Triggers:** Push/PR to main branch

**Features:**
- ✅ Automatic Sphinx documentation generation
- ✅ API documentation from docstrings
- ✅ GitHub Pages deployment
- ✅ RTD theme with modern styling
- ✅ Documentation artifact upload

### 4. Release Management (`.github/workflows/release.yml`)

**Triggers:** Git tags matching `v*` pattern

**Features:**
- ✅ Pre-release testing
- ✅ Package building and validation
- ✅ Automatic GitHub release creation
- ✅ Release notes generation
- ✅ Distribution artifact upload
- ✅ Optional PyPI publishing (commented out)

### 5. Dependency Updates (`.github/workflows/dependency-update.yml`)

**Triggers:** Weekly schedule + manual dispatch

**Features:**
- ✅ Automated dependency updates
- ✅ Security audit with `safety`
- ✅ Automatic PR creation for updates
- ✅ Testing with updated dependencies
- ✅ Security report artifacts

### 6. Performance Benchmarks (`.github/workflows/performance.yml`)

**Triggers:** Push to main, PR to main, monthly schedule

**Features:**
- ✅ Performance regression detection
- ✅ Benchmark result tracking
- ✅ Alert on performance degradation
- ✅ Historical performance data
- ✅ Multiple test sizes (small, medium, large)

## 🛠️ Development Tools

### 1. Pre-commit Hooks (`.pre-commit-config.yaml`)

**Automatic Code Quality:**
- ✅ Black code formatting
- ✅ isort import sorting
- ✅ flake8 linting
- ✅ bandit security checks
- ✅ General file checks (trailing whitespace, etc.)
- ✅ Jupyter notebook cleaning

**Setup:**
```bash
pip install pre-commit
pre-commit install
```

### 2. Development Setup Script (`dev-setup.py`)

**One-command Environment Setup:**
```bash
python dev-setup.py                    # Basic setup
python dev-setup.py --with-optional    # Include PyTorch, etc.
python dev-setup.py --minimal          # Minimal setup only
```

**Features:**
- ✅ Python version compatibility check
- ✅ Package installation in development mode
- ✅ Testing dependencies installation
- ✅ Development tools installation
- ✅ Pre-commit hooks setup
- ✅ Basic functionality verification
- ✅ Optional dependencies installation

### 3. Modern Configuration (`pyproject.toml`)

**Centralized Configuration:**
- ✅ Project metadata and dependencies
- ✅ Build system configuration
- ✅ Tool configurations (Black, isort, pytest, mypy, coverage)
- ✅ Optional dependency groups
- ✅ Development and testing configurations

### 4. Issue and PR Templates

**GitHub Templates:**
- ✅ **Bug Report Template** - Structured bug reporting
- ✅ **Feature Request Template** - Feature proposal format
- ✅ **Pull Request Template** - Comprehensive PR checklist

## 📋 Development Workflow

### For Contributors

1. **Setup Development Environment:**
   ```bash
   git clone https://github.com/kiranvad/Amplitude-Phase-Distance.git
   cd Amplitude-Phase-Distance
   python dev-setup.py
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Develop with Quality Checks:**
   ```bash
   # Code formatting
   black apdist/ testing/tests/
   isort apdist/ testing/tests/
   
   # Run tests
   cd testing && python run_tests.py --fast
   
   # Pre-commit checks
   pre-commit run --all-files
   ```

4. **Submit Pull Request:**
   - Automated CI will run all checks
   - Code quality verification
   - Multi-platform testing
   - Documentation updates

### For Maintainers

1. **Automated Dependency Management:**
   - Weekly dependency update PRs
   - Security vulnerability alerts
   - Automated testing with updates

2. **Release Process:**
   ```bash
   git tag v1.x.x
   git push origin v1.x.x
   # GitHub Actions handles the rest
   ```

3. **Performance Monitoring:**
   - Monthly performance benchmarks
   - Regression detection
   - Historical performance tracking

## 🎯 Benefits

### For Development
- ✅ **Faster Setup** - One-command environment setup
- ✅ **Code Quality** - Automated formatting and linting
- ✅ **Early Error Detection** - Pre-commit hooks catch issues
- ✅ **Consistent Style** - Enforced code formatting

### For Collaboration
- ✅ **Clear Guidelines** - Contributing documentation
- ✅ **Structured Issues** - Issue templates for better reporting
- ✅ **Review Process** - PR templates with checklists
- ✅ **Automated Testing** - CI ensures quality

### For Maintenance
- ✅ **Automated Updates** - Dependency management
- ✅ **Security Monitoring** - Vulnerability scanning
- ✅ **Performance Tracking** - Regression detection
- ✅ **Documentation** - Auto-generated docs

### For Users
- ✅ **Reliable Releases** - Automated testing before release
- ✅ **Up-to-date Documentation** - Auto-generated and deployed
- ✅ **Security** - Regular security audits
- ✅ **Performance** - Monitored and optimized

## 🚀 Getting Started

### For New Contributors
1. Read [`CONTRIBUTING.md`](CONTRIBUTING.md)
2. Run `python dev-setup.py`
3. Check out the testing guide in [`testing/TESTING_GUIDE.md`](testing/TESTING_GUIDE.md)
4. Start with a small issue or improvement

### For Maintainers
1. Enable GitHub Actions in repository settings
2. Set up branch protection rules
3. Configure Codecov integration
4. Enable GitHub Pages for documentation
5. Set up notification preferences

## 📊 Monitoring and Metrics

- **Test Coverage:** Tracked via Codecov
- **Code Quality:** Monitored via GitHub Actions
- **Performance:** Benchmarked monthly
- **Security:** Scanned weekly
- **Dependencies:** Updated weekly

This comprehensive automation setup ensures high code quality, reliable releases, and smooth collaboration! 🎉
