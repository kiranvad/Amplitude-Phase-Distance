#!/usr/bin/env python3
"""
Development setup script for apdist package.
This script helps set up the development environment quickly.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False


def setup_development_environment():
    """Set up the development environment."""
    print("🚀 Setting up apdist development environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    # Install testing dependencies
    if not run_command("pip install -r testing/requirements-test.txt", "Installing testing dependencies"):
        return False
    
    # Install development dependencies
    dev_deps = [
        "black", "isort", "flake8", "mypy", "pre-commit",
        "sphinx", "sphinx-rtd-theme", "sphinx-autodoc-typehints"
    ]
    
    for dep in dev_deps:
        run_command(f"pip install {dep}", f"Installing {dep}", check=False)
    
    # Set up pre-commit hooks
    if run_command("pre-commit --version", "Checking pre-commit availability", check=False):
        run_command("pre-commit install", "Setting up pre-commit hooks", check=False)
    
    # Run basic tests to verify setup
    print("\n🧪 Running basic tests to verify setup...")
    os.chdir("testing")
    
    if run_command("python test_basic_functionality.py", "Running basic functionality test"):
        print("\n🎉 Development environment setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run tests: cd testing && python run_tests.py")
        print("2. Format code: black apdist/ testing/tests/")
        print("3. Check linting: flake8 apdist/")
        print("4. Run pre-commit: pre-commit run --all-files")
        print("5. Start developing! 🚀")
        return True
    else:
        print("\n⚠️  Setup completed but basic tests failed. Please check the installation.")
        return False


def install_optional_dependencies():
    """Install optional dependencies."""
    print("\n🔧 Installing optional dependencies...")
    
    # PyTorch
    torch_cmd = "pip install torch"
    if run_command(torch_cmd, "Installing PyTorch", check=False):
        print("✅ PyTorch installed successfully")
    else:
        print("⚠️  PyTorch installation failed. You can install it manually later.")
    
    # Funcshape (if available)
    funcshape_cmd = "pip install git+https://github.com/kiranvad/funcshape.git"
    if run_command(funcshape_cmd, "Installing funcshape", check=False):
        print("✅ Funcshape installed successfully")
    else:
        print("⚠️  Funcshape installation failed. This is optional for PyTorch features.")
    
    # Warping package (if available)
    warping_cmd = "pip install git+https://github.com/kiranvad/warping.git"
    if run_command(warping_cmd, "Installing warping package", check=False):
        print("✅ Warping package installed successfully")
    else:
        print("⚠️  Warping package installation failed. This is optional for optimization.")


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up apdist development environment")
    parser.add_argument("--with-optional", action="store_true", 
                       help="Install optional dependencies (PyTorch, funcshape, warping)")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup without optional tools")
    
    args = parser.parse_args()
    
    # Basic setup
    success = setup_development_environment()
    
    if not success:
        sys.exit(1)
    
    # Optional dependencies
    if args.with_optional:
        install_optional_dependencies()
    
    if not args.minimal:
        print("\n💡 Tip: Run with --with-optional to install PyTorch and other optional dependencies")
        print("💡 Tip: Run with --minimal for basic setup only")


if __name__ == "__main__":
    main()
