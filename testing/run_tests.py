#!/usr/bin/env python3
"""
Test runner script for the apdist package.

This script provides a convenient way to run tests with different configurations.
"""

import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Please make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for apdist package")
    parser.add_argument("--fast", action="store_true", 
                       help="Run only fast tests (exclude slow and torch tests)")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--torch", action="store_true",
                       help="Run only PyTorch tests")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=apdist", "--cov-report=html", "--cov-report=term"])
    
    # Configure test selection
    if args.fast:
        cmd.extend(["-m", "not slow and not torch"])
        description = "Fast tests"
    elif args.integration:
        cmd.extend(["-m", "integration"])
        description = "Integration tests"
    elif args.torch:
        cmd.extend(["-m", "torch"])
        description = "PyTorch tests"
    else:
        description = "All tests"
    
    # Add test directory
    cmd.append("tests/")
    
    success = run_command(cmd, description)
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
