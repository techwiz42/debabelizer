#!/usr/bin/env python3
"""
Test runner for Debabelizer

Provides convenient test running with different modes and configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result"""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
        
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Debabelizer tests")
    parser.add_argument(
        "--mode", 
        choices=["unit", "integration", "all", "fast", "coverage"],
        default="unit",
        help="Test mode to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--pattern", "-k",
        type=str,
        help="Run tests matching pattern"
    )
    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Run tests with specific markers"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel > 1:
        base_cmd.extend(["-n", str(args.parallel)])
    
    if args.pattern:
        base_cmd.extend(["-k", args.pattern])
    
    if args.markers:
        base_cmd.extend(["-m", args.markers])
    
    # Configure based on mode
    if args.mode == "unit":
        print("🎯 Running Unit Tests Only")
        cmd = base_cmd + ["-m", "not integration and not slow", "tests/"]
        
    elif args.mode == "integration":
        print("🌐 Running Integration Tests")
        print("⚠️  Note: Integration tests require API keys to be set as environment variables")
        cmd = base_cmd + ["-m", "integration", "tests/"]
        
    elif args.mode == "fast":
        print("⚡ Running Fast Tests Only")
        cmd = base_cmd + ["-m", "not slow and not integration", "tests/"]
        
    elif args.mode == "coverage":
        print("📊 Running Tests with Coverage Report")
        cmd = base_cmd + [
            "--cov=src/debabelizer",
            "--cov-report=html",
            "--cov-report=term",
            "--cov-fail-under=80",
            "tests/"
        ]
        
    elif args.mode == "all":
        print("🚀 Running All Tests")
        cmd = base_cmd + ["tests/"]
        
    # Change to project directory
    project_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        import os
        os.chdir(project_dir)
        
        # Check if pytest is available
        try:
            import pytest
        except ImportError:
            print("❌ pytest not found. Please install test dependencies:")
            print("   pip install -e .[test]")
            sys.exit(1)
        
        # Run the tests  
        import shlex
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        success = run_command(cmd_str, f"Running tests in {args.mode} mode")
        
        if args.mode == "coverage" and success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        # Print summary
        print("\n" + "="*60)
        if success:
            print("🎉 All tests completed successfully!")
        else:
            print("💥 Some tests failed. Check output above for details.")
            
        return 0 if success else 1
        
    finally:
        os.chdir(original_dir)


def print_test_info():
    """Print information about available test categories"""
    print("📋 Available Test Categories:")
    print("=" * 40)
    print("🔧 Unit Tests:")
    print("   - test_config.py: Configuration management")
    print("   - test_base_providers.py: Provider interfaces")
    print("   - test_voice_processor.py: Main processor logic")
    print("   - test_stt_providers.py: STT provider contracts")
    print("   - test_tts_providers.py: TTS provider contracts")
    print("   - test_utils.py: Utility functions")
    print()
    print("🌐 Integration Tests:")
    print("   - test_integration.py: End-to-end functionality")
    print("   - Requires API keys for real providers")
    print("   - Tests actual provider functionality")
    print()
    print("🏷️  Test Markers:")
    print("   - unit: Fast unit tests (default)")
    print("   - integration: Tests requiring API keys")
    print("   - slow: Long-running tests")
    print("   - network: Tests requiring network access")
    print()
    print("📊 Eight Test Languages (from Thanotopolis):")
    print("   - English (en), Spanish (es), German (de), Russian (ru)")
    print("   - Chinese (zh), Tagalog (tl), Korean (ko), Vietnamese (vi)")


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        print_test_info()
        print()
    
    sys.exit(main())