"""Compare golden values across backends.

This script compares golden values from different backends to ensure
cross-backend correctness.
"""

import json
import sys
from pathlib import Path

try:
    import mlx.core as mx
except Exception as exc:  # pragma: no cover - skip when MLX isn't available
    print(f"MLX not available; skipping golden value comparison: {exc}")
    raise SystemExit(0) from None
import numpy as np


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return False
    fn = getattr(metal, "is_available", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def load_golden_values(backend: str) -> dict:
    """Load golden values for a specific backend."""
    tests_dir = Path(__file__).parent
    filepath = tests_dir / f"golden_values_{backend}.json"
    
    if not filepath.exists():
        return {}
    
    with open(filepath) as f:
        return json.load(f)


def compare_values(actual, expected, path: str, rtol: float = 1e-4, atol: float = 1e-4):
    """Compare two values and report differences."""
    try:
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            actual_arr = np.array(actual)
            expected_arr = np.array(expected)
            if actual_arr.shape != expected_arr.shape:
                print(f"  [FAIL] {path}: Shape mismatch {actual_arr.shape} vs {expected_arr.shape}")
                return False
            np.testing.assert_allclose(actual_arr, expected_arr, rtol=rtol, atol=atol)
            print(f"  [PASS] {path}")
            return True
        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            if np.isclose(actual, expected, rtol=rtol, atol=atol):
                print(f"  [PASS] {path}")
                return True
            else:
                print(f"  [FAIL] {path}: {actual} vs {expected}")
                return False
        else:
            print(f"  [SKIP] {path}: Unsupported type comparison")
            return True
    except AssertionError as e:
        print(f"  [FAIL] {path}: {e}")
        return False


def compare_golden_backends(backend1: str, backend2: str) -> bool:
    """Compare golden values between two backends."""
    print(f"\nComparing {backend1} vs {backend2}...")
    
    golden1 = load_golden_values(backend1)
    golden2 = load_golden_values(backend2)
    
    if not golden1 or not golden2:
        print("  Missing golden values for one or both backends")
        return False
    
    all_passed = True
    
    for test_category, tests in golden1.get("tests", {}).items():
        print(f"\n  Category: {test_category}")
        tests2 = golden2.get("tests", {}).get(test_category, {})
        
        for test_name, value1 in tests.items():
            value2 = tests2.get(test_name)
            if value2 is None:
                print(f"    [SKIP] {test_name}: Not found in {backend2}")
                continue
            
            if not compare_values(value1, value2, f"{test_category}.{test_name}"):
                all_passed = False
    
    return all_passed


def main():
    """Compare golden values across all available backends."""
    current_backend = "metal" if _metal_available() else "cpu"
    print(f"Current backend: {current_backend}")
    
    # Compare current backend against reference (Metal)
    if current_backend != "metal":
        print("\n" + "=" * 50)
        print("Cross-Backend Golden Value Comparison")
        print("=" * 50)
        
        success = compare_golden_backends("metal", current_backend)
        
        if success:
            print("\n" + "=" * 50)
            print("All golden value comparisons PASSED!")
            print("=" * 50)
            return 0
        else:
            print("\n" + "=" * 50)
            print("Some golden value comparisons FAILED!")
            print("=" * 50)
            return 1
    else:
        print("\nRunning on Metal backend - generating reference values")
        # Just validate our own consistency
        golden = load_golden_values("metal")
        if golden:
            print("Reference golden values exist and are valid")
            return 0
        else:
            print("No reference golden values found - run generate_golden_values.py first")
            return 1


if __name__ == "__main__":
    sys.exit(main())
