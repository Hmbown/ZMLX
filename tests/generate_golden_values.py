"""Generate golden values for cross-backend testing.

This script generates reference outputs from various operations on the
Metal backend, which can then be used to validate other backends.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

def generate_elementwise_golden():
    """Generate golden values for elementwise operations."""
    results = {}
    
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    y = mx.array([0.5, 1.5, 2.5, 3.5])
    
    results["add"] = (x + y).tolist()
    results["multiply"] = (x * y).tolist()
    results["relu"] = mx.maximum(x, 0).tolist()
    results["sigmoid"] = (1 / (1 + mx.exp(-x))).tolist()
    
    return results


def generate_reduction_golden():
    """Generate golden values for reduction operations."""
    results = {}
    
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    results["sum_all"] = mx.sum(x).tolist()
    results["sum_axis0"] = mx.sum(x, axis=0).tolist()
    results["sum_axis1"] = mx.sum(x, axis=1).tolist()
    results["mean_all"] = mx.mean(x).tolist()
    results["max_all"] = mx.max(x).tolist()
    
    return results


def generate_softmax_golden():
    """Generate golden values for softmax operations."""
    results = {}
    
    x = mx.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    
    results["softmax_axis_1"] = mx.softmax(x, axis=-1).tolist()
    results["softmax_axis_0"] = mx.softmax(x, axis=0).tolist()
    
    return results


def main():
    """Generate and save golden values."""
    backend = "metal" if mx.metal.is_available() else "cpu"
    
    golden_values = {
        "backend": backend,
        "version": "0.1.0",
        "tests": {
            "elementwise": generate_elementwise_golden(),
            "reduction": generate_reduction_golden(),
            "softmax": generate_softmax_golden(),
        }
    }
    
    # Create tests directory if it doesn't exist
    tests_dir = Path(__file__).parent
    output_path = tests_dir / f"golden_values_{backend}.json"
    
    with open(output_path, "w") as f:
        json.dump(golden_values, f, indent=2)
    
    print(f"Generated golden values for {backend} backend at {output_path}")


if __name__ == "__main__":
    main()
