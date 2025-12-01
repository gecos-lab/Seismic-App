#!/usr/bin/env python3
"""
Test script for autotracking functionality with fallback support.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seismic_predictor import SeismicPredictor

def test_autotracking():
    """Test the autotracking functionality."""
    print("Testing autotracking functionality...")

    # Create a predictor
    predictor = SeismicPredictor(demo_mode=True)

    # Create some dummy seismic data
    seismic_volume = np.random.randn(50, 100, 100).astype(np.float32)
    predictor.set_seismic_volume(seismic_volume)

    # Set current slice
    slice_data = seismic_volume[25, :, :]
    predictor.set_current_slice('inline', 25, slice_data)

    # Test seed detection from points
    print("\n1. Testing seed detection from points...")
    existing_points = [(50, 30), (70, 40), (30, 20)]  # Some dummy points

    # Simulate having points in the predictor
    predictor.object_annotations[1] = {
        'points': existing_points,
        'labels': [1, 1, 1]  # All foreground
    }

    seeds = predictor.auto_detect_horizon_seeds(slice_data, 25, 5, 'from_points')
    print(f"Detected seeds from points: {seeds}")

    # Test seed detection from masks
    print("\n2. Testing seed detection from masks...")
    # Create a dummy mask
    dummy_mask = np.zeros_like(slice_data, dtype=bool)
    dummy_mask[30:40, 45:55] = True  # A small rectangular mask

    predictor.object_masks[1] = {
        'inline_25': dummy_mask
    }

    seeds_from_masks = predictor.auto_detect_horizon_seeds(slice_data, 25, 5, 'from_masks')
    print(f"Detected seeds from masks: {seeds_from_masks}")

    # Test horizon tracking
    print("\n3. Testing horizon tracking...")
    if seeds:
        results = predictor.track_horizon_automatically(
            25, seeds[:2], 'forward', 10, 'hybrid', 'test_horizon'
        )
        print(f"Tracking results keys: {list(results.keys())}")
        if 'paths' in results:
            print(f"Number of paths: {len(results['paths'])}")

    # Test attribute computation
    print("\n4. Testing attribute computation...")
    attrs = predictor.compute_seismic_attributes(slice_data, ['amplitude'])
    print(f"Computed attributes: {list(attrs.keys())}")

    print("\nAutotracking test completed successfully!")

if __name__ == "__main__":
    test_autotracking()



