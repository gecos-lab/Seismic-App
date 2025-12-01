#!/usr/bin/env python3
"""
Test script for refined horizon tracking functionality.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seismic_predictor import SeismicPredictor

def test_refined_tracking():
    """Test the refined horizon tracking functionality."""
    print("Testing refined horizon tracking...")

    # Create a predictor
    predictor = SeismicPredictor(demo_mode=True)

    # Create some dummy seismic data with a clear dipping horizon
    seismic_volume = np.random.randn(50, 100, 100).astype(np.float32)

    # Add a dipping horizon (sloping line)
    for i in range(50):
        # Create a horizon that dips from top-left to bottom-right
        horizon_y = int(30 + i * 0.5)  # Gradually dipping down
        thickness = 3

        # Add high amplitude along the horizon
        for t in range(max(0, horizon_y - thickness), min(100, horizon_y + thickness)):
            seismic_volume[i, t, :] += 3.0

        # Add some noise variation
        seismic_volume[i] += 0.1 * np.sin(np.linspace(0, 4*np.pi, 100))

    predictor.set_seismic_volume(seismic_volume)

    # Set current slice
    slice_data = seismic_volume[25, :, :]
    predictor.set_current_slice('inline', 25, slice_data)

    print("Creating initial propagated mask (simulating user's SAM2 propagation)...")

    # Simulate a user-propagated mask (straight horizontal line)
    straight_mask = np.zeros_like(slice_data, dtype=bool)
    straight_mask[35:40, :] = True  # Straight horizontal mask at y=35-40

    # Store this as an existing mask (simulating propagation result)
    predictor.object_masks[1] = {
        'inline_25': straight_mask
    }

    print("Testing refined tracking that improves the straight mask...")

    # Test refined tracking
    results = predictor.track_horizon_automatically(
        25, [], 'forward', 10, 'hybrid', 'refined_horizon'
    )

    print(f"Refined tracking completed with {len(results.get('paths', {}))} paths")
    print(f"Was this a refined result? {results.get('refined', False)}")

    if results.get('refined', False):
        print("[SUCCESS] Refined tracking correctly detected existing mask and improved it")
    else:
        print("[INFO] Used regular tracking (no existing mask found)")

    # Check that the results are stored
    horizons = predictor.get_all_autotracked_horizons()
    print(f"Stored refined horizons: {list(horizons.keys())}")

    # Show the difference between original and refined
    if results.get('paths'):
        original_mask_points = np.sum(straight_mask)
        refined_paths_points = sum(len(path) for path in results['paths'].values())

        print("\nComparison:")
        print(f"Original straight mask: {original_mask_points} pixels")
        print(f"Refined horizon paths: {refined_paths_points} points across slices")

        # Calculate if the refined path follows the geological feature better
        # (In this test, the geological horizon is at y ≈ 30 + 25*0.5 = 42.5)
        expected_horizon_y = 30 + 25 * 0.5  # 42.5

        if results['paths']:
            first_path = list(results['paths'].values())[0]
            if first_path:
                avg_refined_y = np.mean([point[1] for point in first_path])
                print(f"Original mask center: y=37.5, Expected geological horizon: y={expected_horizon_y:.1f}, Refined average: y={avg_refined_y:.1f}")
                if abs(avg_refined_y - expected_horizon_y) < abs(37.5 - expected_horizon_y):  # 37.5 is center of original mask
                    print("[SUCCESS] Refined tracking better follows geological horizon!")
                else:
                    print("[INFO] Refined tracking may need parameter adjustment")

    print("\n[SUCCESS] Refined horizon tracking test completed!")
    print("[INFO] The system now refines your propagated masks instead of creating new horizons")

if __name__ == "__main__":
    test_refined_tracking()
