#!/usr/bin/env python3
"""
Test script for visualization changes and autotracking display.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seismic_predictor import SeismicPredictor

def test_visualization():
    """Test the visualization with gray colormap."""
    print("Testing visualization with gray colormap...")

    # Create a predictor
    predictor = SeismicPredictor(demo_mode=True)

    # Create some dummy seismic data with clear features
    seismic_volume = np.random.randn(50, 100, 100).astype(np.float32)

    # Add some horizontal features to make it more visible
    for i in range(50):
        # Add horizontal bands
        seismic_volume[i, 20:30, :] += 2.0
        seismic_volume[i, 50:60, :] += 1.5

        # Add some vertical noise variation
        seismic_volume[i] += 0.1 * np.sin(np.linspace(0, 4*np.pi, 100))

    predictor.set_seismic_volume(seismic_volume)

    # Set current slice
    slice_data = seismic_volume[25, :, :]
    predictor.set_current_slice('inline', 25, slice_data)

    print("Testing gray colormap display...")

    # Test the gray colormap by checking if it would display correctly
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original seismic colormap
    vmin, vmax = np.percentile(slice_data, [5, 95])
    ax1.imshow(slice_data, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('Original Seismic Colormap')
    ax1.set_xlabel('Trace Position')
    ax1.set_ylabel('Time/Depth')

    # New gray colormap
    ax2.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('New Gray Colormap')
    ax2.set_xlabel('Trace Position')
    ax2.set_ylabel('Time/Depth')

    plt.tight_layout()
    plt.savefig('colormap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("[OK] Colormap comparison saved as 'colormap_comparison.png'")
    print("[OK] Left: Original seismic colormap, Right: New gray colormap")

    # Test autotracking seed detection
    print("\nTesting autotracking seed detection...")

    # Simulate having points
    predictor.object_annotations[1] = {
        'points': [(25, 25), (50, 25), (75, 25)],  # Horizontal line seeds
        'labels': [1, 1, 1]
    }

    # Test seed detection
    seeds = predictor.auto_detect_horizon_seeds(slice_data, 25, 5, 'from_points')
    print(f"[OK] Detected seeds from points: {seeds}")

    # Test horizon tracking
    print("\nTesting horizon tracking...")
    results = predictor.track_horizon_automatically(
        25, seeds[:2], 'forward', 10, 'hybrid', 'test_horizon'
    )
    print(f"[OK] Tracking completed with {len(results.get('paths', {}))} paths")

    # Check that autotracked horizons are stored
    horizons = predictor.get_all_autotracked_horizons()
    print(f"[OK] Stored autotracked horizons: {list(horizons.keys())}")

    print("\n[SUCCESS] Visualization and autotracking tests completed successfully!")
    print("[INFO] Check 'colormap_comparison.png' to see the gray colormap difference")

if __name__ == "__main__":
    test_visualization()
