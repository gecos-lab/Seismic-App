"""
Autotracking Module for Petrel-like Seismic Interpretation

This module provides automatic horizon tracking capabilities similar to Petrel,
using seismic attributes and dynamic programming for guided tracking.
"""

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import cv2
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from enum import Enum
import warnings

from seismic_attributes import SeismicAttributes


class TrackingMode(Enum):
    """Different autotracking modes."""
    ATTRIBUTE_GUIDED = "attribute_guided"
    EDGE_BASED = "edge_based"
    PHASE_GUIDED = "phase_guided"
    SIMILARITY_GUIDED = "similarity_guided"
    HYBRID = "hybrid"


class Autotracker:
    """
    Automatic horizon tracking system inspired by Petrel's autotracking capabilities.

    Features:
    - Seismic attribute-guided tracking
    - Dynamic programming for optimal paths
    - Multi-scale processing
    - Quality metrics and confidence scores
    - Parallel processing for efficiency
    """

    def __init__(self, seismic_volume=None, attributes_processor=None,
                 use_gpu: bool = True, n_cores: int = None):
        """
        Initialize the autotracker.

        Args:
            seismic_volume: 3D seismic volume (optional, can be set later)
            attributes_processor: SeismicAttributes instance
            use_gpu: Whether to use GPU acceleration
            n_cores: Number of CPU cores to use (default: auto-detect)
        """
        self.seismic_volume = seismic_volume
        self.attributes_processor = attributes_processor or SeismicAttributes(use_gpu=use_gpu)

        # Processing parameters
        self.use_gpu = use_gpu
        self.n_cores = n_cores or min(mp.cpu_count(), 8)

        # Tracking parameters
        self.tracking_params = {
            'max_dip_angle': 30,  # Maximum allowed dip angle (degrees)
            'smoothness_weight': 0.5,  # Weight for path smoothness
            'attribute_weight': 0.8,  # Weight for attribute guidance
            'edge_weight': 0.3,  # Weight for edge strength
            'min_confidence': 0.6,  # Minimum confidence threshold
            'max_gap_fill': 10,  # Maximum gap to fill in tracking
            'search_window': 15,  # Search window size for dynamic programming
        }

        # Cache for computed attributes and paths
        self._attribute_cache = {}
        self._path_cache = {}

        print(f"Autotracker initialized - GPU: {self.use_gpu}, Cores: {self.n_cores}")

    def set_tracking_parameters(self, **params):
        """Update tracking parameters."""
        self.tracking_params.update(params)
        print(f"Updated tracking parameters: {params}")

    def auto_detect_seeds(self, slice_data: np.ndarray, slice_idx: int,
                         n_seeds: int = 10, method: str = 'from_points',
                         existing_points: List[Tuple[int, int]] = None,
                         existing_masks: Dict[int, np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Automatically detect seed points for horizon tracking using existing points and masks.

        Args:
            slice_data: 2D seismic slice
            slice_idx: Index of the slice
            n_seeds: Number of seed points to detect
            method: Detection method ('from_points', 'from_masks', 'edge', 'amplitude', 'phase', 'hybrid')
            existing_points: List of existing foreground points [(x,y), ...]
            existing_masks: Dictionary of existing masks {obj_id: mask_array}

        Returns:
            List of (x, y) seed point coordinates
        """
        print(f"Auto-detecting {n_seeds} seed points using {method} method")

        # Method 1: Use existing foreground points (recommended)
        if method == 'from_points' and existing_points:
            print(f"Using {len(existing_points)} existing foreground points as seeds")
            return existing_points[:n_seeds]

        # Method 2: Extract seeds from existing propagated masks
        elif method == 'from_masks' and existing_masks:
            seed_candidates = []
            for obj_id, mask in existing_masks.items():
                if mask is not None and np.any(mask):
                    # Extract multiple points along the horizon contour, not just center
                    y_coords, x_coords = np.where(mask)

                    if len(y_coords) > 0:
                        # Sort by x-coordinate to get left-to-right horizon points
                        sorted_indices = np.argsort(x_coords)
                        x_sorted = x_coords[sorted_indices]
                        y_sorted = y_coords[sorted_indices]

                        # Sample points along the horizon (every N pixels or fixed number)
                        n_samples = min(n_seeds, len(x_sorted))
                        if n_samples > 0:
                            step = max(1, len(x_sorted) // n_samples)
                            sampled_indices = np.arange(0, len(x_sorted), step)[:n_samples]

                            for idx in sampled_indices:
                                seed_candidates.append((int(x_sorted[idx]), int(y_sorted[idx])))

                            print(f"Extracted {len(sampled_indices)} seeds along horizon from Object {obj_id} mask")

            if seed_candidates:
                print(f"Total seeds extracted from existing masks: {len(seed_candidates)}")
                return seed_candidates[:n_seeds]

        # Method 3: Fallback to automatic detection using attributes
        print("Falling back to automatic attribute-based seed detection")

        # Compute relevant attributes
        attributes = self.attributes_processor.compute_slice_attributes(
            slice_data, ['amplitude', 'phase', 'edge_strength', 'similarity']
        )

        seed_candidates = []

        if method == 'edge':
            # Use edge strength for seed detection
            edges = attributes.get('edge_strength', np.zeros_like(slice_data))
            # Apply non-maximum suppression and thresholding
            threshold = np.percentile(edges, 85)  # Top 15% of edges
            edge_mask = edges > threshold

            # Find local maxima
            local_max = ndimage.maximum_filter(edges, size=5) == edges
            seed_mask = edge_mask & local_max

            # Extract coordinates
            y_coords, x_coords = np.where(seed_mask)
            seed_candidates = list(zip(x_coords, y_coords))

        elif method == 'amplitude':
            # Use amplitude peaks/troughs
            amplitude = attributes.get('amplitude', np.abs(slice_data))

            # Find local maxima and minima
            local_max = ndimage.maximum_filter(amplitude, size=7) == amplitude
            local_min = ndimage.minimum_filter(amplitude, size=7) == amplitude

            # Combine and threshold
            extrema = (local_max | local_min) & (amplitude > np.percentile(amplitude, 75))

            y_coords, x_coords = np.where(extrema)
            seed_candidates = list(zip(x_coords, y_coords))

        elif method == 'phase':
            # Use phase discontinuities
            phase = attributes.get('phase', np.zeros_like(slice_data))

            # Compute phase gradient magnitude
            phase_grad_y, phase_grad_x = np.gradient(phase)
            phase_grad_mag = np.sqrt(phase_grad_x**2 + phase_grad_y**2)

            # Threshold for phase discontinuities
            threshold = np.percentile(phase_grad_mag, 90)
            phase_mask = phase_grad_mag > threshold

            y_coords, x_coords = np.where(phase_mask)
            seed_candidates = list(zip(x_coords, y_coords))

        elif method == 'hybrid':
            # Combine multiple attributes
            amplitude = attributes.get('amplitude', np.abs(slice_data))
            edges = attributes.get('edge_strength', np.zeros_like(slice_data))
            similarity = attributes.get('similarity', np.zeros_like(slice_data))

            # Create composite score
            amp_norm = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
            edge_norm = (edges - np.min(edges)) / (np.max(edges) - np.min(edges) + 1e-10)
            sim_norm = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity) + 1e-10)

            composite_score = 0.4 * amp_norm + 0.4 * edge_norm + 0.2 * sim_norm

            # Find local maxima in composite score
            local_max = ndimage.maximum_filter(composite_score, size=5) == composite_score
            threshold = np.percentile(composite_score, 85)
            seed_mask = local_max & (composite_score > threshold)

            y_coords, x_coords = np.where(seed_mask)
            seed_candidates = list(zip(x_coords, y_coords))

        # Filter and select top candidates
        if len(seed_candidates) > n_seeds:
            # Score candidates by attribute strength
            scores = []
            for x, y in seed_candidates:
                if 0 <= y < slice_data.shape[0] and 0 <= x < slice_data.shape[1]:
                    score = 0
                    if 'amplitude' in attributes:
                        score += 0.4 * attributes['amplitude'][y, x]
                    if 'edge_strength' in attributes:
                        score += 0.4 * attributes['edge_strength'][y, x]
                    if 'similarity' in attributes:
                        score += 0.2 * attributes['similarity'][y, x]
                    scores.append(score)
                else:
                    scores.append(0)

            # Select top N candidates
            top_indices = np.argsort(scores)[-n_seeds:]
            seed_candidates = [seed_candidates[i] for i in top_indices]

        # Ensure coordinates are within bounds
        valid_seeds = []
        for x, y in seed_candidates:
            if 0 <= y < slice_data.shape[0] and 0 <= x < slice_data.shape[1]:
                valid_seeds.append((int(x), int(y)))

        print(f"Detected {len(valid_seeds)} valid seed points using {method}")
        return valid_seeds[:n_seeds]

    def track_horizon_refined(self, start_slice_idx: int, existing_mask: np.ndarray = None,
                             direction: str = 'forward', max_slices: int = 50,
                             mode: TrackingMode = TrackingMode.HYBRID) -> Dict:
        """
        Track horizon by refining an existing mask to better follow geological features.

        Args:
            start_slice_idx: Starting slice index
            existing_mask: Existing mask to refine (from propagation)
            direction: 'forward' or 'backward'
            max_slices: Maximum slices to track
            mode: Tracking mode

        Returns:
            Dictionary containing refined paths and confidence scores
        """
        print(f"Refining existing horizon mask using {mode.value} tracking")

        if self.seismic_volume is None:
            raise ValueError("Seismic volume not set")

        if existing_mask is None or not np.any(existing_mask):
            print("No existing mask provided, falling back to regular tracking")
            return self.track_horizon_dp(start_slice_idx, [], direction, max_slices, mode)

        # Extract seed points from the existing mask for this slice
        y_coords, x_coords = np.where(existing_mask)
        if len(y_coords) == 0:
            print("No points in existing mask, falling back to regular tracking")
            return self.track_horizon_dp(start_slice_idx, [], direction, max_slices, mode)

        # Sort points by x-coordinate and sample along the horizon
        sorted_indices = np.argsort(x_coords)
        x_sorted = x_coords[sorted_indices]
        y_sorted = y_coords[sorted_indices]

        # Sample points every few pixels along the horizon
        step = max(1, len(x_sorted) // 10)  # Sample ~10 points
        sampled_indices = np.arange(0, len(x_sorted), step)

        seed_points = [(int(x_sorted[idx]), int(y_sorted[idx])) for idx in sampled_indices]
        print(f"Extracted {len(seed_points)} seed points from existing mask for refinement")

        # Use regular tracking but with enhanced parameters for refinement
        results = self.track_horizon_dp(start_slice_idx, seed_points, direction, max_slices, mode)

        # Mark this as a refined result
        results['refined'] = True
        results['original_mask'] = existing_mask

        return results

    def track_horizon_dp(self, start_slice_idx: int, start_points: List[Tuple[int, int]],
                        direction: str = 'forward', max_slices: int = 50,
                        mode: TrackingMode = TrackingMode.HYBRID) -> Dict:
        """
        Track horizon using dynamic programming for optimal path finding.

        Args:
            start_slice_idx: Starting slice index
            start_points: List of (x, y) starting points on the horizon
            direction: 'forward' or 'backward' tracking
            max_slices: Maximum number of slices to track
            mode: Tracking mode to use

        Returns:
            Dictionary containing tracked paths and confidence scores
        """
        print(f"Starting {mode.value} horizon tracking from slice {start_slice_idx} in {direction} direction")

        if self.seismic_volume is None:
            raise ValueError("Seismic volume not set")

        # Initialize tracking results
        tracked_paths = {}
        confidence_scores = {}

        direction_multiplier = 1 if direction == 'forward' else -1
        current_slice_idx = start_slice_idx

        # Track from each starting point
        for point_idx, start_point in enumerate(start_points):
            path = [start_point]  # Store as (x, y) tuples
            confidences = [1.0]  # Start with perfect confidence
            path_key = f"path_{point_idx}"

            print(f"Tracking path {point_idx} from point {start_point}")

            try:
                for step in range(max_slices):
                    current_slice_idx = start_slice_idx + (step + 1) * direction_multiplier

                    # Check bounds
                    if not (0 <= current_slice_idx < self.seismic_volume.shape[0]):
                        break

                    # Get current slice
                    if direction_multiplier == 1:
                        slice_data = self.seismic_volume[current_slice_idx, :, :]
                    else:
                        slice_data = self.seismic_volume[current_slice_idx, :, :]

                    # Find next point using dynamic programming
                    next_point, confidence = self._find_next_point_dp(
                        slice_data, path[-1], mode
                    )

                    if next_point is None or confidence < self.tracking_params['min_confidence']:
                        print(f"Low confidence ({confidence:.3f}) or no valid point found at slice {current_slice_idx}")
                        break

                    path.append(next_point)
                    confidences.append(confidence)

                # Store results
                tracked_paths[path_key] = path
                confidence_scores[path_key] = confidences

                print(f"Tracked {len(path)} points for path {point_idx}")

            except Exception as e:
                print(f"Error tracking path {point_idx}: {e}")
                continue

        return {
            'paths': tracked_paths,
            'confidences': confidence_scores,
            'start_slice': start_slice_idx,
            'direction': direction,
            'mode': mode.value
        }

    def _find_next_point_dp(self, slice_data: np.ndarray, current_point: Tuple[int, int],
                           mode: TrackingMode) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find the next point on horizon using dynamic programming.

        Args:
            slice_data: Current seismic slice
            current_point: Current (x, y) position
            mode: Tracking mode

        Returns:
            Tuple of (next_point, confidence_score)
        """
        x_curr, y_curr = current_point
        h, w = slice_data.shape

        # Define search window
        window_size = self.tracking_params['search_window']
        x_min = max(0, x_curr - window_size)
        x_max = min(w, x_curr + window_size + 1)
        y_min = max(0, y_curr - window_size)
        y_max = min(h, y_curr + window_size + 1)

        # Compute cost function based on tracking mode
        cost_matrix = self._compute_tracking_cost(slice_data, x_min, x_max, y_min, y_max, mode)

        # Apply dip constraints
        cost_matrix = self._apply_dip_constraints(cost_matrix, current_point,
                                                x_min, x_max, y_min, y_max)

        # Find optimal path using dynamic programming
        best_point, confidence = self._optimize_path_dp(cost_matrix, x_min, y_min)

        if best_point is None:
            return None, 0.0

        # Convert back to global coordinates
        global_point = (x_min + best_point[0], y_min + best_point[1])

        return global_point, confidence

    def _compute_tracking_cost(self, slice_data: np.ndarray, x_min: int, x_max: int,
                              y_min: int, y_max: int, mode: TrackingMode) -> np.ndarray:
        """
        Compute cost matrix for tracking based on the selected mode.
        """
        window_data = slice_data[y_min:y_max, x_min:x_max]

        # Compute relevant attributes for the window
        attributes = self.attributes_processor.compute_slice_attributes(
            window_data, ['amplitude', 'phase', 'edge_strength', 'similarity', 'frequency']
        )

        # Initialize cost matrix
        cost_matrix = np.zeros(window_data.shape)

        if mode == TrackingMode.ATTRIBUTE_GUIDED:
            # Use amplitude as primary guide
            amplitude = attributes.get('amplitude', np.abs(window_data))
            # Higher amplitude = lower cost (more likely to be horizon)
            amp_norm = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
            cost_matrix += (1 - amp_norm) * self.tracking_params['attribute_weight']

        elif mode == TrackingMode.EDGE_BASED:
            # Use edge strength
            edges = attributes.get('edge_strength', np.zeros_like(window_data))
            edge_norm = (edges - np.min(edges)) / (np.max(edges) - np.min(edges) + 1e-10)
            cost_matrix += (1 - edge_norm) * self.tracking_params['edge_weight']

        elif mode == TrackingMode.PHASE_GUIDED:
            # Use phase discontinuities
            phase = attributes.get('phase', np.zeros_like(window_data))
            phase_grad_y, phase_grad_x = np.gradient(phase)
            phase_grad_mag = np.sqrt(phase_grad_x**2 + phase_grad_y**2)
            phase_norm = (phase_grad_mag - np.min(phase_grad_mag)) / (np.max(phase_grad_mag) - np.min(phase_grad_mag) + 1e-10)
            cost_matrix += (1 - phase_norm) * self.tracking_params['attribute_weight']

        elif mode == TrackingMode.SIMILARITY_GUIDED:
            # Use trace similarity
            similarity = attributes.get('similarity', np.zeros_like(window_data))
            sim_norm = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity) + 1e-10)
            cost_matrix += (1 - sim_norm) * self.tracking_params['attribute_weight']

        elif mode == TrackingMode.HYBRID:
            # Combine multiple attributes
            amplitude = attributes.get('amplitude', np.abs(window_data))
            edges = attributes.get('edge_strength', np.zeros_like(window_data))
            similarity = attributes.get('similarity', np.zeros_like(window_data))

            # Normalize each attribute
            amp_norm = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
            edge_norm = (edges - np.min(edges)) / (np.max(edges) - np.min(edges) + 1e-10)
            sim_norm = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity) + 1e-10)

            # Weighted combination
            cost_matrix += (1 - amp_norm) * 0.4 * self.tracking_params['attribute_weight']
            cost_matrix += (1 - edge_norm) * 0.3 * self.tracking_params['edge_weight']
            cost_matrix += (1 - sim_norm) * 0.3 * self.tracking_params['attribute_weight']

        # Add smoothness term (distance from center - prefer straight paths)
        y_center, x_center = cost_matrix.shape[0] // 2, cost_matrix.shape[1] // 2
        y_coords, x_coords = np.ogrid[:cost_matrix.shape[0], :cost_matrix.shape[1]]
        distance_from_center = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
        max_distance = np.sqrt(y_center**2 + x_center**2)
        smoothness_cost = distance_from_center / max_distance if max_distance > 0 else 0
        cost_matrix += smoothness_cost * self.tracking_params['smoothness_weight']

        return cost_matrix

    def _apply_dip_constraints(self, cost_matrix: np.ndarray, current_point: Tuple[int, int],
                             x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """
        Apply dip angle constraints to the cost matrix.
        """
        max_dip_rad = np.radians(self.tracking_params['max_dip_angle'])

        # Compute expected dip direction (rough estimate)
        y_center, x_center = cost_matrix.shape[0] // 2, cost_matrix.shape[1] // 2

        # Create dip constraint mask
        y_coords, x_coords = np.ogrid[:cost_matrix.shape[0], :cost_matrix.shape[1]]
        dy = y_coords - y_center
        dx = x_coords - x_center

        # Compute dip angles
        dip_angles = np.arctan2(dy, dx)
        dip_magnitudes = np.sqrt(dy**2 + dx**2)

        # Allow points within dip constraints
        dip_mask = np.abs(dip_angles) <= max_dip_rad

        # Apply constraint (increase cost for points outside dip limits)
        cost_matrix[~dip_mask] += 2.0  # Significant penalty

        return cost_matrix

    def _optimize_path_dp(self, cost_matrix: np.ndarray, x_offset: int, y_offset: int) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Use dynamic programming to find optimal next point.
        """
        # For simplicity, find minimum cost point (can be extended to full DP path optimization)
        min_cost_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        min_cost = cost_matrix[min_cost_idx]

        # Convert cost to confidence score (lower cost = higher confidence)
        max_cost = np.max(cost_matrix)
        min_cost_val = np.min(cost_matrix)
        if max_cost > min_cost_val:
            confidence = 1.0 - (min_cost - min_cost_val) / (max_cost - min_cost_val)
        else:
            confidence = 1.0

        return min_cost_idx, confidence

    def track_multiple_horizons(self, start_slice_idx: int, n_horizons: int = 3,
                              direction: str = 'forward', max_slices: int = 50) -> Dict:
        """
        Track multiple horizons simultaneously, avoiding conflicts.
        """
        print(f"Tracking {n_horizons} horizons from slice {start_slice_idx}")

        # Auto-detect seed points
        start_slice_data = self.seismic_volume[start_slice_idx, :, :]
        seed_points = self.auto_detect_seeds(start_slice_data, start_slice_idx,
                                           n_seeds=n_horizons * 2)  # Extra candidates

        # Track each horizon
        all_results = []
        used_points = set()

        for i in range(n_horizons):
            # Filter available seed points
            available_seeds = [pt for pt in seed_points if pt not in used_points]

            if not available_seeds:
                break

            # Select best remaining seed
            best_seed = self._select_best_seed(available_seeds, start_slice_data, used_points)

            if best_seed is None:
                break

            used_points.add(best_seed)

            # Track horizon from this seed
            result = self.track_horizon_dp(
                start_slice_idx, [best_seed], direction, max_slices,
                TrackingMode.HYBRID
            )

            all_results.append(result)

        return {
            'horizons': all_results,
            'n_tracked': len(all_results)
        }

    def _select_best_seed(self, candidates: List[Tuple[int, int]], slice_data: np.ndarray,
                         used_points: set) -> Optional[Tuple[int, int]]:
        """Select the best seed point from candidates."""
        if not candidates:
            return None

        # Score candidates by attribute strength and separation from used points
        best_score = -np.inf
        best_candidate = None

        attributes = self.attributes_processor.compute_slice_attributes(
            slice_data, ['amplitude', 'edge_strength']
        )

        for candidate in candidates:
            x, y = candidate

            # Attribute score
            amp_score = attributes.get('amplitude', np.zeros_like(slice_data))[y, x]
            edge_score = attributes.get('edge_strength', np.zeros_like(slice_data))[y, x]
            attr_score = 0.6 * amp_score + 0.4 * edge_score

            # Separation score (prefer points far from used ones)
            min_distance = np.inf
            for used_point in used_points:
                dist = np.sqrt((x - used_point[0])**2 + (y - used_point[1])**2)
                min_distance = min(min_distance, dist)

            sep_score = min(min_distance / 50.0, 1.0)  # Normalize to 0-1

            total_score = 0.7 * attr_score + 0.3 * sep_score

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        return best_candidate

    def refine_tracking_with_sam(self, tracking_results: Dict, predictor) -> Dict:
        """
        Refine autotracking results using SAM (SAM2 or SAM3) for better accuracy.

        Args:
            tracking_results: Results from autotracking
            predictor: SeismicPredictor or SeismicPredictorSAM3 instance

        Returns:
            Refined tracking results
        """
        print("Refining autotracking with SAM...")

        refined_paths = {}

        for path_key, path in tracking_results['paths'].items():
            refined_path = []

            # Use every nth point as seed for SAM refinement
            seed_interval = max(1, len(path) // 10)  # Use 10 seed points max

            for i in range(0, len(path), seed_interval):
                point = path[i]
                slice_idx = tracking_results['start_slice'] + i * (1 if tracking_results['direction'] == 'forward' else -1)

                if 0 <= slice_idx < self.seismic_volume.shape[0]:
                    slice_data = self.seismic_volume[slice_idx, :, :]

                    # Use SAM to refine the point
                    try:
                        # Set context for predictor
                        # We use a dummy slice type since we are just predicting on this specific slice data
                        predictor.set_current_slice("autotrack", slice_idx, slice_data)
                        
                        masks, scores, _ = predictor.predict_masks_from_points(
                            points=[point],
                            point_labels=[1],  # Foreground
                            multimask_output=True
                        )

                        # Use highest scoring mask
                        best_mask = masks[np.argmax(scores)]

                        # Find refined point (center of mass of mask)
                        y_coords, x_coords = np.where(best_mask)
                        if len(y_coords) > 0:
                            refined_point = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                            refined_path.append(refined_point)
                        else:
                            refined_path.append(point)

                    except Exception as e:
                        print(f"SAM refinement failed for point {point}: {e}")
                        refined_path.append(point)

            refined_paths[path_key] = refined_path

        tracking_results['refined_paths'] = refined_paths
        return tracking_results


    def compute_tracking_quality(self, tracking_results: Dict) -> Dict[str, float]:
        """
        Compute quality metrics for tracking results.

        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'mean_confidence': 0.0,
            'path_smoothness': 0.0,
            'attribute_consistency': 0.0,
            'overall_quality': 0.0
        }

        if not tracking_results.get('confidences'):
            return metrics

        # Mean confidence
        all_confidences = []
        for confidences in tracking_results['confidences'].values():
            all_confidences.extend(confidences)

        if all_confidences:
            metrics['mean_confidence'] = np.mean(all_confidences)

        # Path smoothness (lower is smoother)
        smoothness_scores = []
        for path in tracking_results['paths'].values():
            if len(path) > 2:
                # Compute second derivative (curvature)
                points = np.array(path)
                dx = np.diff(points[:, 0])
                dy = np.diff(points[:, 1])
                ddx = np.diff(dx)
                ddy = np.diff(dy)
                curvature = np.sqrt(ddx**2 + ddy**2)
                smoothness_scores.append(np.mean(curvature))

        if smoothness_scores:
            metrics['path_smoothness'] = 1.0 / (1.0 + np.mean(smoothness_scores))  # Normalize to 0-1

        # Overall quality (weighted combination)
        metrics['overall_quality'] = (
            0.5 * metrics['mean_confidence'] +
            0.3 * metrics['path_smoothness'] +
            0.2 * metrics['attribute_consistency']
        )

        return metrics
