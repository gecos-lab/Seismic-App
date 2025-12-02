"""
Seismic Attributes Module for Petrel-like Autotracking

This module provides seismic attribute computation for guided horizon tracking,
similar to attributes used in Petrel for automatic interpretation.
"""

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy import fft
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import Dict, List, Tuple, Optional, Union
import time
import psutil


class SeismicAttributes:
    """
    Computes seismic attributes for guided autotracking similar to Petrel.
    Includes amplitude, phase, frequency, and structural attributes.
    """

    def __init__(self, use_gpu: bool = True, cache_enabled: bool = True):
        """
        Initialize seismic attributes processor.

        Args:
            use_gpu: Whether to use GPU acceleration when available
            cache_enabled: Whether to cache computed attributes
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._cache_lock = threading.Lock()

        # CPU count for parallel processing
        self.n_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores max

        print(f"SeismicAttributes initialized - GPU: {self.use_gpu}, Cores: {self.n_cores}")

    def compute_slice_attributes(self, slice_data: np.ndarray,
                               attribute_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Compute multiple seismic attributes for a single slice.

        Args:
            slice_data: 2D seismic slice
            attribute_types: List of attributes to compute. If None, computes all.

        Returns:
            Dictionary of computed attributes
        """
        if attribute_types is None:
            attribute_types = ['amplitude', 'phase', 'frequency', 'similarity',
                             'dip_azimuth', 'dip_magnitude', 'edge_strength']

        # Check cache first
        cache_key = self._get_cache_key(slice_data.shape, attribute_types)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        attributes = {}

        # Parallel computation for independent attributes
        with ThreadPoolExecutor(max_workers=min(len(attribute_types), 4)) as executor:
            futures = {}

            # Submit computation tasks
            if 'amplitude' in attribute_types:
                futures['amplitude'] = executor.submit(self.compute_amplitude, slice_data)

            if 'phase' in attribute_types:
                futures['phase'] = executor.submit(self.compute_phase, slice_data)

            if 'frequency' in attribute_types:
                futures['frequency'] = executor.submit(self.compute_instantaneous_frequency, slice_data)

            if 'similarity' in attribute_types:
                futures['similarity'] = executor.submit(self.compute_similarity, slice_data)

            if 'dip_azimuth' in attribute_types or 'dip_magnitude' in attribute_types:
                dip_result = executor.submit(self.compute_dip_attributes, slice_data)
                futures['dip'] = dip_result

            if 'edge_strength' in attribute_types:
                futures['edge_strength'] = executor.submit(self.compute_edge_strength, slice_data)

            # Collect results
            for attr_name, future in futures.items():
                try:
                    if attr_name == 'dip':
                        dip_azimuth, dip_magnitude = future.result()
                        if 'dip_azimuth' in attribute_types:
                            attributes['dip_azimuth'] = dip_azimuth
                        if 'dip_magnitude' in attribute_types:
                            attributes['dip_magnitude'] = dip_magnitude
                    else:
                        attributes[attr_name] = future.result()
                except Exception as e:
                    print(f"Error computing {attr_name}: {e}")
                    attributes[attr_name] = np.zeros_like(slice_data)

        # Cache results
        if self.cache_enabled:
            with self._cache_lock:
                self._cache[cache_key] = attributes.copy()

        return attributes

    def compute_amplitude(self, data: np.ndarray) -> np.ndarray:
        """Compute amplitude attribute (envelope/reflection strength)."""
        if self.use_gpu:
            return self._compute_amplitude_gpu(data)
        else:
            return self._compute_amplitude_cpu(data)

    def _compute_amplitude_cpu(self, data: np.ndarray) -> np.ndarray:
        """CPU-based amplitude computation using Hilbert transform."""
        # Apply Hilbert transform for envelope
        analytic_signal = signal.hilbert(data, axis=0)
        amplitude = np.abs(analytic_signal)
        return amplitude

    def _compute_amplitude_gpu(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated amplitude computation."""
        try:
            data_tensor = torch.tensor(data, dtype=torch.complex64, device=self.device)
            # Use torch's FFT for Hilbert transform approximation
            fft_data = torch.fft.fft(data_tensor, dim=0)
            # Create Hilbert mask
            n = data.shape[0]
            mask = torch.ones(n, device=self.device, dtype=torch.complex64)
            mask[n//2+1:] = 0  # Zero out negative frequencies
            mask[1:n//2] *= 2  # Double positive frequencies

            hilbert_data = torch.fft.ifft(fft_data * mask, dim=0)
            amplitude = torch.abs(hilbert_data)
            return amplitude.cpu().numpy()
        except Exception as e:
            print(f"GPU amplitude computation failed, falling back to CPU: {e}")
            return self._compute_amplitude_cpu(data)

    def compute_phase(self, data: np.ndarray) -> np.ndarray:
        """Compute instantaneous phase."""
        if self.use_gpu:
            return self._compute_phase_gpu(data)
        else:
            return self._compute_phase_cpu(data)

    def _compute_phase_cpu(self, data: np.ndarray) -> np.ndarray:
        """CPU-based phase computation."""
        analytic_signal = signal.hilbert(data, axis=0)
        phase = np.angle(analytic_signal)
        return phase

    def _compute_phase_gpu(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated phase computation."""
        try:
            data_tensor = torch.tensor(data, dtype=torch.complex64, device=self.device)
            fft_data = torch.fft.fft(data_tensor, dim=0)

            # Hilbert mask
            n = data.shape[0]
            mask = torch.ones(n, device=self.device, dtype=torch.complex64)
            mask[n//2+1:] = 0
            mask[1:n//2] *= 2

            hilbert_data = torch.fft.ifft(fft_data * mask, dim=0)
            phase = torch.angle(hilbert_data)
            return phase.cpu().numpy()
        except Exception as e:
            print(f"GPU phase computation failed, falling back to CPU: {e}")
            return self._compute_phase_cpu(data)

    def compute_instantaneous_frequency(self, data: np.ndarray) -> np.ndarray:
        """Compute instantaneous frequency."""
        # Compute phase
        phase = self.compute_phase(data)

        # Compute frequency as derivative of phase
        freq = np.zeros_like(phase)

        # Central difference for interior points
        freq[1:-1] = (phase[2:] - phase[:-2]) / (2 * np.pi)

        # Forward/backward difference for boundaries
        freq[0] = (phase[1] - phase[0]) / np.pi
        freq[-1] = (phase[-1] - phase[-2]) / np.pi

        return freq

    def compute_similarity(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Compute trace-to-trace similarity (coherence-like attribute)."""
        if self.use_gpu:
            return self._compute_similarity_gpu(data, window_size)
        else:
            return self._compute_similarity_cpu(data, window_size)

    def _compute_similarity_cpu(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """CPU-based similarity computation."""
        h, w = data.shape
        similarity = np.zeros((h, w))

        # Compute similarity for each trace pair
        for i in range(w):
            for j in range(max(0, i - window_size), min(w, i + window_size + 1)):
                if i != j:
                    # Cross-correlation coefficient
                    trace1 = data[:, i]
                    trace2 = data[:, j]

                    # Remove mean
                    trace1_dm = trace1 - np.mean(trace1)
                    trace2_dm = trace2 - np.mean(trace2)

                    # Compute correlation
                    numerator = np.sum(trace1_dm * trace2_dm)
                    denominator = np.sqrt(np.sum(trace1_dm**2) * np.sum(trace2_dm**2))

                    if denominator > 1e-10:
                        corr = numerator / denominator
                        similarity[:, i] += corr
                        similarity[:, j] += corr

        # Average similarity
        count = np.zeros(w)
        for i in range(w):
            neighbors = min(i + window_size + 1, w) - max(0, i - window_size)
            count[i] = neighbors - 1  # Exclude self

        similarity = similarity / count[np.newaxis, :]
        return np.clip(similarity, -1, 1)

    def _compute_similarity_gpu(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """GPU-accelerated similarity computation."""
        try:
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            h, w = data.shape

            similarity = torch.zeros((h, w), device=self.device)

            # Vectorized computation
            for offset in range(1, window_size + 1):
                # Positive offset
                if offset < w:
                    trace1 = data_tensor[:, :-offset]
                    trace2 = data_tensor[:, offset:]

                    # Remove mean
                    trace1_dm = trace1 - torch.mean(trace1, dim=0, keepdim=True)
                    trace2_dm = trace2 - torch.mean(trace2, dim=0, keepdim=True)

                    # Correlation
                    numerator = torch.sum(trace1_dm * trace2_dm, dim=0)
                    denominator = torch.sqrt(torch.sum(trace1_dm**2, dim=0) * torch.sum(trace2_dm**2, dim=0))

                    mask = denominator > 1e-10
                    corr = torch.zeros_like(numerator)
                    corr[mask] = numerator[mask] / denominator[mask]

                    # Add to similarity matrix
                    similarity[:, :-offset] += corr
                    similarity[:, offset:] += corr

            # Average by number of neighbors
            count = torch.zeros(w, device=self.device)
            for i in range(w):
                neighbors = min(i + window_size + 1, w) - max(0, i - window_size)
                count[i] = neighbors - 1

            similarity = similarity / count.unsqueeze(0)
            return torch.clamp(similarity, -1, 1).cpu().numpy()

        except Exception as e:
            print(f"GPU similarity computation failed, falling back to CPU: {e}")
            return self._compute_similarity_cpu(data, window_size)

    def compute_dip_attributes(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dip azimuth and magnitude."""
        # Simple dip estimation using gradient
        grad_y, grad_x = np.gradient(data)

        # Compute dip magnitude (steepness)
        dip_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Compute dip azimuth (direction)
        dip_azimuth = np.arctan2(grad_y, grad_x)

        return dip_azimuth, dip_magnitude

    def compute_edge_strength(self, data: np.ndarray) -> np.ndarray:
        """Compute edge strength using gradient magnitude."""
        # Simple edge detection using Sobel operator
        sobel_x = ndimage.sobel(data, axis=1)
        sobel_y = ndimage.sobel(data, axis=0)

        edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
        return edge_strength

    def compute_volume_attributes(self, volume: np.ndarray,
                                attribute_types: List[str] = None,
                                slice_axis: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute attributes for an entire 3D volume.

        Args:
            volume: 3D seismic volume
            attribute_types: List of attributes to compute
            slice_axis: Axis along which to slice (0=time/depth, 1=crossline, 2=inline)

        Returns:
            Dictionary of 3D attribute volumes
        """
        if attribute_types is None:
            attribute_types = ['amplitude', 'phase', 'frequency', 'similarity']

        print(f"Computing volume attributes for axis {slice_axis}, volume shape: {volume.shape}")

        # Process slices in parallel
        n_slices = volume.shape[slice_axis]
        attribute_volumes = {}

        # Initialize volume dictionaries
        for attr_type in attribute_types:
            shape = list(volume.shape)
            attribute_volumes[attr_type] = np.zeros(shape, dtype=np.float32)

        # Process slices with progress tracking
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            futures = []

            for i in range(n_slices):
                if slice_axis == 0:
                    slice_data = volume[i, :, :]
                elif slice_axis == 1:
                    slice_data = volume[:, i, :]
                else:  # slice_axis == 2
                    slice_data = volume[:, :, i]

                future = executor.submit(self.compute_slice_attributes, slice_data, attribute_types)
                futures.append((i, future))

            # Collect results
            for i, future in futures:
                try:
                    slice_attrs = future.result()
                    for attr_name, attr_data in slice_attrs.items():
                        if slice_axis == 0:
                            attribute_volumes[attr_name][i, :, :] = attr_data
                        elif slice_axis == 1:
                            attribute_volumes[attr_name][:, i, :] = attr_data
                        else:  # slice_axis == 2
                            attribute_volumes[attr_name][:, :, i] = attr_data
                except Exception as e:
                    print(f"Error processing slice {i}: {e}")

        return attribute_volumes

    def _get_cache_key(self, shape: Tuple[int, ...], attribute_types: List[str]) -> str:
        """Generate cache key for computed attributes."""
        return f"{shape}_{'_'.join(sorted(attribute_types))}"

    def clear_cache(self):
        """Clear the attribute cache."""
        with self._cache_lock:
            self._cache.clear()
            print("Attribute cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cache_items': len(self._cache)
        }

    # ==================== FAULT DETECTION ATTRIBUTES ====================
    
    def compute_coherence(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Compute coherence attribute for fault detection.
        Low coherence indicates discontinuities (faults).
        
        Args:
            data: 2D seismic slice
            window_size: Size of analysis window
            
        Returns:
            Coherence map (0 = fault/discontinuity, 1 = continuous)
        """
        from scipy.ndimage import uniform_filter
        
        h, w = data.shape
        coherence = np.ones((h, w), dtype=np.float32)
        
        # Compute local semblance/coherence
        half_win = window_size // 2
        
        for i in range(half_win, w - half_win):
            # Get window of traces
            window = data[:, i-half_win:i+half_win+1]
            
            # Compute coherence as normalized cross-correlation
            mean_trace = np.mean(window, axis=1)
            
            # Sum of cross-correlations
            numerator = np.sum(mean_trace ** 2)
            denominator = np.mean(np.sum(window ** 2, axis=0))
            
            if denominator > 1e-10:
                coherence[:, i] = numerator / (denominator + 1e-10)
        
        # Normalize to 0-1
        coherence = np.clip(coherence, 0, 1)
        
        # Smooth slightly
        coherence = uniform_filter(coherence, size=3)
        
        return coherence
    
    def compute_variance(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Compute variance attribute for fault detection.
        High variance indicates edges and faults.
        
        Args:
            data: 2D seismic slice
            window_size: Size of analysis window
            
        Returns:
            Variance map (high = fault/edge)
        """
        from scipy.ndimage import uniform_filter
        
        # Local mean
        local_mean = uniform_filter(data.astype(np.float64), size=window_size)
        
        # Local variance
        local_sq_mean = uniform_filter(data.astype(np.float64) ** 2, size=window_size)
        variance = local_sq_mean - local_mean ** 2
        
        # Normalize
        variance = variance / (variance.max() + 1e-10)
        
        return variance.astype(np.float32)
    
    def compute_fault_likelihood(self, data: np.ndarray) -> np.ndarray:
        """
        Compute fault likelihood combining multiple attributes.
        
        Args:
            data: 2D seismic slice
            
        Returns:
            Fault likelihood map (0-1, high = likely fault)
        """
        from scipy.ndimage import sobel, gaussian_filter
        
        # 1. Coherence (low = fault)
        coherence = self.compute_coherence(data, window_size=5)
        fault_from_coherence = 1.0 - coherence
        
        # 2. Variance (high = fault)
        variance = self.compute_variance(data, window_size=5)
        
        # 3. Vertical edge detection (faults are often vertical/diagonal)
        smoothed = gaussian_filter(data.astype(float), sigma=1)
        vertical_edges = np.abs(sobel(smoothed, axis=1))  # Horizontal derivative
        vertical_edges = vertical_edges / (vertical_edges.max() + 1e-10)
        
        # 4. Dip discontinuity
        dip_azimuth, dip_mag = self.compute_dip_attributes(data)
        dip_gradient = np.abs(np.gradient(dip_azimuth, axis=1))
        dip_discontinuity = dip_gradient / (dip_gradient.max() + 1e-10)
        
        # Combine attributes (weighted sum)
        fault_likelihood = (
            0.3 * fault_from_coherence +
            0.2 * variance +
            0.3 * vertical_edges +
            0.2 * dip_discontinuity
        )
        
        # Normalize to 0-1
        fault_likelihood = np.clip(fault_likelihood, 0, 1)
        
        # Apply slight smoothing
        fault_likelihood = gaussian_filter(fault_likelihood, sigma=1)
        
        return fault_likelihood.astype(np.float32)
    
    def extract_fault_lines(self, fault_likelihood: np.ndarray, 
                           threshold: float = 0.3,
                           min_length: int = 20) -> List[List[Tuple[int, int]]]:
        """
        Extract fault lines from fault likelihood map.
        
        Args:
            fault_likelihood: Fault probability map
            threshold: Minimum likelihood to consider
            min_length: Minimum fault line length
            
        Returns:
            List of fault lines, each as list of (x, y) points
        """
        from scipy.ndimage import binary_dilation, binary_erosion, label
        
        # Threshold to binary
        fault_binary = fault_likelihood > threshold
        
        # Check if we have any faults
        if not np.any(fault_binary):
            print(f"  No pixels above threshold {threshold} (max: {fault_likelihood.max():.3f})")
            return []
        
        print(f"  Pixels above threshold: {np.sum(fault_binary)}")
        
        # Clean up - gentle morphological operations
        fault_binary = binary_dilation(fault_binary, iterations=1)
        fault_binary = binary_erosion(fault_binary, iterations=1)
        
        # Try to use skimage for skeletonization, otherwise use simple approach
        try:
            from skimage.morphology import skeletonize, remove_small_objects
            fault_binary = remove_small_objects(fault_binary, min_size=min_length)
            fault_skeleton = skeletonize(fault_binary)
        except ImportError:
            # Simple fallback - just use the binary mask
            fault_skeleton = fault_binary
        except Exception as e:
            print(f"  Skeletonization failed: {e}")
            fault_skeleton = fault_binary
        
        # Extract connected components as fault lines
        labeled, num_features = label(fault_skeleton)
        print(f"  Found {num_features} connected components")
        
        fault_lines = []
        for i in range(1, num_features + 1):
            # Get points for this fault
            ys, xs = np.where(labeled == i)
            
            if len(xs) < min_length:
                continue
            
            # Sort by y coordinate (typically faults go top to bottom)
            sorted_indices = np.argsort(ys)
            points = [(int(xs[j]), int(ys[j])) for j in sorted_indices]
            
            fault_lines.append(points)
        
        return fault_lines
    
    def detect_faults(self, data: np.ndarray, 
                     threshold: float = 0.4,
                     min_length: int = 20) -> Dict:
        """
        Complete fault detection pipeline.
        
        Args:
            data: 2D seismic slice
            threshold: Fault likelihood threshold
            min_length: Minimum fault line length
            
        Returns:
            Dictionary with fault_likelihood, fault_lines, and fault_mask
        """
        # Compute fault likelihood
        fault_likelihood = self.compute_fault_likelihood(data)
        
        # Extract fault lines
        fault_lines = self.extract_fault_lines(fault_likelihood, threshold, min_length)
        
        # Create fault mask
        fault_mask = fault_likelihood > threshold
        
        return {
            'fault_likelihood': fault_likelihood,
            'fault_lines': fault_lines,
            'fault_mask': fault_mask,
            'num_faults': len(fault_lines)
        }
