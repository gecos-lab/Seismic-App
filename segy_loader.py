import numpy as np
import segyio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from scipy import ndimage

class SegyLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.segy_file = None
        self.data = None
        self.shape = None
        self.inlines = None
        self.crosslines = None
        self.timeslices = None
        
    def load_file(self, file_path=None):
        """Load a SEGY file and extract data"""
        if file_path:
            self.file_path = file_path
            
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"SEGY file not found: {self.file_path}")
            
        # Load SEGY file with segyio
        try:
            self.segy_file = segyio.open(self.file_path, 'r')
            self.segy_file.mmap()  # Memory map for faster access
            
            # Get basic metadata
            self.inlines = self.segy_file.ilines
            self.crosslines = self.segy_file.xlines
            self.timeslices = range(len(self.segy_file.samples))
            
            # Extract full volume
            self.data = segyio.tools.cube(self.segy_file)
            self.shape = self.data.shape
            
            print(f"Loaded SEGY file: {self.file_path}")
            print(f"Volume shape: {self.shape}")
            print(f"Inlines: {len(self.inlines)}, Crosslines: {len(self.crosslines)}, Time samples: {len(self.timeslices)}")
            
            return True
        except Exception as e:
            print(f"Error loading SEGY file: {e}")
            return False

    def generate_synthetic(self, ni=80, nj=200, nt=800, dt_ms=2.0, freq=25.0, snr_db=30.0, seed=7,
                            num_reflectors=5, fault={'x': None, 'throw': 12}):
        """Generate a clean synthetic seismic volume in-memory.

        Args:
            ni, nj, nt: inlines, crosslines, time samples
            dt_ms: sample rate in ms (metadata only for export)
            freq: dominant wavelet frequency (Hz, used if SciPy available)
            snr_db: signal-to-noise ratio in dB
            seed: RNG seed for reproducibility
            num_reflectors: number of primary reflectors
            fault: dict with optional 'x' (crossline index) and 'throw' (samples)
        """
        rng = np.random.default_rng(seed)
        # Create horizon times (samples) with gentle dip and undulation
        I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')  # (ni,nj)
        horizons = []
        base = 120
        step = max(60, nt // (num_reflectors + 2))
        for k in range(num_reflectors):
            t0 = base + k * step
            dip_i = rng.uniform(0.1, 0.6) * (1 if k % 2 == 0 else -1)
            dip_j = rng.uniform(0.05, 0.3) * (1 if (k // 2) % 2 == 0 else -1)
            und_i = 8 + 6 * rng.random()
            und_j = 8 + 6 * rng.random()
            phi_i = rng.uniform(0, 2 * np.pi)
            phi_j = rng.uniform(0, 2 * np.pi)
            T = (
                t0
                + dip_i * I
                + dip_j * J
                + 6 * np.sin(2 * np.pi * I / max(1, ni) + phi_i)
                + 6 * np.sin(2 * np.pi * J / max(1, nj) + phi_j)
            )
            horizons.append(T)
        reflectivity = np.zeros((ni, nj, nt), dtype=float)
        # Optional simple normal fault: shift deeper horizons on one side
        if fault is None:
            fault = {'x': None, 'throw': 0}
        fx = fault.get('x')
        fthrow = int(fault.get('throw', 0))
        if fx is None:
            fx = int(0.6 * nj)
        # Stamp impulses for each horizon
        amps = (rng.random(len(horizons)) * 0.6 + 0.4) * ((rng.integers(0, 2, len(horizons)) * 2) - 1)
        for idx, (T, a) in enumerate(zip(horizons, amps)):
            # Apply throw to right side for deeper reflectors
            T_shift = T.copy()
            if fthrow and idx >= 2:
                T_shift[:, fx:] = T_shift[:, fx:] + fthrow
            t_idx = np.clip(np.rint(T_shift), 0, nt - 1).astype(int)
            # Vectorized stamping
            rows = np.repeat(np.arange(ni), nj)
            cols = np.tile(np.arange(nj), ni)
            times = t_idx.reshape(-1)
            reflectivity[rows, cols, times] += a

        # Convolve along time with a Ricker-like wavelet (if SciPy present)
        try:
            from scipy.signal import ricker, fftconvolve
            dt = dt_ms / 1000.0
            # Choose wavelet length ~ 0.256 s
            wlen = int(max(16, round(0.256 / dt)))
            if wlen % 2 == 0:
                wlen += 1
            # ricker takes width parameter; approximate
            w = ricker(wlen, a=max(1.0, (wlen * freq * dt) / np.pi))
            w = w / (np.max(np.abs(w)) + 1e-6)
            volume = np.zeros_like(reflectivity)
            for i in range(ni):
                for j in range(nj):
                    volume[i, j] = fftconvolve(reflectivity[i, j], w, mode='same')
        except Exception:
            # Fallback: simple temporal smoothing
            volume = ndimage.gaussian_filter1d(reflectivity, sigma=1.2, axis=2)

        # Add controlled noise
        if snr_db is not None:
            sig_std = np.std(volume)
            if sig_std < 1e-6:
                sig_std = 1.0
            noise_std = sig_std / (10 ** (snr_db / 20.0))
            volume = volume + rng.normal(0.0, noise_std, size=volume.shape)

        # Normalize
        vmax = np.percentile(np.abs(volume), 99.5)
        if vmax > 0:
            volume = volume / vmax

        self.data = volume.astype(np.float32)
        self.shape = self.data.shape
        self.inlines = np.arange(ni)
        self.crosslines = np.arange(nj)
        self.timeslices = np.arange(nt)
        self.file_path = None
        print(f"Generated synthetic volume: shape={self.shape}, dt={dt_ms}ms, freq~{freq}Hz, SNR={snr_db}dB")
        return True
    
    def get_inline_slice(self, inline_idx):
        """Get a specific inline slice"""
        if self.data is None:
            raise ValueError("No SEGY data loaded")
        
        if inline_idx < 0 or inline_idx >= len(self.inlines):
            raise IndexError(f"Inline index out of range: {inline_idx}")
            
        # Extract and transpose the data to ensure horizontal orientation
        slice_data = self.data[inline_idx, :, :]
        # Swap axes to get horizontal orientation (time/depth as y-axis)
        return slice_data.T
    
    def get_crossline_slice(self, crossline_idx):
        """Get a specific crossline slice"""
        if self.data is None:
            raise ValueError("No SEGY data loaded")
        
        if crossline_idx < 0 or crossline_idx >= len(self.crosslines):
            raise IndexError(f"Crossline index out of range: {crossline_idx}")
            
        # Extract and transpose the data to ensure horizontal orientation
        slice_data = self.data[:, crossline_idx, :]
        # Swap axes to get horizontal orientation (time/depth as y-axis)
        return slice_data.T
    
    def get_timeslice(self, time_idx):
        """Get a specific time slice"""
        if self.data is None:
            raise ValueError("No SEGY data loaded")
        
        if time_idx < 0 or time_idx >= len(self.timeslices):
            raise IndexError(f"Time index out of range: {time_idx}")
            
        # For timeslice, the orientation is different
        return self.data[:, :, time_idx]
    
    def create_figure(self, slice_data, vmin=None, vmax=None, cmap='seismic_r'):
        """Create a matplotlib figure from slice data"""
        # No need to transpose or rotate inline/crossline slices as they're already properly oriented
        # Just add rotation if needed for specific slice types
        
        if vmin is None:
            vmin = np.percentile(slice_data, 5)
        if vmax is None:
            vmax = np.percentile(slice_data, 95)
            
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display with standard orientation
        im = ax.imshow(slice_data, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Set proper axis labels
        ax.set_xlabel('Trace Position')
        ax.set_ylabel('Time/Depth')
        
        fig.colorbar(im)
        return fig
        
    def close(self):
        """Close the SEGY file"""
        if self.segy_file:
            self.segy_file.close()
            print("SEGY file closed") 