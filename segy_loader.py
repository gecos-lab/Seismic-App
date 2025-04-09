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
    
    def create_figure(self, slice_data, vmin=None, vmax=None, cmap='gray'):
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
        im = ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
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