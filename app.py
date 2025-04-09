import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import argparse
# setting proper environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Add parent directory to path for importing our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seismic_app.segy_loader import SegyLoader
from seismic_app.seismic_predictor import SeismicPredictor

# Import PyVista for 3D visualization
try:
    import pyvista as pv
    import numpy as np
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. 3D visualization will be disabled.")

# Import LoopStructural for surface generation
try:
    # Try various import paths that might be used depending on installation method
    try:
        import LoopStructural as ls
        from LoopStructural import GeologicalModel
        from LoopStructural.interpolators import PiecewiseLinearInterpolator, BiharmonicInterpolator
        LOOPSTRUCTURAL_AVAILABLE = True
    except ImportError:
        # Try alternate import path (sometimes package names vary by installation)
        import loopstructural as ls
        from loopstructural import GeologicalModel
        from loopstructural.interpolators import PiecewiseLinearInterpolator, BiharmonicInterpolator
        LOOPSTRUCTURAL_AVAILABLE = True
except ImportError:
    LOOPSTRUCTURAL_AVAILABLE = False
    print("LoopStructural not available. Surface generation will be disabled.")
    
# Debug statement to show whether LoopStructural is available
print(f"LoopStructural available: {LOOPSTRUCTURAL_AVAILABLE}")

class SeismicApp(ttk.Frame):
    def __init__(self, master=None, segy_path=None, demo_mode=False, model_id=None):
        super().__init__(master)
        self.master = master
        self.master.title("Seismic Interpretation App")
        self.pack(fill=tk.BOTH, expand=True)
        
        # Initialize the seismic predictor with SAM2 model
        # Set demo_mode=False to use the real model when available
        self.predictor = SeismicPredictor(demo_mode=demo_mode, model_id=model_id)
        
        # Try to load the real model (will fall back to demo mode if unavailable)
        success = self.predictor.load_model()
        if success and not self.predictor.demo_mode:
            print("Successfully loaded SAM2 model for seismic interpretation")
        else:
            print("Using demo mode for seismic interpretation")
        
        # Initialize seismic data
        self.segy_loader = SegyLoader()
        
        self.current_slice_type = tk.StringVar(value="inline")
        self.current_slice_idx = tk.IntVar(value=0)
        self.current_object_id = tk.IntVar(value=1)
        
        # Initialize point collection for annotations
        self.points = []
        self.point_labels = []
        self.drawing_mode = tk.StringVar(value="foreground")  # or "background"
        
        # Status variables
        self.status_text = tk.StringVar(value="Ready. Load a SEGY file to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)
        
        # Message queue for thread communication
        self.queue = queue.Queue()
        
        # Create GUI
        self._create_menu()
        self._create_main_layout()
        
        # Bind events
        self._bind_events()
        
        # Start with model loading
        self._load_model()
        
        # Start queue processing
        self.process_queue()
        
    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.master)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open SEGY...", command=self._open_segy_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Slice menu
        slice_menu = tk.Menu(menubar, tearoff=0)
        slice_menu.add_radiobutton(label="Inline", variable=self.current_slice_type, 
                                  value="inline", command=self._update_slice_view)
        slice_menu.add_radiobutton(label="Crossline", variable=self.current_slice_type, 
                                  value="crossline", command=self._update_slice_view)
        slice_menu.add_radiobutton(label="Time/Depth", variable=self.current_slice_type, 
                                  value="timeslice", command=self._update_slice_view)
        menubar.add_cascade(label="Slice Type", menu=slice_menu)
        
        # SAM2 menu
        sam2_menu = tk.Menu(menubar, tearoff=0)
        sam2_menu.add_command(label="Clear Current Annotations", command=self._clear_annotations)
        sam2_menu.add_command(label="Propagate to All Slices", command=self._propagate_to_all)
        # Add 3D Visualization option to the menu
        sam2_menu.add_command(label="Open 3D Visualization", command=self._open_3d_visualization)
        # Add 3D Surface Generation option to the menu
        sam2_menu.add_command(label="Generate 3D Surface", command=self._open_3d_surface_generation)
        menubar.add_cascade(label="SAM2", menu=sam2_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Instructions", command=self._show_instructions)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.master.config(menu=menubar)
    
    def _create_main_layout(self):
        """Create the main application layout"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Slice controls
        slice_frame = ttk.Frame(control_frame)
        slice_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Label(slice_frame, text="Slice Index:").pack(side=tk.LEFT, padx=(0, 5))
        self.slice_scale = ttk.Scale(slice_frame, from_=0, to=100, 
                                     variable=self.current_slice_idx,
                                     command=self._on_slice_change)
        self.slice_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.slice_entry = ttk.Entry(slice_frame, width=5, textvariable=self.current_slice_idx)
        self.slice_entry.pack(side=tk.LEFT, padx=5)
        self.slice_entry.bind("<Return>", self._on_slice_entry_change)
        
        # Annotation controls
        annot_frame = ttk.LabelFrame(control_frame, text="Annotation")
        annot_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(annot_frame, text="Foreground", variable=self.drawing_mode, 
                       value="foreground").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(annot_frame, text="Background", variable=self.drawing_mode, 
                       value="background").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(annot_frame, text="Object ID:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Spinbox(annot_frame, from_=1, to=10, width=3, 
                   textvariable=self.current_object_id).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Generate Mask", 
                  command=self._generate_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear Points", 
                  command=self._clear_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Propagate", 
                  command=self._propagate_to_all).pack(side=tk.LEFT, padx=5)
        # Add 3D Visualization button
        ttk.Button(action_frame, text="3D View", 
                  command=self._open_3d_visualization).pack(side=tk.LEFT, padx=5)
        # Add Surface Generation button
        ttk.Button(action_frame, text="3D Surface", 
                  command=self._open_3d_surface_generation).pack(side=tk.LEFT, padx=5)
        
        # Canvas for displaying the slice and annotations
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure and canvas for seismic display
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        self.toolbar.update()
        
        # Status bar at bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           mode='determinate', length=200)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT, fill=tk.X)
    
    def _bind_events(self):
        """Bind events to widgets"""
        # Canvas click events for annotations
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        # Window close event
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_canvas_click(self, event):
        """Handle click events on the canvas for adding annotation points"""
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            return
            
        if event.xdata is None or event.ydata is None:
            return  # Click outside the plot area
            
        # Add the point and label
        self.points.append([event.xdata, event.ydata])
        label = 1 if self.drawing_mode.get() == "foreground" else 0
        self.point_labels.append(label)
        
        # Update the display
        self._update_display_with_points()
        
    def _update_display_with_points(self):
        """Update the display with current slice and annotation points"""
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            return
            
        # Clear the axis
        self.ax.clear()
        
        # Display the slice
        vmin, vmax = np.percentile(self.current_slice, [5, 95])
        self.ax.imshow(self.current_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add foreground points (red)
        fg_points = [p for i, p in enumerate(self.points) if self.point_labels[i] == 1]
        if fg_points:
            fg_points = np.array(fg_points)
            self.ax.scatter(fg_points[:, 0], fg_points[:, 1], color='red', marker='o', s=30)
            
        # Add background points (blue)
        bg_points = [p for i, p in enumerate(self.points) if self.point_labels[i] == 0]
        if bg_points:
            bg_points = np.array(bg_points)
            self.ax.scatter(bg_points[:, 0], bg_points[:, 1], color='blue', marker='o', s=30)
            
        # Update title
        self.ax.set_title(f"{self.current_slice_type.get().capitalize()} {self.current_slice_idx.get()}")
        
        # Redraw canvas
        self.canvas.draw()
    
    def _open_segy_file(self):
        """Open a SEGY file dialog and load the selected file"""
        filepath = filedialog.askopenfilename(
            title="Open SEGY File",
            filetypes=[("SEGY files", "*.segy *.sgy"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        self.status_text.set(f"Loading SEGY file: {os.path.basename(filepath)}...")
        self.progress_var.set(10)
        
        # Start loading in a thread
        threading.Thread(target=self._load_segy_file_thread, args=(filepath,), daemon=True).start()
    
    def _load_segy_file_thread(self, filepath):
        """Thread function for loading SEGY file"""
        try:
            # Load the SEGY file
            success = self.segy_loader.load_file(filepath)
            
            if success:
                # Set up the slice scale limits based on data dimensions
                if self.current_slice_type.get() == "inline":
                    max_slice = len(self.segy_loader.inlines) - 1
                elif self.current_slice_type.get() == "crossline":
                    max_slice = len(self.segy_loader.crosslines) - 1
                else:  # timeslice
                    max_slice = len(self.segy_loader.timeslices) - 1
                
                # Update UI in main thread
                self.queue.put(("update_scale", max_slice))
                
                # Set seismic volume in predictor
                self.predictor.set_seismic_volume(self.segy_loader)
                
                # Load initial slice
                self._load_current_slice()
                
                # Update status
                self.queue.put(("status", f"Loaded SEGY file: {os.path.basename(filepath)}"))
                self.queue.put(("progress", 100))
            else:
                self.queue.put(("error", f"Failed to load SEGY file: {os.path.basename(filepath)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            self.queue.put(("error", f"Error loading SEGY file: {str(e)}"))
            self.queue.put(("progress", 0))
    
    def _load_model(self):
        """Load the SAM2 model"""
        self.status_text.set("Loading SAM2 model...")
        self.progress_var.set(5)
        
        # Start loading in a thread
        threading.Thread(target=self._load_model_thread, daemon=True).start()
    
    def _load_model_thread(self):
        """Thread function for loading the SAM2 model"""
        try:
            success = self.predictor.load_model()
            
            if success:
                self.queue.put(("status", "SAM2 model loaded successfully"))
                self.queue.put(("progress", 100))
            else:
                self.queue.put(("error", "Failed to load SAM2 model"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            self.queue.put(("error", f"Error loading SAM2 model: {str(e)}"))
            self.queue.put(("progress", 0))
    
    def _update_slice_view(self):
        """Update the view when slice type changes"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        # Update the slice scale limits based on data dimensions
        if self.current_slice_type.get() == "inline":
            max_slice = len(self.segy_loader.inlines) - 1
        elif self.current_slice_type.get() == "crossline":
            max_slice = len(self.segy_loader.crosslines) - 1
        else:  # timeslice
            max_slice = len(self.segy_loader.timeslices) - 1
            
        self.slice_scale.configure(to=max_slice)
        
        # Reset index if out of bounds
        if self.current_slice_idx.get() > max_slice:
            self.current_slice_idx.set(0)
            
        # Clear any existing annotations
        self._clear_annotations()
        
        # Load the current slice
        self._load_current_slice()
    
    def _on_slice_change(self, event=None):
        """Handle slice slider change"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        # Load the new slice
        self._load_current_slice()
    
    def _on_slice_entry_change(self, event=None):
        """Handle slice entry change"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        try:
            idx = int(self.current_slice_idx.get())
            max_idx = self.slice_scale.cget("to")
            
            if idx < 0:
                idx = 0
            elif idx > max_idx:
                idx = int(max_idx)
                
            self.current_slice_idx.set(idx)
            self._load_current_slice()
            
        except ValueError:
            # Reset to previous value
            self.current_slice_idx.set(int(self.slice_scale.get()))
    
    def _load_current_slice(self):
        """Load and display the current slice"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        # Get the current slice
        idx = self.current_slice_idx.get()
        slice_type = self.current_slice_type.get()
        
        print(f"\nLoading {slice_type} slice {idx}")
        
        try:
            if slice_type == "inline":
                self.current_slice = self.segy_loader.get_inline_slice(idx)
            elif slice_type == "crossline":
                self.current_slice = self.segy_loader.get_crossline_slice(idx)
            else:  # timeslice
                self.current_slice = self.segy_loader.get_timeslice(idx)
                
            print(f"Slice loaded successfully, shape: {self.current_slice.shape}")
                
            # Set current slice in predictor
            self.predictor.set_current_slice(slice_type, idx, self.current_slice)
            
            # If there's a propagated mask available, try to load it
            if self.predictor.demo_mode and hasattr(self.predictor, 'demo_inference_state'):
                print(f"Checking for demo mask")
                
                try:
                    # In demo mode, we just need to find the frame position
                    frame_pos = idx  # First try the direct position
                    
                    # Debug output to see what's available in output
                    output_frames = list(self.predictor.demo_inference_state["output"].keys())
                    print(f"Available demo frames: {output_frames[:10]}{'...' if len(output_frames) > 10 else ''}")
                    
                    obj_id = self.current_object_id.get()
                    
                    # Get the mask for this frame
                    mask = None
                    try:
                        mask = self.predictor.get_mask_for_frame(frame_pos, obj_id)
                        if mask is not None and np.any(mask):
                            print(f"Found mask for slice {idx}, frame {frame_pos}, shape: {mask.shape}, non-zero: {np.count_nonzero(mask)}")
                            self.display_mask(mask)
                            return
                        else:
                            print(f"No mask content found for slice {idx}, frame {frame_pos}")
                            
                            # Try nearby frames if direct position didn't work
                            for offset in range(1, 10):
                                for pos in [frame_pos - offset, frame_pos + offset]:
                                    if pos in self.predictor.demo_inference_state["output"]:
                                        try:
                                            mask = self.predictor.get_mask_for_frame(pos, obj_id)
                                            if mask is not None and np.any(mask):
                                                print(f"Found mask at nearby position {pos}")
                                                self.display_mask(mask)
                                                return
                                        except Exception:
                                            pass
                    except Exception as e:
                        print(f"Error getting mask for frame {frame_pos}: {e}")
                    
                    # If we still have no mask, check if we're just trying to display the slice without any masks
                    print("No valid mask found, displaying without mask")
                    self._update_display_with_points()
                    return
                    
                except Exception as e:
                    print(f"Could not load demo mask: {e}")
            
            elif hasattr(self.predictor, 'inference_state') and self.predictor.inference_state is not None:
                print(f"Checking for propagated mask")
                
                try:
                    # First try using the slice-to-frame mapping if available
                    frame_pos = None
                    
                    # Check if we have a slice-to-frame mapping
                    if hasattr(self.predictor, 'slice_to_frame_map'):
                        if idx in self.predictor.slice_to_frame_map:
                            frame_pos = self.predictor.slice_to_frame_map[idx]
                            print(f"Using mapped frame position {frame_pos} for slice {idx}")
                    
                    # If no mapping found, try to work out the index based on current slice configuration
                    if frame_pos is None:
                        if self.predictor.demo_mode:
                            # In demo mode, frame_pos is the position in the sequence of frames
                            # For demo mode, we're now using position indices (0 to N-1) instead of actual inline/crossline numbers
                            if hasattr(self.predictor, 'demo_slice_indices'):
                                # For demo mode we need the current position (idx) in the demo_slice_indices
                                demo_indices = list(self.predictor.demo_slice_indices)
                                
                                # Check if current idx is directly in demo_slice_indices
                                if idx in demo_indices:
                                    frame_pos = demo_indices.index(idx)
                                    print(f"Current position {idx} found in demo indices at position {frame_pos}")
                                # Otherwise we can just use the position directly since demo_slice_indices is now [0..N-1]
                                elif 0 <= idx < len(demo_indices):
                                    frame_pos = idx
                                    print(f"Using position index directly: {frame_pos}")
                                else:
                                    # Find closest position
                                    closest_pos = max(0, min(idx, len(demo_indices)-1))
                                    frame_pos = closest_pos
                                    print(f"Using closest position {frame_pos} to {idx}")
                            else:
                                # Fallback to using idx directly as the frame position
                                frame_pos = idx
                                print(f"No demo indices found, using position directly: {frame_pos}")
                        else:
                            # For non-demo mode, still need to find the position in the slice range
                            if slice_type == "inline":
                                slices = list(range(len(self.segy_loader.inlines)))
                            elif slice_type == "crossline":
                                slices = list(range(len(self.segy_loader.crosslines)))
                            else:  # timeslice
                                slices = list(range(len(self.segy_loader.timeslices)))
                                
                            # Check if idx is in the valid range
                            if 0 <= idx < len(slices):
                                frame_pos = idx
                                print(f"Using position {frame_pos} for slice {idx}")
                            else:
                                print(f"Slice {idx} out of range, skipping mask display")
                                self._update_display_with_points()
                                return
                    
                    # If we still don't have a valid frame_pos, skip mask display
                    if frame_pos is None:
                        print(f"Couldn't determine frame position for slice {idx}, skipping mask display")
                        self._update_display_with_points()
                        return
                    
                    obj_id = self.current_object_id.get()
                    print(f"Looking for mask at frame position {frame_pos}, object ID {obj_id}")
                    
                    # Get the mask for this frame
                    mask = None
                    try:
                        mask = self.predictor.get_mask_for_frame(frame_pos, obj_id)
                        if mask is not None and np.any(mask):
                            print(f"Found mask for slice {idx}, frame {frame_pos}, shape: {mask.shape}, non-zero: {np.count_nonzero(mask)}")
                            self.display_mask(mask)
                            return
                        else:
                            print(f"No mask content found (mask is None or empty) for slice {idx}, frame {frame_pos}")
                    except Exception as e:
                        print(f"Error getting mask for slice {idx}, frame {frame_pos}: {e}")
                    
                    # If we reach here, no valid mask was found, just display the slice with points
                    print("No valid mask found, displaying without mask")
                    self._update_display_with_points()
                    return
                    
                except Exception as e:
                    print(f"Could not load propagated mask for slice {idx}: {e}")
                
            # Update the display
            print("No propagation info found, displaying slice with points only")
            self._update_display_with_points()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load slice: {str(e)}")
    
    def _generate_mask(self):
        """Generate mask from annotation points using SAM2"""
        if not self.points:
            messagebox.showinfo("Info", "Please add at least one annotation point first.")
            return
            
        self.status_text.set("Generating mask...")
        self.progress_var.set(50)
        
        # Start prediction in a thread
        threading.Thread(target=self._generate_mask_thread, daemon=True).start()
    
    def _generate_mask_thread(self):
        """Thread function for generating mask"""
        try:
            # Predict masks
            masks, scores, logits = self.predictor.predict_masks_from_points(
                self.points, self.point_labels, multimask_output=True
            )
            
            # Select the mask with highest score
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Update UI in main thread
            self.queue.put(("display_mask", best_mask))
            self.queue.put(("status", f"Mask generated with score: {scores[best_mask_idx]:.4f}"))
            self.queue.put(("progress", 100))
                
        except Exception as e:
            self.queue.put(("error", f"Error generating mask: {str(e)}"))
            self.queue.put(("progress", 0))
    
    def _clear_annotations(self):
        """Clear annotation points"""
        self.points = []
        self.point_labels = []
        self._update_display_with_points()
    
    def _propagate_to_all(self):
        """Propagate the mask to all slices using SAM2 video predictor"""
        if not self.points:
            messagebox.showinfo("Info", "Please add at least one annotation point and generate a mask first.")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Confirm", "This will propagate the mask to all slices. It may take some time. Continue?"):
            return
            
        self.status_text.set("Preparing to propagate masks...")
        self.progress_var.set(10)
        
        # Start propagation in a thread
        threading.Thread(target=self._propagate_thread, daemon=True).start()
    
    def _propagate_thread(self):
        """Thread function for mask propagation"""
        try:
            slice_type = self.current_slice_type.get()
            
            # Determine slice range
            if slice_type == "inline":
                slice_count = len(self.segy_loader.inlines)
                # Use position indices (0 to len-1) instead of actual inline numbers
                slices = list(range(slice_count))
                # For display purposes only
                actual_indices = list(self.segy_loader.inlines) if hasattr(self.segy_loader.inlines, '__iter__') else []
            elif slice_type == "crossline":
                slice_count = len(self.segy_loader.crosslines)
                slices = list(range(slice_count))
                actual_indices = list(self.segy_loader.crosslines) if hasattr(self.segy_loader.crosslines, '__iter__') else []
            else:  # timeslice
                slice_count = len(self.segy_loader.timeslices)
                slices = list(range(slice_count))
                actual_indices = list(range(slice_count))
                
            # Get current slice
            current_frame_idx = self.current_slice_idx.get()
            current_frame_pos = current_frame_idx  # We're using position indices directly
            
            self.queue.put(("status", f"Preparing to propagate masks from {slice_type} {current_frame_idx}..."))
            self.queue.put(("progress", 15))
            
            print(f"Slice type: {slice_type}, Current idx: {current_frame_idx}, Position: {current_frame_pos}")
            
            # Force demo mode if video predictor failed to initialize properly
            use_demo_mode = self.predictor.demo_mode
            
            # Check if we have a real video predictor or if we need to fall back to demo
            if not use_demo_mode and not hasattr(self.predictor, 'video_predictor'):
                print("Video predictor not initialized, falling back to demo mode")
                use_demo_mode = True
                
            # Check if video predictor is a mock (has no __module__ attribute or it's not 'sam2.sam2_video_predictor')
            if not use_demo_mode and (not hasattr(self.predictor.video_predictor, '__module__') or 
                                     'sam2.sam2_video_predictor' not in self.predictor.video_predictor.__module__):
                print("Video predictor is a mock, falling back to demo mode")
                use_demo_mode = True
                
            print(f"Using {'demo' if use_demo_mode else 'real'} mode for propagation")
            
            # In demo mode, use position indices (0 to len-1) to avoid out of range errors
            if use_demo_mode:
                # Store a subset of position indices for demo mode
                MAX_SLICES = 100
                if slice_count > MAX_SLICES:
                    # Center around current slice
                    half_window = MAX_SLICES // 2
                    start_pos = max(0, current_frame_pos - half_window)
                    end_pos = min(slice_count, start_pos + MAX_SLICES)
                    
                    # Adjust if we're at the edge
                    if end_pos - start_pos < MAX_SLICES:
                        start_pos = max(0, end_pos - MAX_SLICES)
                        
                    # Use position indices, not actual slice indices
                    demo_slices = list(range(start_pos, end_pos))
                else:
                    demo_slices = list(range(slice_count))
                    
                print(f"Using {len(demo_slices)} slices for demo mode propagation, centered around position {current_frame_pos}")
                print(f"Demo slices positions (first 5): {demo_slices[:5]}")
                    
                # Initialize video predictor with position indices
                self.queue.put(("status", "Initializing video predictor..."))
                self.queue.put(("progress", 20))
                
                try:
                    # In demo mode, we're using position indices
                    self.predictor.demo_mode = True  # Force demo mode
                    self.predictor.init_video_predictor(demo_slices)
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    self.queue.put(("progress", 0))
                    return
                
                # Add points to current frame
                # Find the index of the current slice in our demo slices
                try:
                    # Current frame position relative to demo slices
                    if current_frame_pos in demo_slices:
                        # Use list.index() to find the position in demo_slices
                        relative_pos = demo_slices.index(current_frame_pos)
                        print(f"Current position {current_frame_pos} found at index {relative_pos} in demo_slices")
                    else:
                        # Find closest position manually
                        closest_pos = None
                        min_diff = float('inf')
                        for i, pos in enumerate(demo_slices):
                            diff = abs(pos - current_frame_pos)
                            if diff < min_diff:
                                min_diff = diff
                                closest_pos = pos
                                relative_pos = i
                        
                        current_frame_pos = relative_pos  # Use position within demo_slices
                        self.queue.put(("status", f"Using closest position {closest_pos} instead of {current_frame_idx}"))
                        print(f"Using closest position {closest_pos} at index {relative_pos} in demo_slices")
                except Exception as e:
                    self.queue.put(("error", f"Error finding frame position: {str(e)}"))
                    self.queue.put(("progress", 0))
                    return
            else:
                # Initialize video predictor - for non-demo mode we still use position indices
                self.queue.put(("status", "Initializing video predictor..."))
                self.queue.put(("progress", 20))
                
                # Limit to a reasonable number of slices to prevent memory issues
                MAX_SLICES = 100
                
                try:
                    self.predictor.init_video_predictor(slices, max_slices=MAX_SLICES)
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    print("Falling back to demo mode for propagation")
                    
                    # Retry with demo mode
                    try:
                        self.predictor.demo_mode = True
                        self.predictor.init_video_predictor(slices[:MAX_SLICES])
                        use_demo_mode = True
                    except Exception as retry_e:
                        self.queue.put(("error", f"Also failed with demo mode: {str(retry_e)}"))
                        self.queue.put(("progress", 0))
                        return
                
                # Find frame position in the selected slices
                if not use_demo_mode and hasattr(self.predictor, 'slice_to_frame_map'):
                    if current_frame_idx in self.predictor.slice_to_frame_map:
                        current_frame_pos = self.predictor.slice_to_frame_map[current_frame_idx]
                    else:
                        # If the current slice wasn't included in the processing, find the closest one
                        self.queue.put(("error", f"Current slice {current_frame_idx} not included in processing. Try a different slice."))
                        self.queue.put(("progress", 0))
                        return
            
            obj_id = self.current_object_id.get()
            
            self.queue.put(("status", f"Adding points to frame {current_frame_idx} (position {current_frame_pos})..."))
            self.queue.put(("progress", 30))
            
            try:
                self.predictor.add_point_to_video(
                    current_frame_pos,  # Use position in frame sequence
                    obj_id,
                    self.points,
                    self.point_labels
                )
            except Exception as e:
                self.queue.put(("error", f"Error adding points to video: {str(e)}"))
                self.queue.put(("progress", 0))
                return
            
            # Propagate forward
            self.queue.put(("status", "Propagating masks forward..."))
            self.queue.put(("progress", 50))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    reverse=False
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks forward: {str(e)}"))
                # Continue with backward propagation even if forward fails
            
            # Propagate backward
            self.queue.put(("status", "Propagating masks backward..."))
            self.queue.put(("progress", 80))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    reverse=True
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks backward: {str(e)}"))
                # Continue to get mask if possible
            
            # Get the result for current frame
            self.queue.put(("status", "Getting mask for current frame..."))
            self.queue.put(("progress", 90))
            
            try:
                mask = self.predictor.get_mask_for_frame(
                    current_frame_pos,  # Use position in sequence 
                    obj_id
                )
                
                # Display the result
                self.queue.put(("display_mask", mask))
                self.queue.put(("status", "Propagation complete."))
                self.queue.put(("progress", 100))
            except Exception as e:
                self.queue.put(("error", f"Error getting mask for current frame: {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error propagating masks: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
    
    def display_mask(self, mask):
        """Display the generated mask as overlay"""
        # Clear the axis
        self.ax.clear()
        
        # Display the slice
        vmin, vmax = np.percentile(self.current_slice, [5, 95])
        self.ax.imshow(self.current_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Create mask overlay with transparency
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask > 0] = [1, 0, 0, 0.5]  # Red with alpha
        
        # Display mask overlay
        self.ax.imshow(mask_overlay, aspect='auto')
        
        # Add points
        fg_points = [p for i, p in enumerate(self.points) if self.point_labels[i] == 1]
        if fg_points:
            fg_points = np.array(fg_points)
            self.ax.scatter(fg_points[:, 0], fg_points[:, 1], color='red', marker='o', s=30)
            
        bg_points = [p for i, p in enumerate(self.points) if self.point_labels[i] == 0]
        if bg_points:
            bg_points = np.array(bg_points)
            self.ax.scatter(bg_points[:, 0], bg_points[:, 1], color='blue', marker='o', s=30)
        
        # Update title
        self.ax.set_title(f"{self.current_slice_type.get().capitalize()} {self.current_slice_idx.get()} with Mask")
        
        # Redraw canvas
        self.canvas.draw()
    
    def process_queue(self):
        """Process messages from worker threads"""
        try:
            while True:
                msg = self.queue.get_nowait()
                
                cmd = msg[0]
                
                if cmd == "status":
                    self.status_text.set(msg[1])
                elif cmd == "progress":
                    self.progress_var.set(msg[1])
                elif cmd == "error":
                    messagebox.showerror("Error", msg[1])
                    self.status_text.set("Error occurred.")
                    self.progress_var.set(0)
                elif cmd == "update_scale":
                    self.slice_scale.configure(to=msg[1])
                    self._load_current_slice()
                elif cmd == "display_mask":
                    self.display_mask(msg[1])
                    
                self.queue.task_done()
        except queue.Empty:
            pass
            
        # Schedule next queue check
        self.master.after(100, self.process_queue)
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Seismic Interpretation with SAM2\n\n"
            "An application for seismic interpretation using Segment Anything Model 2 (SAM2).\n\n"
            "© 2025"
        )
    
    def _show_instructions(self):
        """Show instructions dialog"""
        messagebox.showinfo(
            "Instructions",
            "How to use this application:\n\n"
            "1. Load a SEGY file using File > Open SEGY.\n"
            "2. Select a slice type (Inline, Crossline, or Time/Depth).\n"
            "3. Navigate through slices using the slider.\n"
            "4. Add annotation points by clicking on the image. Use the Foreground/Background options to specify point types.\n"
            "5. Click 'Generate Mask' to create a segmentation using SAM2.\n"
            "6. Use 'Propagate' to extend the segmentation to all slices.\n\n"
            "Tips:\n"
            "- Red points indicate foreground (the feature you want to segment).\n"
            "- Blue points indicate background (areas to exclude).\n"
            "- You can use multiple object IDs to segment different features."
        )
    
    def _on_close(self):
        """Handle window close event"""
        # Clean up resources
        if hasattr(self, 'segy_loader'):
            self.segy_loader.close()
            
        self.master.destroy()

    def _open_3d_visualization(self):
        """Open a 3D visualization window using PyVista"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            messagebox.showinfo("Info", "Please load a SEGY file first.")
            return
            
        if not PYVISTA_AVAILABLE:
            messagebox.showerror("Error", "PyVista is not available. Please install it with: pip install pyvista")
            return
            
        # Warn user about potential memory issues
        if np.prod(self.segy_loader.data.shape) > 100000000:  # If volume is larger than ~100M voxels
            if not messagebox.askyesno("Warning", 
                                      "The seismic volume is very large and may cause memory issues. "
                                      "Continue with visualization?"):
                return
        
        # Ask how many slices to visualize
        try:
            from tkinter.simpledialog import askinteger
            num_slices = askinteger("Input", "Enter number of slices to visualize (5-100):", 
                                   initialvalue=20, minvalue=5, maxvalue=100)
            if num_slices is None:  # User canceled
                num_slices = 20  # Default
        except:
            num_slices = 20  # Default if dialog fails
        
        # Check if we have masks at all
        has_masks = False
        if hasattr(self.predictor, 'inference_state') and self.predictor.inference_state is not None:
            has_masks = len(self.predictor.inference_state["output"]) > 0
        elif hasattr(self.predictor, 'demo_inference_state') and self.predictor.demo_inference_state is not None:
            has_masks = len(self.predictor.demo_inference_state["output"]) > 0
        
        # If we have masks but they're not propagated to enough slices, ask to propagate
        if has_masks:
            slice_type = self.current_slice_type.get()
            
            # Check if we need to propagate first
            if slice_type == "inline":
                total_slices = len(self.segy_loader.inlines)
            elif slice_type == "crossline":
                total_slices = len(self.segy_loader.crosslines)
            else:  # timeslice
                total_slices = len(self.segy_loader.timeslices)
            
            # Count how many masks we have
            mask_count = 0
            obj_id = self.current_object_id.get()
            
            # Sample a few slices to see if we have masks
            sample_indices = np.linspace(0, total_slices-1, min(10, total_slices), dtype=int)
            for idx in sample_indices:
                try:
                    mask = self.predictor.get_mask_for_frame(idx, obj_id)
                    if mask is not None and np.any(mask):
                        mask_count += 1
                except:
                    pass
            
            # If less than 70% of sampled slices have masks, offer to propagate
            if mask_count < 0.7 * len(sample_indices):
                if messagebox.askyesno("Propagation Needed", 
                                     "Masks have not been fully propagated to all slices. "
                                     "Propagate masks before visualization? (Recommended)"):
                    # Run propagation with a wider range to cover all visualization slices
                    self._propagate_all_for_visualization(num_slices)
                    # Return early - the 3D visualization will be triggered after propagation completes
                    return
                
        # Start visualization in a separate thread to avoid blocking the main GUI
        self.status_text.set("Preparing 3D visualization...")
        self.progress_var.set(10)
        
        threading.Thread(target=lambda: self._create_3d_visualization(num_slices), daemon=True).start()
    
    def _propagate_all_for_visualization(self, num_slices=100):
        """Propagate masks to all slices needed for visualization"""
        # Start propagation in a thread
        self.status_text.set("Propagating masks for visualization...")
        self.progress_var.set(5)
        
        # Run with increased slice coverage
        threading.Thread(target=lambda: self._propagate_thread_for_viz(num_slices), daemon=True).start()
    
    def _propagate_thread_for_viz(self, num_slices=100):
        """Thread function for mask propagation with visualization followup"""
        try:
            slice_type = self.current_slice_type.get()
            
            # Determine slice range
            if slice_type == "inline":
                slice_count = len(self.segy_loader.inlines)
                # Use position indices (0 to len-1) instead of actual inline numbers
                slices = list(range(slice_count))
                # For display purposes only
                actual_indices = list(self.segy_loader.inlines) if hasattr(self.segy_loader.inlines, '__iter__') else []
            elif slice_type == "crossline":
                slice_count = len(self.segy_loader.crosslines)
                slices = list(range(slice_count))
                actual_indices = list(self.segy_loader.crosslines) if hasattr(self.segy_loader.crosslines, '__iter__') else []
            else:  # timeslice
                slice_count = len(self.segy_loader.timeslices)
                slices = list(range(slice_count))
                actual_indices = list(range(slice_count))
                
            # Get current slice
            current_frame_idx = self.current_slice_idx.get()
            current_frame_pos = current_frame_idx  # We're using position indices directly
            
            self.queue.put(("status", f"Preparing to propagate masks from {slice_type} {current_frame_idx}..."))
            self.queue.put(("progress", 15))
            
            print(f"Slice type: {slice_type}, Current idx: {current_frame_idx}, Position: {current_frame_pos}")
            
            # Force demo mode if video predictor failed to initialize properly
            use_demo_mode = self.predictor.demo_mode
            
            # Check if we have a real video predictor or if we need to fall back to demo
            if not use_demo_mode and not hasattr(self.predictor, 'video_predictor'):
                print("Video predictor not initialized, falling back to demo mode")
                use_demo_mode = True
                
            # Check if video predictor is a mock (has no __module__ attribute or it's not 'sam2.sam2_video_predictor')
            if not use_demo_mode and (not hasattr(self.predictor.video_predictor, '__module__') or 
                                     'sam2.sam2_video_predictor' not in self.predictor.video_predictor.__module__):
                print("Video predictor is a mock, falling back to demo mode")
                use_demo_mode = True
                
            print(f"Using {'demo' if use_demo_mode else 'real'} mode for propagation")
            
            # In demo mode, use position indices (0 to len-1) to avoid out of range errors
            if use_demo_mode:
                # Store a subset of position indices for demo mode
                # For visualization, ensure we cover the full range needed
                MAX_SLICES = max(200, num_slices * 2)  # Ensure wide coverage for visualization
                if slice_count > MAX_SLICES:
                    # Center around current slice
                    half_window = MAX_SLICES // 2
                    start_pos = max(0, current_frame_pos - half_window)
                    end_pos = min(slice_count, start_pos + MAX_SLICES)
                    
                    # Adjust if we're at the edge
                    if end_pos - start_pos < MAX_SLICES:
                        start_pos = max(0, end_pos - MAX_SLICES)
                        
                    # Use position indices, not actual slice indices
                    demo_slices = list(range(start_pos, end_pos))
                else:
                    demo_slices = list(range(slice_count))
                    
                print(f"Using {len(demo_slices)} slices for demo mode propagation, centered around position {current_frame_pos}")
                print(f"Demo slices positions (first 5): {demo_slices[:5]}")
                    
                # Initialize video predictor with position indices
                self.queue.put(("status", "Initializing video predictor..."))
                self.queue.put(("progress", 20))
                
                try:
                    # In demo mode, we're using position indices
                    self.predictor.demo_mode = True  # Force demo mode
                    self.predictor.init_video_predictor(demo_slices)
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    self.queue.put(("progress", 0))
                    return
                
                # Add points to current frame
                # Find the index of the current slice in our demo slices
                try:
                    # Current frame position relative to demo slices
                    if current_frame_pos in demo_slices:
                        # Use list.index() to find the position in demo_slices
                        relative_pos = demo_slices.index(current_frame_pos)
                        print(f"Current position {current_frame_pos} found at index {relative_pos} in demo_slices")
                    else:
                        # Find closest position manually
                        closest_pos = None
                        min_diff = float('inf')
                        for i, pos in enumerate(demo_slices):
                            diff = abs(pos - current_frame_pos)
                            if diff < min_diff:
                                min_diff = diff
                                closest_pos = pos
                                relative_pos = i
                        
                        current_frame_pos = relative_pos  # Use position within demo_slices
                        self.queue.put(("status", f"Using closest position {closest_pos} instead of {current_frame_idx}"))
                        print(f"Using closest position {closest_pos} at index {relative_pos} in demo_slices")
                except Exception as e:
                    self.queue.put(("error", f"Error finding frame position: {str(e)}"))
                    self.queue.put(("progress", 0))
                    return
            else:
                # Initialize video predictor - for non-demo mode we still use position indices
                self.queue.put(("status", "Initializing video predictor..."))
                self.queue.put(("progress", 20))
                
                # Limit to a reasonable number of slices to prevent memory issues
                MAX_SLICES = max(200, num_slices * 2)  # Ensure wide coverage for visualization
                
                try:
                    self.predictor.init_video_predictor(slices, max_slices=MAX_SLICES)
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    print("Falling back to demo mode for propagation")
                    
                    # Retry with demo mode
                    try:
                        self.predictor.demo_mode = True
                        self.predictor.init_video_predictor(slices[:MAX_SLICES])
                        use_demo_mode = True
                    except Exception as retry_e:
                        self.queue.put(("error", f"Also failed with demo mode: {str(retry_e)}"))
                        self.queue.put(("progress", 0))
                        return
                
                # Find frame position in the selected slices
                if not use_demo_mode and hasattr(self.predictor, 'slice_to_frame_map'):
                    if current_frame_idx in self.predictor.slice_to_frame_map:
                        current_frame_pos = self.predictor.slice_to_frame_map[current_frame_idx]
                    else:
                        # If the current slice wasn't included in the processing, find the closest one
                        self.queue.put(("error", f"Current slice {current_frame_idx} not included in processing. Try a different slice."))
                        self.queue.put(("progress", 0))
                        return
            
            obj_id = self.current_object_id.get()
            
            self.queue.put(("status", f"Adding points to frame {current_frame_idx} (position {current_frame_pos})..."))
            self.queue.put(("progress", 30))
            
            try:
                self.predictor.add_point_to_video(
                    current_frame_pos,  # Use position in frame sequence
                    obj_id,
                    self.points,
                    self.point_labels
                )
            except Exception as e:
                self.queue.put(("error", f"Error adding points to video: {str(e)}"))
                self.queue.put(("progress", 0))
                return
            
            # Propagate forward
            self.queue.put(("status", "Propagating masks forward..."))
            self.queue.put(("progress", 50))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    reverse=False
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks forward: {str(e)}"))
                # Continue with backward propagation even if forward fails
            
            # Propagate backward
            self.queue.put(("status", "Propagating masks backward..."))
            self.queue.put(("progress", 80))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    reverse=True
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks backward: {str(e)}"))
                # Continue to get mask if possible
            
            # Get the result for current frame
            self.queue.put(("status", "Getting mask for current frame..."))
            self.queue.put(("progress", 90))
            
            try:
                mask = self.predictor.get_mask_for_frame(
                    current_frame_pos,  # Use position in sequence 
                    obj_id
                )
                
                # Display the result
                self.queue.put(("display_mask", mask))
                self.queue.put(("status", "Propagation complete. Starting 3D visualization..."))
                self.queue.put(("progress", 95))
                
                # Now start the 3D visualization
                threading.Thread(target=lambda: self._create_3d_visualization(num_slices), daemon=True).start()
                
            except Exception as e:
                self.queue.put(("error", f"Error getting mask for current frame: {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error propagating masks: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))

    def _create_3d_visualization(self, num_slices=20):
        """Create a very simple 3D visualization of seismic data and masks using PyVista"""
        try:
            self.queue.put(("status", f"Creating ultra-simplified 3D visualization with {num_slices} slices..."))
            self.queue.put(("progress", 30))
            
            # Get the seismic data
            seismic_volume = self.segy_loader.data
            
            # Get dimensions
            ni, nj, nk = seismic_volume.shape
            
            # Current slice type
            slice_type = self.current_slice_type.get()
            
            # Get current object ID for masks
            obj_id = self.current_object_id.get()
            
            # Create a PyVista plotter
            plotter = pv.Plotter(notebook=False)
            plotter.set_background("white")
            
            # Calculate downsampling factors to reduce size
            # Target dimensions less than 200 in each direction
            ds_i = max(1, ni // 150)
            ds_j = max(1, nj // 150)
            ds_k = max(1, nk // 150)
            
            # Report downsampling
            self.queue.put(("status", f"Downsampling factors: {ds_i}x{ds_j}x{ds_k}"))
            
            # Determine which slices to show
            if slice_type == "inline":
                indices = np.linspace(0, ni-1, num_slices, dtype=int)
                self.queue.put(("status", f"Showing {num_slices} inline slices: {indices[:5]}..."))
            elif slice_type == "crossline":
                indices = np.linspace(0, nj-1, num_slices, dtype=int)
                self.queue.put(("status", f"Showing {num_slices} crossline slices: {indices[:5]}..."))
            else:  # timeslice
                indices = np.linspace(0, nk-1, num_slices, dtype=int)
                self.queue.put(("status", f"Showing {num_slices} time slices: {indices[:5]}..."))
                
            # Create a simple display for each slice
            for i, idx in enumerate(indices):
                try:
                    self.queue.put(("status", f"Processing slice {i+1}/{num_slices}: {idx}"))
                    self.queue.put(("progress", 30 + int(60 * i / num_slices)))
                    
                    # Process differently based on slice type
                    if slice_type == "inline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[idx, ::ds_j, ::ds_k]
                        
                        # Get dimensions after downsampling
                        s_nj, s_nk = slice_data.shape
                        
                        # Create a simple surface at the correct position
                        # We use PolyData instead of StructuredGrid for simpler memory usage
                        x_coords = np.ones(s_nj * s_nk) * idx
                        y_coords = np.repeat(np.arange(0, s_nj * ds_j, ds_j), s_nk)
                        z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_nj) / 8  # Scale Z
                        
                        # Create points array
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        
                        # Create mesh using pyvista points_to_poly_data
                        grid = pv.PolyData(points)
                        
                        # Set scalar data for colormapping (normalized)
                        vmin, vmax = np.percentile(slice_data, [5, 95])
                        norm_data = np.clip(slice_data, vmin, vmax)
                        norm_data = (norm_data - vmin) / (vmax - vmin)
                        grid.point_data["intensity"] = norm_data.flatten()
                        
                        # Create a delimiter for this slice
                        delimiter = pv.Plane(center=(idx, s_nj//2, s_nk//2), 
                                            direction=(1, 0, 0), 
                                            i_size=1, j_size=s_nj*ds_j)
                        
                        # Add mesh to plotter with seismic colormap
                        plotter.add_mesh(grid, cmap="seismic", point_size=3, render_points_as_spheres=True)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                        
                        # Try to add mask if available
                        try:
                            mask = self.predictor.get_mask_for_frame(idx, obj_id)
                            if mask is not None and np.any(mask):
                                # Transpose and downsample
                                mask_t = mask.T
                                if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                    mask_ds = mask_t[::ds_j, ::ds_k]
                                    if mask_ds.shape == slice_data.shape:
                                        # Create mask points - only where mask is True
                                        mask_points = []
                                        for j in range(s_nj):
                                            for k in range(s_nk):
                                                if mask_ds[j, k]:
                                                    mask_points.append([idx, j*ds_j, k*ds_k/8])
                                        
                                        if mask_points:
                                            # Create mesh for mask points
                                            mask_poly = pv.PolyData(np.array(mask_points))
                                            plotter.add_mesh(mask_poly, color="blue", point_size=5, 
                                                            render_points_as_spheres=True)
                        except:
                            pass
                            
                    elif slice_type == "crossline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, idx, ::ds_k]
                        
                        # Get dimensions after downsampling
                        s_ni, s_nk = slice_data.shape
                        
                        # Create a simple point cloud at the correct position
                        x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nk)
                        y_coords = np.ones(s_ni * s_nk) * idx
                        z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_ni) / 8  # Scale Z
                        
                        # Create points array
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        
                        # Create mesh using pyvista points
                        grid = pv.PolyData(points)
                        
                        # Set scalar data for colormapping (normalized)
                        vmin, vmax = np.percentile(slice_data, [5, 95])
                        norm_data = np.clip(slice_data, vmin, vmax)
                        norm_data = (norm_data - vmin) / (vmax - vmin)
                        grid.point_data["intensity"] = norm_data.flatten()
                        
                        # Create a delimiter for this slice
                        delimiter = pv.Plane(center=(s_ni//2, idx, s_nk//2), 
                                            direction=(0, 1, 0), 
                                            i_size=s_ni*ds_i, j_size=1)
                        
                        # Add mesh to plotter with seismic colormap
                        plotter.add_mesh(grid, cmap="seismic", point_size=3, render_points_as_spheres=True)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                        
                        # Try to add mask if available
                        try:
                            mask = self.predictor.get_mask_for_frame(idx, obj_id)
                            if mask is not None and np.any(mask):
                                # Transpose and downsample
                                mask_t = mask.T
                                if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                    mask_ds = mask_t[::ds_i, ::ds_k]
                                    if mask_ds.shape == slice_data.shape:
                                        # Create mask points - only where mask is True
                                        mask_points = []
                                        for i in range(s_ni):
                                            for k in range(s_nk):
                                                if mask_ds[i, k]:
                                                    mask_points.append([i*ds_i, idx, k*ds_k/8])
                                        
                                        if mask_points:
                                            # Create mesh for mask points
                                            mask_poly = pv.PolyData(np.array(mask_points))
                                            plotter.add_mesh(mask_poly, color="blue", point_size=5, 
                                                            render_points_as_spheres=True)
                        except:
                            pass
                    
                    else:  # timeslice
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, ::ds_j, idx]
                        
                        # Get dimensions after downsampling
                        s_ni, s_nj = slice_data.shape
                        
                        # Create a simple point cloud at the correct position
                        x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nj)
                        y_coords = np.tile(np.arange(0, s_nj * ds_j, ds_j), s_ni)
                        z_coords = np.ones(s_ni * s_nj) * idx / 8  # Scale Z
                        
                        # Create points array
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        
                        # Create mesh using pyvista points
                        grid = pv.PolyData(points)
                        
                        # Set scalar data for colormapping (normalized)
                        vmin, vmax = np.percentile(slice_data, [5, 95])
                        norm_data = np.clip(slice_data, vmin, vmax)
                        norm_data = (norm_data - vmin) / (vmax - vmin)
                        grid.point_data["intensity"] = norm_data.flatten()
                        
                        # Create a delimiter for this slice
                        delimiter = pv.Plane(center=(s_ni//2, s_nj//2, idx/8), 
                                            direction=(0, 0, 1), 
                                            i_size=s_ni*ds_i, j_size=s_nj*ds_j)
                        
                        # Add mesh to plotter with seismic colormap
                        plotter.add_mesh(grid, cmap="seismic", point_size=3, render_points_as_spheres=True)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                        
                        # Try to add mask if available
                        try:
                            mask = self.predictor.get_mask_for_frame(idx, obj_id)
                            if mask is not None and np.any(mask):
                                if mask.shape[0] > 1 and mask.shape[1] > 1:
                                    mask_ds = mask[::ds_i, ::ds_j]
                                    if mask_ds.shape == slice_data.shape:
                                        # Create mask points - only where mask is True
                                        mask_points = []
                                        for i in range(s_ni):
                                            for j in range(s_nj):
                                                if mask_ds[i, j]:
                                                    mask_points.append([i*ds_i, j*ds_j, idx/8])
                                        
                                        if mask_points:
                                            # Create mesh for mask points
                                            mask_poly = pv.PolyData(np.array(mask_points))
                                            plotter.add_mesh(mask_poly, color="blue", point_size=5, 
                                                            render_points_as_spheres=True)
                        except:
                            pass
                
                except Exception as slice_err:
                    self.queue.put(("status", f"Error processing slice {idx}: {slice_err}"))
                    # Continue with next slice
            
            # Add axes for reference and finalize
            try:
                plotter.add_axes()
                plotter.show_grid()
            except:
                pass
            
            # Update progress
            self.queue.put(("status", "Showing 3D visualization..."))
            self.queue.put(("progress", 90))
            
            # Try to show the window with fallbacks
            try:
                # First try interactive viewing
                plotter.show(title=f"Seismic {slice_type} Slices with Masks")
                self.queue.put(("status", "3D visualization complete."))
                self.queue.put(("progress", 100))
                
            except Exception as e1:
                self.queue.put(("status", f"Interactive viewing failed: {e1}, trying off_screen..."))
                
                try:
                    # Try screenshot rendering as fallback
                    plotter = pv.Plotter(off_screen=True)
                    plotter.set_background("white")
                    
                    # Create a simple cube as placeholder
                    mesh = pv.Cube()
                    plotter.add_mesh(mesh, cmap="seismic")
                    
                    # Save to temporary file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    plotter.screenshot(temp_file.name)
                    
                    # Show the image in a simple tkinter window
                    self.queue.put(("status", "Opening image in separate window..."))
                    
                    # Create function to show image in tkinter
                    def show_image(filename):
                        import tkinter as tk
                        from PIL import Image, ImageTk
                        
                        img_window = tk.Toplevel()
                        img_window.title("3D Visualization (Static Image)")
                        
                        # Load the image
                        img = Image.open(filename)
                        tk_img = ImageTk.PhotoImage(img)
                        
                        # Show the image
                        label = tk.Label(img_window, image=tk_img)
                        label.image = tk_img  # Keep a reference
                        label.pack()
                        
                        # Add a close button
                        tk.Button(img_window, text="Close", command=img_window.destroy).pack()
                    
                    # Schedule showing the image in the main thread
                    self.master.after(100, lambda: show_image(temp_file.name))
                    
                    self.queue.put(("status", "Showing static image."))
                    self.queue.put(("progress", 100))
                    
                except Exception as e2:
                    self.queue.put(("error", f"All visualization methods failed. Error: {e2}"))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error creating 3D visualization: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))

    def _open_3d_surface_generation(self):
        """Open a 3D surface generation and visualization window using LoopStructural or triangulation"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            messagebox.showinfo("Info", "Please load a SEGY file first.")
            return
            
        if not PYVISTA_AVAILABLE:
            messagebox.showerror("Error", "PyVista is not available. Please install it with: pip install pyvista")
            return
            
        # Check if we have any masks
        has_masks = False
        if hasattr(self.predictor, 'inference_state') and self.predictor.inference_state is not None:
            has_masks = len(self.predictor.inference_state["output"]) > 0
        elif hasattr(self.predictor, 'demo_inference_state') and self.predictor.demo_inference_state is not None:
            has_masks = len(self.predictor.demo_inference_state["output"]) > 0
            
        if not has_masks:
            if not messagebox.askyesno("No Masks Detected", 
                                     "No masks have been generated. You need to create and propagate masks before generating surfaces. Do you want to generate a surface anyway?"):
                return
        
        # Ask the user for the surface generation method
        use_triangulation = True
        if LOOPSTRUCTURAL_AVAILABLE:
            if messagebox.askyesno("Surface Generation Method", 
                               "LoopStructural is available. Would you like to use it for advanced surface generation?\n\n"
                               "Yes: Use LoopStructural (better for sparse data)\n"
                               "No: Use simple triangulation (more reliable but less smooth)"):
                use_triangulation = False
        
        # Ask for surface generation resolution
        try:
            from tkinter.simpledialog import askinteger
            num_slices = askinteger("Input", "Number of slices to sample for surface (5-100):", 
                                   initialvalue=20, minvalue=5, maxvalue=100)
            if num_slices is None:
                num_slices = 20  # Default if canceled
                
            if use_triangulation:
                # For triangulation, ask about point reduction
                max_points = askinteger("Input", "Maximum number of points to use (100-5000):", 
                                      initialvalue=1000, minvalue=100, maxvalue=5000)
                if max_points is None:
                    max_points = 1000  # Default if canceled
                smoothing = 0  # Not used for triangulation
            else:
                # For LoopStructural, ask about smoothing
                smoothing = askinteger("Input", "Surface smoothing factor (1-50):", 
                                    initialvalue=10, minvalue=1, maxvalue=50)
                if smoothing is None:
                    smoothing = 10  # Default if canceled
                max_points = 500  # Default for LoopStructural
        except:
            num_slices = 20
            max_points = 1000
            smoothing = 10
        
        # Start the surface generation in a separate thread
        self.status_text.set("Preparing surface generation...")
        self.progress_var.set(10)
        
        threading.Thread(target=lambda: self._generate_3d_surface(num_slices, smoothing, use_triangulation, max_points), 
                        daemon=True).start()

    def _generate_3d_surface(self, num_slices=20, smoothing=10, use_triangulation=False, max_points=1000):
        """Generate a 3D surface from mask points using LoopStructural or triangulation"""
        try:
            self.queue.put(("status", f"Generating 3D surface from {num_slices} slices..."))
            self.queue.put(("progress", 20))
            
            # Get the seismic data dimensions
            seismic_volume = self.segy_loader.data
            ni, nj, nk = seismic_volume.shape
            
            # Current slice type and object ID
            slice_type = self.current_slice_type.get()
            obj_id = self.current_object_id.get()
            
            # Determine slices to sample based on slice type
            if slice_type == "inline":
                indices = np.linspace(0, ni-1, num_slices, dtype=int)
                self.queue.put(("status", f"Sampling {num_slices} inline slices..."))
            elif slice_type == "crossline":
                indices = np.linspace(0, nj-1, num_slices, dtype=int)
                self.queue.put(("status", f"Sampling {num_slices} crossline slices..."))
            else:  # timeslice
                indices = np.linspace(0, nk-1, num_slices, dtype=int)
                self.queue.put(("status", f"Sampling {num_slices} time slices..."))
            
            # Check if we need to propagate masks first
            self._ensure_masks_propagated(indices, obj_id)
            
            # Collect all mask points for surface generation
            all_points = []
            
            # Process slices to collect points where masks exist
            for i, idx in enumerate(indices):
                self.queue.put(("status", f"Processing slice {i+1}/{num_slices}: {idx}"))
                self.queue.put(("progress", 20 + int(30 * i / num_slices)))
                
                try:
                    # Get mask for this slice
                    mask = self.predictor.get_mask_for_frame(idx, obj_id)
                    
                    if mask is not None and np.any(mask):
                        # Process points based on slice type
                        if slice_type == "inline":
                            # For inline slices, x is fixed at idx, y and z vary
                            mask_t = mask.T  # Transpose the mask
                            ys, zs = np.where(mask_t)
                            for y, z in zip(ys, zs):
                                all_points.append([idx, y, z/8])  # Scale z as in visualization
                                
                        elif slice_type == "crossline":
                            # For crossline slices, y is fixed at idx, x and z vary
                            mask_t = mask.T  # Transpose the mask
                            xs, zs = np.where(mask_t)
                            for x, z in zip(xs, zs):
                                all_points.append([x, idx, z/8])  # Scale z as in visualization
                                
                        else:  # timeslice
                            # For time slices, z is fixed at idx, x and y vary
                            xs, ys = np.where(mask)
                            for x, y in zip(xs, ys):
                                all_points.append([x, y, idx/8])  # Scale z as in visualization
                
                except Exception as e:
                    print(f"Error processing mask for frame {idx}: {e}")
            
            # Check if we have enough points for surface generation
            if len(all_points) < 10:
                self.queue.put(("error", "Not enough mask points found. Please generate more masks or try a different object ID."))
                return
                
            self.queue.put(("status", f"Collected {len(all_points)} points for surface generation"))
            self.queue.put(("progress", 50))
            
            # Convert points to numpy array
            points_array = np.array(all_points)
            
            # Create a simple PyVista plotter for basic visualization
            plotter = pv.Plotter(notebook=False)
            plotter.set_background("white")
            
            # Add seismic data slices to context
            self.queue.put(("status", "Adding seismic slices for context..."))
            self._add_seismic_slices_to_plot(plotter, slice_type, seismic_volume, indices)
            
            # Add scatter plot of the points
            point_cloud = pv.PolyData(points_array)
            plotter.add_mesh(point_cloud, color="blue", point_size=5, render_points_as_spheres=True)
            
            # Thin the points if needed to improve performance and reduce complexity
            if len(points_array) > max_points:
                self.queue.put(("status", f"Reducing points from {len(points_array)} to {max_points} for performance..."))
                indices = np.random.choice(len(points_array), max_points, replace=False)
                subset_points = points_array[indices]
            else:
                subset_points = points_array
                
            if use_triangulation:
                self.queue.put(("status", "Creating surface using triangulation..."))
                self.queue.put(("progress", 70))
                self._create_triangulation_surface(subset_points, plotter)
            else:
                self.queue.put(("status", "Creating surface using LoopStructural..."))
                self.queue.put(("progress", 60))
                # This might fail if LoopStructural isn't available
                try:
                    self._create_loopstructural_surface(subset_points, plotter, smoothing, ni, nj, nk)
                except ImportError:
                    # Fall back to triangulation
                    self.queue.put(("status", "LoopStructural failed, falling back to triangulation..."))
                    self._create_triangulation_surface(subset_points, plotter)
                    
            # Add axes and show plot
            plotter.add_axes()
            plotter.show_grid()
            
            # Show the plotter
            self.queue.put(("status", "Displaying 3D surface..."))
            self.queue.put(("progress", 95))
            
            try:
                plotter.show(title="3D Surface from Masks")
                self.queue.put(("status", "3D surface visualization complete"))
                self.queue.put(("progress", 100))
            except Exception as e:
                self.queue.put(("error", f"Error displaying 3D surface: {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error in surface generation: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
        
    def _create_triangulation_surface(self, points, plotter):
        """Create a surface using simple triangulation"""
        try:
            # Import scipy for Delaunay triangulation
            from scipy.spatial import Delaunay
            
            self.queue.put(("status", "Preparing points for triangulation..."))
            
            # Special handling for points that lie in a plane
            # Check dimensionality of the points
            unique_x = np.unique(points[:, 0])
            unique_y = np.unique(points[:, 1])
            unique_z = np.unique(points[:, 2])
            
            # If all points have the same value in any dimension, we need special handling
            is_planar = len(unique_x) == 1 or len(unique_y) == 1 or len(unique_z) == 1
            
            if is_planar:
                self.queue.put(("status", "Detected planar data, using 2D triangulation..."))
                
                # Determine which dimension is constant
                if len(unique_x) == 1:
                    # All points have same x, use y-z for triangulation
                    points_2d = points[:, 1:3]
                    const_dim = 0
                elif len(unique_y) == 1:
                    # All points have same y, use x-z for triangulation
                    points_2d = np.column_stack([points[:, 0], points[:, 2]])
                    const_dim = 1
                else:
                    # All points have same z, use x-y for triangulation
                    points_2d = points[:, 0:2]
                    const_dim = 2
                    
                # Create a 2D triangulation
                try:
                    tri = Delaunay(points_2d)
                    faces = tri.simplices
                    
                    # Create a PolyData object for the triangulated mesh
                    mesh = pv.PolyData(points, faces=np.column_stack([np.full(len(faces), 3), faces]))
                    
                    # Smooth the mesh if it has enough points
                    if len(points) > 50:
                        mesh = mesh.smooth(n_iter=100, relaxation_factor=0.1)
                        
                    plotter.add_mesh(mesh, color="red", opacity=0.5)
                    self.queue.put(("status", "Created triangulated surface (planar)"))
                    
                except Exception as e:
                    self.queue.put(("error", f"Error in 2D triangulation: {str(e)}"))
                    self.queue.put(("status", "Unable to triangulate points, showing point cloud only"))
                    
            else:
                # Use PyVista's Delaunay 3D for normal point clouds
                self.queue.put(("status", "Creating 3D triangulation..."))
                
                try:
                    # Create the triangulation using PyVista's implementation
                    surf = None
                    
                    try:
                        # First try PyVista's delaunay_3d
                        cloud = pv.PolyData(points)
                        surf = cloud.delaunay_3d()
                        # Extract the surface
                        surf = surf.extract_surface()
                    except Exception as e:
                        print(f"PyVista Delaunay3D failed: {e}, trying alpha shapes")
                        
                        # If that fails, try alpha shapes which can be more robust
                        try:
                            # Alpha shape approaches create better surfaces for irregular point clouds
                            cloud = pv.PolyData(points)
                            # Generate a surface using an appropriate alpha value (smaller = tighter fit)
                            alpha = 5.0  # This is a relative value, can be adjusted based on point density
                            surf = cloud.delaunay_3d(alpha=alpha)
                            surf = surf.extract_surface()
                        except:
                            # If both fail, try a backup using scipy and manual face creation
                            hull = scipy.spatial.ConvexHull(points)
                            faces = []
                            for simplex in hull.simplices:
                                faces.append([3, simplex[0], simplex[1], simplex[2]])
                            surf = pv.PolyData(points, faces=np.array(faces))
                            
                    # If we have a surface, add it to the plotter
                    if surf is not None:
                        # Attempt to smooth the surface if it has enough faces
                        if surf.n_faces > 10:
                            try:
                                surf = surf.smooth(n_iter=100, relaxation_factor=0.1)
                            except:
                                pass  # Smoothing failed, use unsmoothed surface
                        
                        plotter.add_mesh(surf, color="red", opacity=0.5)
                        self.queue.put(("status", "Created triangulated surface"))
                    else:
                        raise ValueError("Surface generation failed with all methods")
                        
                except Exception as e:
                    self.queue.put(("error", f"Error in 3D triangulation: {str(e)}"))
                    # Create a convex hull as a last resort
                    try:
                        cloud = pv.PolyData(points)
                        hull = cloud.delaunay_3d().extract_surface()
                        plotter.add_mesh(hull, color="red", opacity=0.5)
                        self.queue.put(("status", "Created convex hull surface (fallback)"))
                    except:
                        self.queue.put(("status", "Unable to create surface, showing point cloud only"))
                        
        except Exception as e:
            self.queue.put(("error", f"Triangulation failed: {str(e)}"))
            self.queue.put(("status", "Unable to create surface, showing point cloud only"))

    def _create_loopstructural_surface(self, points, plotter, smoothing, ni, nj, nk):
        """Create a surface using LoopStructural"""
        try:
            # Try to directly import LoopStructural
            try:
                from LoopStructural import GeologicalModel
                from LoopStructural.interpolators import BiharmonicInterpolator
            except ImportError:
                from loopstructural import GeologicalModel
                from loopstructural.interpolators import BiharmonicInterpolator
            
            # Create bounding box with some padding
            # Determine x, y, z ranges from points
            x_min, y_min, z_min = np.min(points, axis=0) - 10
            x_max, y_max, z_max = np.max(points, axis=0) + 10
            
            # Define model bounds and resolution
            bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
            
            # Calculate appropriate resolution based on data size
            res = max(3, min(20, int(min(ni, nj, nk) / 10)))
            
            # Create geological model
            model = GeologicalModel(bounds, res)
            
            # Create interpolator
            self.queue.put(("status", "Setting up interpolator..."))
            self.queue.put(("progress", 70))
            
            # Use BiharmonicInterpolator which is better for sparse points
            interpolator = BiharmonicInterpolator(model.interpolator.support)
            
            # Add points to interpolator (points are already thinned if needed)
            value = 0  # Target value for the isosurface
            for point in points:
                interpolator.add_point(point, value)
            
            # Apply smoothing constraints
            self.queue.put(("status", "Applying smoothing constraints..."))
            self.queue.put(("progress", 80))
            
            # The smoothness factor controls how smooth the surface is
            interpolator.add_smoothness_constraint(smoothing)
            
            # Solve the interpolation
            self.queue.put(("status", "Solving interpolation..."))
            self.queue.put(("progress", 85))
            
            interpolator.solve_system()
            
            # Evaluate on grid
            self.queue.put(("status", "Creating 3D surface..."))
            self.queue.put(("progress", 90))
            
            # Create surface from interpolation result
            xi = np.linspace(x_min, x_max, res)
            yi = np.linspace(y_min, y_max, res)
            zi = np.linspace(z_min, z_max, res)
            
            X, Y, Z = np.meshgrid(xi, yi, zi)
            pts = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
            
            scalar_field = np.zeros(pts.shape[0])
            for i, p in enumerate(pts):
                scalar_field[i] = interpolator.evaluate_value(p)
            
            # Create grid and add scalar field
            grid = pv.StructuredGrid(X, Y, Z)
            grid["values"] = scalar_field
            
            # Extract isosurface at zero value
            surf = grid.contour([0])
            
            # If surface creation was successful, add to plot
            if surf.n_points > 0:
                plotter.add_mesh(surf, color="red", opacity=0.5)
                self.queue.put(("status", "Surface created successfully"))
            else:
                raise ValueError("Failed to create surface - try adjusting smoothness")
            
        except Exception as e:
            self.queue.put(("error", f"Error in LoopStructural surface generation: {str(e)}"))
            raise  # Re-raise to trigger triangulation fallback

    def _add_seismic_slices_to_plot(self, plotter, slice_type, seismic_volume, indices):
        """Add representative seismic slices to the 3D visualization for context"""
        try:
            ni, nj, nk = seismic_volume.shape
            
            # Calculate downsampling factors to match standard 3D visualization
            ds_i = max(1, ni // 150)
            ds_j = max(1, nj // 150)
            ds_k = max(1, nk // 150)
            
            # Process each slice based on type and add to visualization
            for i, idx in enumerate(indices):
                idx = int(idx)  # Ensure integer index
                
                try:
                    # Process differently based on slice type
                    if slice_type == "inline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[idx, ::ds_j, ::ds_k]
                        
                        # Get dimensions after downsampling
                        s_nj, s_nk = slice_data.shape
                        
                        # Create a grid at the appropriate x position
                        x = np.ones((s_nj, s_nk)) * idx
                        y = np.tile(np.arange(0, s_nj * ds_j, ds_j).reshape(-1, 1), (1, s_nk))
                        z = np.tile(np.arange(0, s_nk * ds_k, ds_k), (s_nj, 1)) / 8  # Scale z as in visualization
                        
                    elif slice_type == "crossline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, idx, ::ds_k]
                        
                        # Get dimensions after downsampling
                        s_ni, s_nk = slice_data.shape
                        
                        # Create a grid at the appropriate y position
                        x = np.tile(np.arange(0, s_ni * ds_i, ds_i).reshape(-1, 1), (1, s_nk))
                        y = np.ones((s_ni, s_nk)) * idx
                        z = np.tile(np.arange(0, s_nk * ds_k, ds_k), (s_ni, 1)) / 8  # Scale z as in visualization
                        
                    else:  # timeslice
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, ::ds_j, idx]
                        
                        # Get dimensions after downsampling
                        s_ni, s_nj = slice_data.shape
                        
                        # Create a grid at the appropriate z position
                        x = np.tile(np.arange(0, s_ni * ds_i, ds_i).reshape(-1, 1), (1, s_nj))
                        y = np.tile(np.arange(0, s_nj * ds_j, ds_j), (s_ni, 1))
                        z = np.ones((s_ni, s_nj)) * idx / 8  # Scale z as in visualization
                    
                    # Normalize slice data for display
                    vmin, vmax = np.percentile(slice_data, [5, 95])
                    norm_data = np.clip(slice_data, vmin, vmax)
                    norm_data = (norm_data - vmin) / (vmax - vmin)
                    
                    # Create a structured grid for the slice
                    grid = pv.StructuredGrid(x, y, z)
                    grid.point_data["values"] = norm_data.flatten(order='F')
                    
                    # Add the slice to the plotter - use seismic colormap to match 3D visualization
                    plotter.add_mesh(grid, scalars="values", opacity=0.5, cmap="seismic", show_edges=False)
                    
                    # Add a delimiter plane to match 3D visualization
                    # Note: Use the PyVista Plane parameters correctly without k_size
                    if slice_type == "inline":
                        delimiter = pv.Plane(center=(idx, s_nj*ds_j//2, s_nk*ds_k//16), 
                                           direction=(1, 0, 0), 
                                           i_size=1, j_size=s_nj*ds_j)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                    elif slice_type == "crossline":
                        delimiter = pv.Plane(center=(s_ni*ds_i//2, idx, s_nk*ds_k//16), 
                                           direction=(0, 1, 0), 
                                           i_size=s_ni*ds_i, j_size=1)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                    else:  # timeslice
                        delimiter = pv.Plane(center=(s_ni*ds_i//2, s_nj*ds_j//2, idx/8), 
                                           direction=(0, 0, 1), 
                                           i_size=s_ni*ds_i, j_size=s_nj*ds_j)
                        plotter.add_mesh(delimiter, color="grey", opacity=0.1)
                        
                except Exception as inner_e:
                    print(f"Error adding slice {idx}: {inner_e}")
                    continue
        
        except Exception as e:
            print(f"Error adding seismic slices to visualization: {e}")
            # Continue without seismic slices if there's an error

    def _ensure_masks_propagated(self, indices, obj_id):
        """Ensure masks are propagated for the given slices before visualization"""
        # Calculate how many slices to check
        check_count = min(len(indices), 5)
        check_indices = indices[::len(indices)//check_count] if check_count > 1 else [indices[0]]
        
        # Count how many have masks
        mask_count = 0
        for idx in check_indices:
            try:
                mask = self.predictor.get_mask_for_frame(idx, obj_id)
                if mask is not None and np.any(mask):
                    mask_count += 1
            except:
                pass
        
        # If less than 70% have masks, propagate
        if mask_count < 0.7 * len(check_indices):
            self.queue.put(("status", "Some masks missing. Propagating masks..."))
            
            # Current slice type
            slice_type = self.current_slice_type.get()
            
            # Get the current position
            current_frame_idx = self.current_slice_idx.get()
            
            # Store the current position for the demo predictor
            self.predictor.demo_frame_idx = current_frame_idx
            
            # Force demo mode for more reliable propagation
            use_demo_mode = True
            
            # Initialize video predictor with all necessary indices
            slice_indices = list(indices)
            try:
                if use_demo_mode:
                    self.predictor.demo_mode = True
                    self.predictor.init_video_predictor(slice_indices)
                else:
                    self.predictor.init_video_predictor(slice_indices)
            except Exception as e:
                print(f"Error initializing video predictor: {e}")
                return
            
            # Add points to current frame
            if hasattr(self, 'points') and len(self.points) > 0:
                try:
                    # Get labels
                    labels = [1 if label == 'foreground' else 0 for label in self.point_labels]
                    
                    # Add points to frame
                    if use_demo_mode:
                        self.predictor.demo_points = self.points
                        self.predictor.demo_labels = labels
                        self.predictor.demo_obj_id = obj_id
                    else:
                        # Find frame index in sequence
                        if current_frame_idx in slice_indices:
                            frame_pos = slice_indices.index(current_frame_idx)
                        else:
                            frame_pos = 0  # Default to first frame
                        
                        self.predictor.add_point_to_video(frame_pos, obj_id, self.points, labels)
                except Exception as e:
                    print(f"Error adding points to video: {e}")
                    return
            
            # Propagate masks
            try:
                self.predictor.propagate_masks()
                self.queue.put(("status", "Masks propagated successfully"))
            except Exception as e:
                print(f"Error propagating masks: {e}")
                return


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seismic Interpretation App with SAM2")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without loading the SAM2 model")
    parser.add_argument("--model", choices=["base-plus", "large", "small", "tiny", "2.1-base-plus", "2.1-large"], 
                        default="base-plus", help="SAM2 model size to use")
    args = parser.parse_args()
    
    # Set demo mode based on command line arguments
    demo_mode = args.demo
    
    # Define the model ID based on the selected model size
    model_id = f"facebook/sam2-hiera-{args.model}" if not args.model.startswith("2.1") else f"facebook/sam2.1-hiera-{args.model[4:]}"
    
    if demo_mode:
        print("Starting in demo mode (no model will be loaded)")
    else:
        print(f"Starting with SAM2 model: {model_id}")
    
    root = tk.Tk()
    root.geometry("1200x800")  # Set initial window size
    
    # Create the app with the specified model_id
    app = SeismicApp(root, demo_mode=demo_mode, model_id=model_id)
    
    root.mainloop()

if __name__ == "__main__":
    main()