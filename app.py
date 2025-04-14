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

from segy_loader import SegyLoader
from seismic_predictor import SeismicPredictor

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
        
        # Initialize point collection per object ID
        self.object_annotations = {} # Stores {'points': [], 'labels': []} for each object ID
        
        # Add back the drawing_mode initialization
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
        self.object_id_spinbox = ttk.Spinbox(annot_frame, from_=1, to=10, width=3, 
                                            textvariable=self.current_object_id,
                                            command=self._on_object_id_change) # Add command
        self.object_id_spinbox.pack(side=tk.LEFT, padx=5)
        self.object_id_spinbox.bind("<Return>", self._on_object_id_change) # Bind Return key
        
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
            
        # Get annotations for the current object ID
        points, point_labels = self._get_current_annotations()
        
        # Add the point and label
        points.append([event.xdata, event.ydata])
        label = 1 if self.drawing_mode.get() == "foreground" else 0
        point_labels.append(label)
        
        # Update the display
        self._update_display_with_points()
    
    def _update_display_with_points(self):
        """Update the display with current slice and annotation points for ALL objects."""
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            return
            
        # Clear the axis
        self.ax.clear()
        
        # Display the slice
        vmin, vmax = np.percentile(self.current_slice, [5, 95])
        self.ax.imshow(self.current_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Get current object ID for highlighting
        current_obj_id = self.current_object_id.get()
        colors = plt.get_cmap('tab10').colors # Use consistent colors
        
        # Iterate through all object IDs that have annotations
        for obj_id, annotations in self.object_annotations.items():
            points = annotations['points']
            point_labels = annotations['labels']
            
            if not points: # Skip if no points for this object
                continue
                
            obj_color = colors[ (obj_id - 1) % len(colors) ]
            is_current = (obj_id == current_obj_id)
            marker_size = 40 if is_current else 25 # Make current object points larger
            alpha = 1.0 if is_current else 0.7 # Make other objects slightly transparent
            
            # Add foreground points for this object
            fg_points = [p for i, p in enumerate(points) if point_labels[i] == 1]
            if fg_points:
                fg_points = np.array(fg_points)
                self.ax.scatter(fg_points[:, 0], fg_points[:, 1], 
                                color=obj_color, marker='o', s=marker_size, alpha=alpha,
                                label=f'Obj {obj_id} FG' if is_current else f'_Obj {obj_id} FG') # Underscore hides from legend unless current
                
            # Add background points for this object
            bg_points = [p for i, p in enumerate(points) if point_labels[i] == 0]
            if bg_points:
                bg_points = np.array(bg_points)
                self.ax.scatter(bg_points[:, 0], bg_points[:, 1], 
                                color=obj_color, marker='x', s=marker_size, alpha=alpha,
                                label=f'Obj {obj_id} BG' if is_current else f'_Obj {obj_id} BG')
                                
        # Update title - Indicate the *active* object ID
        self.ax.set_title(f"{self.current_slice_type.get().capitalize()} {self.current_slice_idx.get()} (Active Object: {current_obj_id})")
        self.ax.legend() # Show legend (only for current object due to underscore)
        
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
            
            # --- Attempt to display slice with all relevant masks ---
            current_obj_id = self.current_object_id.get()
            mask_for_current = None
            try:
                # Determine frame position (assuming slice_idx == frame_pos for now)
                # A more robust mapping might be needed depending on propagation implementation
                frame_pos = idx 
                
                print(f"Checking for mask for current Object {current_obj_id} at frame {frame_pos}")
                mask_for_current = self.predictor.get_mask_for_frame(frame_pos, current_obj_id)
                
                if mask_for_current is not None and np.any(mask_for_current):
                     print(f"Found mask for current object {current_obj_id}. Displaying all masks.")
                     # Call display_mask, passing the mask we just found for the current object.
                     # display_mask will handle fetching and displaying others.
                     self.display_mask(mask_for_current) 
                     return # We are done displaying for this slice load

                else:
                     print(f"No mask found for current object {current_obj_id}. Checking other objects.")
                     # Even if the current object has no mask, others might.
                     # We can still call display_mask with None for the current object mask.
                     # It will then try to fetch masks for all other known objects.
                     self.display_mask(None)
                     return # We are done displaying for this slice load

            except Exception as e:
                 print(f"Error fetching or displaying masks: {e}")
                 # Fallback to displaying only points if mask fetching/display fails
                 print("Falling back to displaying points only.")
                 self._update_display_with_points()
                 return

            # This part should theoretically not be reached if the try/except block works correctly
            # but kept as a final fallback.
            # print("Displaying slice with points only (fallback).")
            # self._update_display_with_points()
            
        except Exception as e:
            # Add traceback for debugging
            import traceback
            trace = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to load slice: {str(e)}\n\n{trace}")
    
    def _generate_mask(self):
        """Generate mask from annotation points using SAM2"""
        points, _ = self._get_current_annotations() # Get points for current object
        if not points:
            messagebox.showinfo("Info", "Please add at least one annotation point for the current object first.")
            return
            
        self.status_text.set(f"Generating mask for Object ID {self.current_object_id.get()}...")
        self.progress_var.set(50)
        
        # Start prediction in a thread
        threading.Thread(target=self._generate_mask_thread, daemon=True).start()
    
    def _generate_mask_thread(self):
        """Thread function for generating mask"""
        try:
            # Get points and labels for the current object ID
            points, point_labels = self._get_current_annotations()
            obj_id = self.current_object_id.get()
            
            # Predict masks
            masks, scores, logits = self.predictor.predict_masks_from_points(
                points, point_labels, multimask_output=True
            )
            
            # Select the mask with highest score
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Store the generated mask for this object and slice (in predictor)
            self.predictor.store_mask_for_object(
                self.current_slice_type.get(), 
                self.current_slice_idx.get(), 
                obj_id, 
                best_mask
            )
            
            # Update UI in main thread
            self.queue.put(("display_mask", best_mask))
            self.queue.put(("status", f"Mask generated for Object {obj_id} with score: {scores[best_mask_idx]:.4f}"))
            self.queue.put(("progress", 100))
                
        except Exception as e:
            self.queue.put(("error", f"Error generating mask for Object {self.current_object_id.get()}: {str(e)}"))
            self.queue.put(("progress", 0))
    
    def _clear_annotations(self):
        """Clear annotation points"""
        obj_id = self.current_object_id.get()
        if obj_id in self.object_annotations:
            self.object_annotations[obj_id]['points'] = []
            self.object_annotations[obj_id]['labels'] = []
            print(f"Cleared annotations for Object ID {obj_id}")
        else:
            print(f"No annotations found for Object ID {obj_id} to clear.")
        self._update_display_with_points()
    
    def _propagate_to_all(self):
        """Propagate the mask to all slices using SAM2 video predictor"""
        points, _ = self._get_current_annotations() # Get points for current object
        if not points:
            messagebox.showinfo("Info", "Please add at least one annotation point for the current object and generate a mask first.")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Confirm", f"This will propagate the mask for Object ID {self.current_object_id.get()} to all slices. It may take some time. Continue?"):
            return
            
        self.status_text.set(f"Preparing to propagate masks for Object ID {self.current_object_id.get()}...")
        self.progress_var.set(10)
        
        # Start propagation in a thread
        threading.Thread(target=self._propagate_thread, daemon=True).start()
    
    def _propagate_thread(self):
        """Thread function for mask propagation"""
        try:
            slice_type = self.current_slice_type.get()
            obj_id = self.current_object_id.get() # Get current object ID
            points, point_labels = self._get_current_annotations() # Get annotations for this object
            
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
            
            self.queue.put(("status", f"Preparing to propagate masks for Object {obj_id} from {slice_type} {current_frame_idx}..."))
            self.queue.put(("progress", 15))
            
            print(f"Slice type: {slice_type}, Current idx: {current_frame_idx}, Position: {current_frame_pos}, Object ID: {obj_id}")
            
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
                    self.predictor.init_video_predictor(demo_slices, obj_id) # Pass obj_id
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
                    self.predictor.init_video_predictor(slices, obj_id, max_slices=MAX_SLICES) # Pass obj_id
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    print("Falling back to demo mode for propagation")
                    
                    # Retry with demo mode
                    try:
                        self.predictor.demo_mode = True
                        self.predictor.init_video_predictor(slices[:MAX_SLICES], obj_id) # Pass obj_id
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
            
            self.queue.put(("status", f"Adding points for Object {obj_id} to frame {current_frame_idx} (position {current_frame_pos})..."))
            self.queue.put(("progress", 30))
            
            try:
                # Use the retrieved points and labels for this object
                self.predictor.add_point_to_video(
                    current_frame_pos,  # Use position in frame sequence
                    obj_id,
                    points, 
                    point_labels 
                )
            except Exception as e:
                self.queue.put(("error", f"Error adding points to video for Object {obj_id}: {str(e)}"))
                self.queue.put(("progress", 0))
                return
            
            # Propagate forward
            self.queue.put(("status", f"Propagating masks forward for Object {obj_id}..."))
            self.queue.put(("progress", 50))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    obj_id=obj_id, # Pass obj_id
                    reverse=False
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks forward for Object {obj_id}: {str(e)}"))
                # Continue with backward propagation even if forward fails
            
            # Propagate backward
            self.queue.put(("status", f"Propagating masks backward for Object {obj_id}..."))
            self.queue.put(("progress", 80))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    obj_id=obj_id, # Pass obj_id
                    reverse=True
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks backward for Object {obj_id}: {str(e)}"))
                # Continue to get mask if possible
            
            # Get the result for current frame
            self.queue.put(("status", f"Getting mask for Object {obj_id} on current frame..."))
            self.queue.put(("progress", 90))
            
            try:
                mask = self.predictor.get_mask_for_frame(
                    current_frame_pos,  # Use position in sequence 
                    obj_id
                )
                
                # Display the result
                self.queue.put(("display_mask", mask))
                self.queue.put(("status", f"Propagation complete for Object {obj_id}."))
                self.queue.put(("progress", 100))
            except Exception as e:
                self.queue.put(("error", f"Error getting mask for current frame (Object {obj_id}): {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error propagating masks: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
    
    def display_mask(self, mask_for_current_obj):
        """Display the slice with masks for ALL relevant objects overlaid."""
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            print("Warning: display_mask called without a current slice.")
            return
            
        # Clear the axis
        self.ax.clear()
        
        # Display the slice
        vmin, vmax = np.percentile(self.current_slice, [5, 95])
        self.ax.imshow(self.current_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Get current slice info
        slice_type = self.current_slice_type.get()
        slice_idx = self.current_slice_idx.get()
        frame_pos = slice_idx # Assuming frame position matches slice index for simplicity here
                              # A more robust mapping might be needed if using selected_slice_indices
        
        current_obj_id = self.current_object_id.get()
        colors = plt.get_cmap('tab10').colors
        
        # --- Display masks for ALL objects that might have one ---
        # Combine known object IDs from annotations and predictor states
        all_known_obj_ids = set(self.object_annotations.keys())
        if self.predictor:
             if not self.predictor.demo_mode:
                 all_known_obj_ids.update(self.predictor.inference_state.keys())
             else:
                 all_known_obj_ids.update(self.predictor.demo_state.keys())
                 
        print(f"Displaying masks for potential Object IDs: {list(all_known_obj_ids)}")

        for obj_id in sorted(list(all_known_obj_ids)):
            mask_to_display = None
            is_current = (obj_id == current_obj_id)
            
            # If this is the current object, use the mask passed to the function
            if is_current and mask_for_current_obj is not None:
                mask_to_display = mask_for_current_obj
                print(f"Using provided mask for current Object {obj_id}")
            else:
                # Otherwise, try to fetch the mask from the predictor
                try:
                    # Need the frame position corresponding to the slice index
                    # This might require the slice_to_frame_map if propagation used a subset
                    # For simplicity, we assume frame_pos = slice_idx here. Refine if needed.
                    fetched_mask = self.predictor.get_mask_for_frame(frame_pos, obj_id)
                    if fetched_mask is not None and np.any(fetched_mask):
                         mask_to_display = fetched_mask
                         print(f"Fetched mask for Object {obj_id} on frame {frame_pos}")
                    # else:
                    #     print(f"No mask found for Object {obj_id} on frame {frame_pos}")

                except Exception as e:
                     print(f"Could not fetch mask for Object {obj_id} on frame {frame_pos}: {e}")

            # If we have a mask for this object ID, display it
            if mask_to_display is not None:
                 if mask_to_display.shape != self.current_slice.shape:
                     print(f"Warning: Mask shape {mask_to_display.shape} mismatch for Object {obj_id} on slice {self.current_slice.shape}. Skipping.")
                     continue # Skip overlay if shapes don't match

                 mask_overlay = np.zeros((*mask_to_display.shape, 4))
                 obj_color = colors[ (obj_id - 1) % len(colors) ]
                 alpha = 0.6 if is_current else 0.4 # Current slightly more opaque
                 
                 mask_overlay[mask_to_display > 0] = [*obj_color, alpha]
                 self.ax.imshow(mask_overlay, aspect='auto')
                 print(f"Overlayed mask for Object {obj_id} with color {obj_color} alpha {alpha}")


        # --- Plot points for ALL objects ---
        for obj_id, annotations in self.object_annotations.items():
             points = annotations['points']
             point_labels = annotations['labels']
             
             if not points: continue
                 
             obj_color = colors[ (obj_id - 1) % len(colors) ]
             is_current = (obj_id == current_obj_id)
             marker_size = 40 if is_current else 25
             alpha = 1.0 if is_current else 0.7
             
             fg_points = [p for i, p in enumerate(points) if point_labels[i] == 1]
             if fg_points:
                 fg_points = np.array(fg_points)
                 self.ax.scatter(fg_points[:, 0], fg_points[:, 1], color=obj_color, marker='o', 
                                 s=marker_size, alpha=alpha, label=f'Obj {obj_id} FG' if is_current else f'_Obj {obj_id} FG')
                                 
             bg_points = [p for i, p in enumerate(points) if point_labels[i] == 0]
             if bg_points:
                 bg_points = np.array(bg_points)
                 self.ax.scatter(bg_points[:, 0], bg_points[:, 1], color=obj_color, marker='x', 
                                 s=marker_size, alpha=alpha, label=f'Obj {obj_id} BG' if is_current else f'_Obj {obj_id} BG')

        # Update title
        self.ax.set_title(f"{slice_type.capitalize()} {slice_idx} (Active Object: {current_obj_id}) with Masks")
        self.ax.legend() # Show legend
        
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
        
        # Current object ID
        obj_id = self.current_object_id.get()
        
        # Check if we have masks at all for this object ID using the dedicated method
        has_masks = False
        if hasattr(self.predictor, 'has_masks_for_object') and callable(self.predictor.has_masks_for_object):
            has_masks = self.predictor.has_masks_for_object(obj_id) # Check specific object
        else:
            # Fallback check if the method somehow doesn't exist (should not happen ideally)
            print("Warning: predictor.has_masks_for_object not found. Limited mask checking.")
            if not self.predictor.demo_mode and obj_id in self.predictor.inference_state:
                 has_masks = bool(self.predictor.inference_state[obj_id].get("output"))
            elif self.predictor.demo_mode and obj_id in self.predictor.demo_state:
                 demo_inference_state = self.predictor.demo_state[obj_id].get('inference_state', {})
                 has_masks = bool(demo_inference_state.get("output"))

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
            
            # Count how many masks we have for the current object
            mask_count = 0
            # obj_id is already defined
            
            # Sample a few slices to see if we have masks for this object
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
                                     "Propagate masks for Object {obj_id} before visualization? (Recommended)"):
                    # Run propagation with a wider range to cover all visualization slices
                    self._propagate_all_for_visualization(num_slices, obj_id) # Pass obj_id
                    # Return early - the 3D visualization will be triggered after propagation completes
                    return
                
        # Start visualization in a separate thread to avoid blocking the main GUI
        self.status_text.set(f"Preparing 3D visualization for ALL Objects...") # Updated status
        self.progress_var.set(10)
        
        # Pass only num_slices, _create_3d_visualization will handle finding all objects
        threading.Thread(target=lambda: self._create_3d_visualization(num_slices), daemon=True).start() # REMOVED obj_id
    
    def _propagate_all_for_visualization(self, num_slices=100, obj_id=None): # Accept obj_id
        """Propagate masks to all slices needed for visualization"""
        if obj_id is None:
            obj_id = self.current_object_id.get()
            
        # Start propagation in a thread
        self.status_text.set(f"Propagating masks for Object {obj_id} for visualization...")
        self.progress_var.set(5)
        
        # Run with increased slice coverage
        threading.Thread(target=lambda: self._propagate_thread_for_viz(num_slices, obj_id), daemon=True).start() # Pass obj_id
    
    def _propagate_thread_for_viz(self, num_slices=100, obj_id=None): # Accept obj_id
        """Thread function for mask propagation with visualization followup"""
        try:
            if obj_id is None: # Ensure obj_id is set
                obj_id = self.current_object_id.get()
                
            slice_type = self.current_slice_type.get()
            points, point_labels = self._get_current_annotations() # Use annotations for the specific obj_id
            
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
            
            self.queue.put(("status", f"Preparing to propagate masks for Object {obj_id} from {slice_type} {current_frame_idx}..."))
            self.queue.put(("progress", 15))
            
            print(f"Slice type: {slice_type}, Current idx: {current_frame_idx}, Position: {current_frame_pos}, Object ID: {obj_id}")
            
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
                    self.predictor.init_video_predictor(demo_slices, obj_id) # Pass obj_id
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
                    self.predictor.init_video_predictor(slices, obj_id, max_slices=MAX_SLICES) # Pass obj_id
                except Exception as e:
                    self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                    print("Falling back to demo mode for propagation")
                    
                    # Retry with demo mode
                    try:
                        self.predictor.demo_mode = True
                        self.predictor.init_video_predictor(slices[:MAX_SLICES], obj_id) # Pass obj_id
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
            
            self.queue.put(("status", f"Adding points for Object {obj_id} to frame {current_frame_idx} (position {current_frame_pos})..."))
            self.queue.put(("progress", 30))
            
            try:
                # Use points/labels for this object
                self.predictor.add_point_to_video(
                    current_frame_pos,  # Use position in frame sequence
                    obj_id,
                    points, 
                    point_labels
                )
            except Exception as e:
                self.queue.put(("error", f"Error adding points to video for Object {obj_id}: {str(e)}"))
                self.queue.put(("progress", 0))
                return
            
            # Propagate forward
            self.queue.put(("status", f"Propagating masks forward for Object {obj_id}..."))
            self.queue.put(("progress", 50))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    obj_id=obj_id, # Pass obj_id
                    reverse=False
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks forward for Object {obj_id}: {str(e)}"))
                # Continue with backward propagation even if forward fails
            
            # Propagate backward
            self.queue.put(("status", f"Propagating masks backward for Object {obj_id}..."))
            self.queue.put(("progress", 80))
            
            try:
                self.predictor.propagate_masks(
                    start_frame_idx=current_frame_pos,  # Use position in sequence
                    obj_id=obj_id, # Pass obj_id
                    reverse=True
                )
            except Exception as e:
                self.queue.put(("error", f"Error propagating masks backward for Object {obj_id}: {str(e)}"))
                # Continue to get mask if possible
            
            # Get the result for current frame
            self.queue.put(("status", f"Getting mask for Object {obj_id} on current frame..."))
            self.queue.put(("progress", 90))
            
            try:
                mask = self.predictor.get_mask_for_frame(
                    current_frame_pos,  # Use position in sequence 
                    obj_id
                )
                
                # Display the result on the 2D canvas
                self.queue.put(("display_mask", mask))
                self.queue.put(("status", f"Propagation complete for Object {obj_id}. Starting 3D visualization..."))
                self.queue.put(("progress", 95))
                
                # Now start the 3D visualization (no obj_id needed, it will show all)
                threading.Thread(target=lambda: self._create_3d_visualization(num_slices), daemon=True).start() # REMOVED obj_id
                
            except Exception as e:
                self.queue.put(("error", f"Error getting mask for current frame (Object {obj_id}): {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error propagating masks: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))

    def _create_3d_visualization(self, num_slices=20): # REMOVED obj_id parameter
        """Create a very simple 3D visualization of seismic data and ALL available masks using PyVista"""
        try:
            # if obj_id is None: # Ensure obj_id is set - No longer needed
            #     obj_id = self.current_object_id.get()
                
            self.queue.put(("status", f"Creating 3D visualization with {num_slices} slices for ALL objects...")) # Updated status
            self.queue.put(("progress", 30))
            
            # Get the seismic data
            seismic_volume = self.segy_loader.data
            
            # Get dimensions
            ni, nj, nk = seismic_volume.shape
            
            # Current slice type
            slice_type = self.current_slice_type.get()
            
            # # Get current object ID for masks (already passed as argument) - No longer needed
            # # obj_id = self.current_object_id.get()
            
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
                
            # Get all known object IDs to check for masks
            all_known_obj_ids = set(self.object_annotations.keys())
            if self.predictor:
                if not self.predictor.demo_mode:
                    all_known_obj_ids.update(self.predictor.inference_state.keys())
                else:
                    all_known_obj_ids.update(self.predictor.demo_state.keys())
            print(f"Checking for masks for Object IDs in 3D view: {list(all_known_obj_ids)}")
            colors = plt.get_cmap('tab10').colors # Use consistent colors

            # Create a simple display for each slice
            for i, idx in enumerate(indices):
                try:
                    self.queue.put(("status", f"Processing slice {i+1}/{num_slices}: {idx}"))
                    self.queue.put(("progress", 30 + int(60 * i / num_slices)))
                    
                    # Get slice data only once per slice index
                    slice_data = None
                    grid = None # Initialize grid to None
                    
                    # Process differently based on slice type
                    if slice_type == "inline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[idx, ::ds_j, ::ds_k]
                        s_nj, s_nk = slice_data.shape
                        x_coords = np.ones(s_nj * s_nk) * idx
                        y_coords = np.repeat(np.arange(0, s_nj * ds_j, ds_j), s_nk)
                        z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_nj) / 8
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        grid = pv.PolyData(points)
                        
                    elif slice_type == "crossline":
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, idx, ::ds_k]
                        s_ni, s_nk = slice_data.shape
                        x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nk)
                        y_coords = np.ones(s_ni * s_nk) * idx
                        z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_ni) / 8
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        grid = pv.PolyData(points)
                        
                    else:  # timeslice
                        # Get slice data with downsampling
                        slice_data = seismic_volume[::ds_i, ::ds_j, idx]
                        s_ni, s_nj = slice_data.shape
                        x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nj)
                        y_coords = np.tile(np.arange(0, s_nj * ds_j, ds_j), s_ni)
                        z_coords = np.ones(s_ni * s_nj) * idx / 8
                        points = np.column_stack((x_coords, y_coords, z_coords))
                        grid = pv.PolyData(points)

                    # Add seismic data to plotter if available
                    if grid is not None and slice_data is not None:
                         vmin, vmax = np.percentile(slice_data, [5, 95])
                         norm_data = np.clip(slice_data, vmin, vmax)
                         norm_data = (norm_data - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-6 else np.zeros_like(slice_data)
                         grid.point_data["intensity"] = norm_data.flatten()
                         
                         # Add mesh to plotter with seismic colormap
                         plotter.add_mesh(grid, cmap="seismic", point_size=3, render_points_as_spheres=True, scalars="intensity")
                         
                         # Add delimiter plane (optional, can be removed if too cluttered)
                         # ... (delimiter code can remain or be removed) ...

                    # --- Loop through all known object IDs to add their masks ---
                    for current_obj_id_to_display in all_known_obj_ids:
                        try:
                            mask = self.predictor.get_mask_for_frame(idx, current_obj_id_to_display) 
                            if mask is not None and np.any(mask):
                                # Get color for this specific object
                                obj_color = colors[ (current_obj_id_to_display - 1) % len(colors) ]
                                
                                # Process mask based on slice type (handle downsampling and coordinates)
                                mask_points = []
                                if slice_type == "inline":
                                     mask_t = mask.T
                                     if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                         mask_ds = mask_t[::ds_j, ::ds_k]
                                         if mask_ds.shape == slice_data.shape:
                                             for j in range(s_nj):
                                                 for k in range(s_nk):
                                                     if mask_ds[j, k]:
                                                         mask_points.append([idx, j*ds_j, k*ds_k/8])
                                elif slice_type == "crossline":
                                     mask_t = mask.T
                                     if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                         mask_ds = mask_t[::ds_i, ::ds_k]
                                         if mask_ds.shape == slice_data.shape:
                                             for i_ in range(s_ni): # Use different index variable
                                                 for k in range(s_nk):
                                                     if mask_ds[i_, k]:
                                                         mask_points.append([i_*ds_i, idx, k*ds_k/8])
                                else: # timeslice
                                     if mask.shape[0] > 1 and mask.shape[1] > 1:
                                         mask_ds = mask[::ds_i, ::ds_j]
                                         if mask_ds.shape == slice_data.shape:
                                             for i_ in range(s_ni): # Use different index variable
                                                 for j in range(s_nj):
                                                     if mask_ds[i_, j]:
                                                         mask_points.append([i_*ds_i, j*ds_j, idx/8])

                                # If points were found for this mask, add them
                                if mask_points:
                                    mask_poly = pv.PolyData(np.array(mask_points))
                                    plotter.add_mesh(mask_poly, color=obj_color, point_size=5, 
                                                    render_points_as_spheres=True, 
                                                    label=f"Object {current_obj_id_to_display}") # Add label

                        except Exception as mask_err:
                            print(f"Error processing mask for Object {current_obj_id_to_display} on slice {idx}: {mask_err}")
                            pass # Continue to next object or slice
                            
                except Exception as slice_err:
                    self.queue.put(("status", f"Error processing slice {idx}: {slice_err}"))
                    # Continue with next slice
            
            # Add axes for reference and finalize
            try:
                plotter.add_axes()
                plotter.show_grid()
                plotter.add_legend() # Add legend to identify mask colors
            except:
                pass
            
            # Update progress
            self.queue.put(("status", "Showing 3D visualization..."))
            self.queue.put(("progress", 90))
            
            # Try to show the window with fallbacks
            try:
                # First try interactive viewing
                plotter.show(title=f"Seismic {slice_type} Slices with All Masks") # Update title
                self.queue.put(("status", f"3D visualization complete.")) # Updated status
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
                                     "No masks have been generated. You need to create and propagate masks before generating surfaces. Do you want to generate a surface anyway (might be empty)?"):
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
        self.status_text.set(f"Preparing surface generation for Object {self.current_object_id.get()}...")
        self.progress_var.set(10)
        
        threading.Thread(target=lambda: self._generate_3d_surface(num_slices, smoothing, use_triangulation, max_points, self.current_object_id.get()), # Pass obj_id
                        daemon=True).start()

    def _generate_3d_surface(self, num_slices=20, smoothing=10, use_triangulation=False, max_points=1000, obj_id=None): # Accept obj_id
        """Generate a 3D surface from mask points using LoopStructural or triangulation"""
        try:
            if obj_id is None: # Ensure obj_id is set
                obj_id = self.current_object_id.get()
                
            self.queue.put(("status", f"Generating 3D surface for Object {obj_id} from {num_slices} slices..."))
            self.queue.put(("progress", 20))
            
            # Get the seismic data dimensions
            seismic_volume = self.segy_loader.data
            ni, nj, nk = seismic_volume.shape
            
            # Current slice type and object ID (already passed as argument)
            slice_type = self.current_slice_type.get()
            # obj_id = self.current_object_id.get()
            
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
            
            # Check if we need to propagate masks first for this object
            self._ensure_masks_propagated(indices, obj_id) # Pass obj_id
            
            # Collect all mask points for surface generation for this object
            all_points = []
            
            # Process slices to collect points where masks exist for this object
            for i, idx in enumerate(indices):
                self.queue.put(("status", f"Processing slice {i+1}/{num_slices}: {idx} for Object {obj_id}"))
                self.queue.put(("progress", 20 + int(30 * i / num_slices)))
                
                try:
                    # Get mask for this slice and object
                    mask = self.predictor.get_mask_for_frame(idx, obj_id) # Use obj_id
                    
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
                    print(f"Error processing mask for frame {idx}, Object {obj_id}: {e}")
            
            # Check if we have enough points for surface generation
            if len(all_points) < 10:
                self.queue.put(("error", f"Not enough mask points found for Object {obj_id}. Please generate more masks or try a different object ID."))
                return
                
            self.queue.put(("status", f"Collected {len(all_points)} points for Object {obj_id} surface generation"))
            self.queue.put(("progress", 50))
            
            # Convert points to numpy array
            points_array = np.array(all_points)
            
            # Create a simple PyVista plotter for basic visualization
            plotter = pv.Plotter(notebook=False)
            plotter.set_background("white")
            
            # Add seismic data slices to context
            self.queue.put(("status", "Adding seismic slices for context..."))
            self._add_seismic_slices_to_plot(plotter, slice_type, seismic_volume, indices)
            
            # Use object color for point cloud and surface
            colors = plt.get_cmap('tab10').colors
            obj_color = colors[ (obj_id - 1) % len(colors) ]
            
            # Add scatter plot of the points
            point_cloud = pv.PolyData(points_array)
            plotter.add_mesh(point_cloud, color=obj_color, point_size=5, render_points_as_spheres=True) # Use object color
            
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
                self._create_triangulation_surface(subset_points, plotter, obj_color) # Pass color
            else:
                self.queue.put(("status", "Creating surface using LoopStructural..."))
                self.queue.put(("progress", 60))
                # This might fail if LoopStructural isn't available
                try:
                    self._create_loopstructural_surface(subset_points, plotter, smoothing, ni, nj, nk, obj_color) # Pass color
                except ImportError:
                    # Fall back to triangulation
                    self.queue.put(("status", "LoopStructural failed, falling back to triangulation..."))
                    self._create_triangulation_surface(subset_points, plotter, obj_color) # Pass color
                    
            # Add axes and show plot
            plotter.add_axes()
            plotter.show_grid()
            
            # Show the plotter
            self.queue.put(("status", "Displaying 3D surface..."))
            self.queue.put(("progress", 95))
            
            try:
                plotter.show(title=f"3D Surface from Masks (Object {obj_id})") # Update title
                self.queue.put(("status", f"3D surface visualization complete for Object {obj_id}"))
                self.queue.put(("progress", 100))
            except Exception as e:
                self.queue.put(("error", f"Error displaying 3D surface: {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error in surface generation: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
        
    def _create_triangulation_surface(self, points, plotter, color="red"): # Accept color
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
                        
                    plotter.add_mesh(mesh, color=color, opacity=0.5)
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
                        
                        plotter.add_mesh(surf, color=color, opacity=0.5)
                        self.queue.put(("status", "Created triangulated surface"))
                    else:
                        raise ValueError("Surface generation failed with all methods")
                        
                except Exception as e:
                    self.queue.put(("error", f"Error in 3D triangulation: {str(e)}"))
                    # Create a convex hull as a last resort
                    try:
                        cloud = pv.PolyData(points)
                        hull = cloud.delaunay_3d().extract_surface()
                        plotter.add_mesh(hull, color=color, opacity=0.5)
                        self.queue.put(("status", "Created convex hull surface (fallback)"))
                    except:
                        self.queue.put(("status", "Unable to create surface, showing point cloud only"))
                        
        except Exception as e:
            self.queue.put(("error", f"Triangulation failed: {str(e)}"))
            self.queue.put(("status", "Unable to create surface, showing point cloud only"))

    def _create_loopstructural_surface(self, points, plotter, smoothing, ni, nj, nk, color="red"): # Accept color
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
                plotter.add_mesh(surf, color=color, opacity=0.5)
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

    def _ensure_masks_propagated(self, indices, obj_id): # Accept obj_id
        """Ensure masks are propagated for the given slices before visualization"""
        # Calculate how many slices to check
        check_count = min(len(indices), 5)
        check_indices = indices[::len(indices)//check_count] if check_count > 1 else [indices[0]]
        
        # Count how many have masks for this object
        mask_count = 0
        for idx in check_indices:
            try:
                mask = self.predictor.get_mask_for_frame(idx, obj_id) # Use obj_id
                if mask is not None and np.any(mask):
                    mask_count += 1
            except:
                pass
        
        # If less than 70% have masks, propagate for this object
        if mask_count < 0.7 * len(check_indices):
            self.queue.put(("status", f"Some masks missing for Object {obj_id}. Propagating..."))
            
            # Current slice type
            slice_type = self.current_slice_type.get()
            
            # Get the current position
            current_frame_idx = self.current_slice_idx.get()
            
            # Get annotations for the specific object ID
            points, labels = [], []
            if obj_id in self.object_annotations:
                 points = self.object_annotations[obj_id]['points']
                 labels = self.object_annotations[obj_id]['labels']
            
            # Store the current position for the demo predictor (per object)
            self.predictor.set_demo_context(current_frame_idx, obj_id, points, labels)
            
            # Force demo mode for more reliable propagation
            use_demo_mode = True
            
            # Initialize video predictor with all necessary indices
            slice_indices = list(indices)
            try:
                if use_demo_mode:
                    self.predictor.demo_mode = True
                    self.predictor.init_video_predictor(slice_indices, obj_id) # Pass obj_id
                else:
                    self.predictor.init_video_predictor(slice_indices, obj_id) # Pass obj_id
            except Exception as e:
                print(f"Error initializing video predictor: {e}")
                return
            
            # Add points to current frame if they exist for this object
            if points and len(points) > 0:
                try:
                    # Labels are already 0 or 1
                    # labels = [1 if label == 'foreground' else 0 for label in self.point_labels]
                    
                    # Add points to frame
                    if use_demo_mode:
                        self.predictor.demo_obj_id = obj_id # Ensure demo obj_id is set
                        self.predictor.demo_points = points
                        self.predictor.demo_labels = labels
                        # demo_frame_idx is set via set_demo_context
                    else:
                        # Find frame index in sequence
                        if current_frame_idx in slice_indices:
                            frame_pos = slice_indices.index(current_frame_idx)
                        else:
                            frame_pos = 0  # Default to first frame
                        
                        self.predictor.add_point_to_video(frame_pos, obj_id, points, labels) # Use correct points/labels
                except Exception as e:
                    print(f"Error adding points to video: {e}")
                    return
            
            # Propagate masks for this object
            try:
                self.predictor.propagate_masks(obj_id=obj_id) # Pass obj_id
                self.queue.put(("status", f"Masks propagated successfully for Object {obj_id}"))
            except Exception as e:
                print(f"Error propagating masks for Object {obj_id}: {e}")
                return

    def _get_current_annotations(self):
        """Retrieve points and labels for the current object ID."""
        obj_id = self.current_object_id.get()
        if obj_id not in self.object_annotations:
            self.object_annotations[obj_id] = {'points': [], 'labels': []}
        return self.object_annotations[obj_id]['points'], self.object_annotations[obj_id]['labels']

    def _on_object_id_change(self, event=None):
        """Handle change in object ID."""
        print(f"Object ID changed to: {self.current_object_id.get()}")
        # Reload the current slice view to show annotations for the new object ID
        self._load_current_slice()


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