import os
import sys
import numpy as np

# Set environment variables for Qt and matplotlib before any Qt imports
os.environ['QT_API'] = 'pyside6'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # Let Qt find plugins automatically
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set matplotlib to use the correct Qt backend for PySide6
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for better PySide6 compatibility

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import Qt backend for matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                               QSlider, QLineEdit, QSpinBox, QRadioButton, 
                               QButtonGroup, QProgressBar, QMenuBar, QMenu, 
                               QMessageBox, QFileDialog, QFrame, QSizePolicy,
                               QGroupBox, QCheckBox, QInputDialog)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QPixmap
import threading
import queue
import argparse

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

class SeismicApp(QMainWindow):
    # Define custom signals for thread communication
    status_updated = Signal(str)
    progress_updated = Signal(float)
    error_occurred = Signal(str)
    scale_updated = Signal(int)
    mask_ready = Signal(object)
    
    def __init__(self, segy_path=None, demo_mode=False, model_id=None):
        super().__init__()
        self.setWindowTitle("Seismic Interpretation App")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # Convert StringVar and IntVar to regular variables with property-like access
        self._current_slice_type = "inline"
        self._current_slice_idx = 0
        self._current_object_id = 1
        self._drawing_mode = "foreground"
        self._status_text = "Ready. Load a SEGY file to begin."
        self._progress_value = 0.0
        
        # Initialize point collection per object ID
        self.object_annotations = {} # Stores {'points': [], 'labels': []} for each object ID
        
        # Message queue for thread communication (keeping for compatibility)
        self.queue = queue.Queue()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        # Create GUI
        self._create_menu()
        self._create_main_layout()
        
        # Connect signals
        self._connect_signals()
        
        # Start with model loading
        self._load_model()
        
        # Start queue processing with QTimer
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self.process_queue)
        self.queue_timer.start(100)  # Process queue every 100ms
    
    # Property methods to maintain compatibility with original code
    @property
    def current_slice_type(self):
        class MockVar:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
        return MockVar(self._current_slice_type)
    
    @property 
    def current_slice_idx(self):
        class MockVar:
            def __init__(self, parent):
                self.parent = parent
            def get(self):
                return self.parent._current_slice_idx
            def set(self, value):
                self.parent._current_slice_idx = value
                if hasattr(self.parent, 'slice_slider'):
                    self.parent.slice_slider.setValue(value)
        return MockVar(self)
    
    @property
    def current_object_id(self):
        class MockVar:
            def __init__(self, parent):
                self.parent = parent
            def get(self):
                return self.parent._current_object_id
            def set(self, value):
                self.parent._current_object_id = value
                if hasattr(self.parent, 'object_id_spinbox'):
                    self.parent.object_id_spinbox.setValue(value)
        return MockVar(self)
    
    @property
    def drawing_mode(self):
        class MockVar:
            def __init__(self, parent):
                self.parent = parent
            def get(self):
                return self.parent._drawing_mode
            def set(self, value):
                self.parent._drawing_mode = value
        return MockVar(self)
    
    @property
    def status_text(self):
        class MockVar:
            def __init__(self, parent):
                self.parent = parent
            def get(self):
                return self.parent._status_text
            def set(self, value):
                self.parent._status_text = value
                if hasattr(self.parent, 'status_label'):
                    self.parent.status_label.setText(value)
        return MockVar(self)
    
    @property
    def progress_var(self):
        class MockVar:
            def __init__(self, parent):
                self.parent = parent
            def get(self):
                return self.parent._progress_value
            def set(self, value):
                self.parent._progress_value = value
                if hasattr(self.parent, 'progress_bar'):
                    self.parent.progress_bar.setValue(int(value))
        return MockVar(self)
        
    def _create_menu(self):
        """Create application menu"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open SEGY...", self)
        open_action.triggered.connect(self._open_segy_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Slice menu
        slice_menu = menubar.addMenu("Slice Type")
        
        # Create action group for radio button behavior
        slice_group = QButtonGroup(self)
        
        inline_action = QAction("Inline", self)
        inline_action.setCheckable(True)
        inline_action.setChecked(True)
        inline_action.triggered.connect(lambda: self._set_slice_type("inline"))
        slice_menu.addAction(inline_action)
        slice_group.addButton(QPushButton())  # Placeholder for grouping
        
        crossline_action = QAction("Crossline", self)
        crossline_action.setCheckable(True)
        crossline_action.triggered.connect(lambda: self._set_slice_type("crossline"))
        slice_menu.addAction(crossline_action)
        
        timeslice_action = QAction("Time/Depth", self)
        timeslice_action.setCheckable(True)
        timeslice_action.triggered.connect(lambda: self._set_slice_type("timeslice"))
        slice_menu.addAction(timeslice_action)
        
        # Store actions for later reference
        self.slice_actions = [inline_action, crossline_action, timeslice_action]
        
        # SAM2 menu
        sam2_menu = menubar.addMenu("SAM2")
        
        clear_action = QAction("Clear Current Annotations", self)
        clear_action.triggered.connect(self._clear_annotations)
        sam2_menu.addAction(clear_action)
        
        propagate_action = QAction("Propagate to All Slices", self)
        propagate_action.triggered.connect(self._propagate_to_all)
        sam2_menu.addAction(propagate_action)
        
        viz_3d_action = QAction("Open 3D Visualization", self)
        viz_3d_action.triggered.connect(self._open_3d_visualization)
        sam2_menu.addAction(viz_3d_action)
        
        surface_3d_action = QAction("Generate 3D Surface", self)
        surface_3d_action.triggered.connect(self._open_3d_surface_generation)
        sam2_menu.addAction(surface_3d_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        instructions_action = QAction("Instructions", self)
        instructions_action.triggered.connect(self._show_instructions)
        help_menu.addAction(instructions_action)
    
    def _set_slice_type(self, slice_type):
        """Handle slice type change from menu"""
        self._current_slice_type = slice_type
        # Update radio button states
        type_map = {"inline": 0, "crossline": 1, "timeslice": 2}
        for i, action in enumerate(self.slice_actions):
            action.setChecked(i == type_map[slice_type])
        self._update_slice_view()
    
    def _create_main_layout(self):
        """Create the main application layout"""
        # Top control panel
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Slice controls
        slice_widget = QWidget()
        slice_layout = QHBoxLayout(slice_widget)
        slice_layout.addWidget(QLabel("Slice Index:"))
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_change)
        slice_layout.addWidget(self.slice_slider, 1)  # stretch factor 1
        
        self.slice_entry = QLineEdit()
        self.slice_entry.setMaximumWidth(60)
        self.slice_entry.setText("0")
        self.slice_entry.returnPressed.connect(self._on_slice_entry_change)
        slice_layout.addWidget(self.slice_entry)
        
        control_layout.addWidget(slice_widget, 1)
        
        # Annotation controls
        annot_group = QGroupBox("Annotation")
        annot_layout = QHBoxLayout(annot_group)
        
        # Radio buttons for drawing mode
        self.drawing_mode_group = QButtonGroup(self)
        self.foreground_radio = QRadioButton("Foreground")
        self.background_radio = QRadioButton("Background")
        self.foreground_radio.setChecked(True)
        
        self.drawing_mode_group.addButton(self.foreground_radio, 0)
        self.drawing_mode_group.addButton(self.background_radio, 1)
        self.drawing_mode_group.buttonClicked.connect(self._on_drawing_mode_change)
        
        annot_layout.addWidget(self.foreground_radio)
        annot_layout.addWidget(self.background_radio)
        
        annot_layout.addWidget(QLabel("Object ID:"))
        self.object_id_spinbox = QSpinBox()
        self.object_id_spinbox.setMinimum(1)
        self.object_id_spinbox.setMaximum(10)
        self.object_id_spinbox.setValue(1)
        self.object_id_spinbox.valueChanged.connect(self._on_object_id_change)
        annot_layout.addWidget(self.object_id_spinbox)
        
        control_layout.addWidget(annot_group)
        
        # Action buttons
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        
        generate_btn = QPushButton("Generate Mask")
        generate_btn.clicked.connect(self._generate_mask)
        action_layout.addWidget(generate_btn)
        
        clear_btn = QPushButton("Clear Points")
        clear_btn.clicked.connect(self._clear_annotations)
        action_layout.addWidget(clear_btn)
        
        propagate_btn = QPushButton("Propagate")
        propagate_btn.clicked.connect(self._propagate_to_all)
        action_layout.addWidget(propagate_btn)
        
        viz_3d_btn = QPushButton("3D View")
        viz_3d_btn.clicked.connect(self._open_3d_visualization)
        action_layout.addWidget(viz_3d_btn)
        
        surface_3d_btn = QPushButton("3D Surface")
        surface_3d_btn.clicked.connect(self._open_3d_surface_generation)
        action_layout.addWidget(surface_3d_btn)
        
        save_3d_view_btn = QPushButton("Save 3D View...")
        save_3d_view_btn.clicked.connect(self._prompt_and_save_3d_view)
        action_layout.addWidget(save_3d_view_btn)
        
        save_3d_surface_btn = QPushButton("Save 3D Surface...")
        save_3d_surface_btn.clicked.connect(self._prompt_and_save_3d_surface)
        action_layout.addWidget(save_3d_surface_btn)
        
        control_layout.addWidget(action_widget)
        
        self.main_layout.addWidget(control_group)
        
        # Canvas for displaying the slice and annotations
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        
        # Create figure and canvas for seismic display
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        canvas_layout.addWidget(self.canvas)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        
        self.main_layout.addWidget(canvas_widget, 1)  # stretch factor 1
        
        # Status bar at bottom
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        status_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready. Load a SEGY file to begin.")
        status_layout.addWidget(self.status_label, 1)  # stretch factor 1
        
        self.main_layout.addWidget(status_widget)
    
    def _on_drawing_mode_change(self, button):
        """Handle drawing mode radio button change"""
        if button == self.foreground_radio:
            self._drawing_mode = "foreground"
        else:
            self._drawing_mode = "background"
    
    def _connect_signals(self):
        """Connect Qt signals to slots"""
        # Canvas click events for annotations
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        # Connect custom signals
        self.status_updated.connect(self._update_status)
        self.progress_updated.connect(self._update_progress)
        self.error_occurred.connect(self._show_error)
        self.scale_updated.connect(self._update_scale)
        self.mask_ready.connect(self._display_mask)
    
    def _update_status(self, text):
        """Update status label"""
        self.status_label.setText(text)
    
    def _update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(int(value))
    
    def _show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText("Error occurred.")
        self.progress_bar.setValue(0)
    
    def _update_scale(self, max_value):
        """Update slice slider maximum"""
        self.slice_slider.setMaximum(max_value)
        self._load_current_slice()
    
    def _display_mask(self, mask):
        """Display mask on canvas"""
        self.display_mask(mask)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        if hasattr(self, 'segy_loader'):
            self.segy_loader.close()
        event.accept()
    
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
        self.ax.imshow(self.current_slice, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
        
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
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open SEGY File",
            "",
            "SEGY files (*.segy *.sgy);;All files (*.*)"
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
        if self._current_slice_type == "inline":
            max_slice = len(self.segy_loader.inlines) - 1
        elif self._current_slice_type == "crossline":
            max_slice = len(self.segy_loader.crosslines) - 1
        else:  # timeslice
            max_slice = len(self.segy_loader.timeslices) - 1
            
        self.slice_slider.setMaximum(max_slice)
        
        # Reset index if out of bounds
        if self._current_slice_idx > max_slice:
            self._current_slice_idx = 0
            self.slice_slider.setValue(0)
            self.slice_entry.setText("0")
            
        # Clear any existing annotations
        self._clear_annotations()
        
        # Load the current slice
        self._load_current_slice()
    
    def _on_slice_change(self, value):
        """Handle slice slider change"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
        
        self._current_slice_idx = value
        self.slice_entry.setText(str(value))
        self._load_current_slice()
    
    def _on_slice_entry_change(self):
        """Handle slice entry change"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        try:
            idx = int(self.slice_entry.text())
            max_idx = self.slice_slider.maximum()
            
            if idx < 0:
                idx = 0
            elif idx > max_idx:
                idx = max_idx
                
            self._current_slice_idx = idx
            self.slice_slider.setValue(idx)
            self._load_current_slice()
            
        except ValueError:
            # Reset to previous value
            self.slice_entry.setText(str(self._current_slice_idx))
    
    def _load_current_slice(self):
        """Load and display the current slice"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            return
            
        # Get the current slice
        idx = self._current_slice_idx
        slice_type = self._current_slice_type
        
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
            QMessageBox.critical(self, "Error", f"Failed to load slice: {str(e)}\n\n{trace}")
    
    def _generate_mask(self):
        """Generate mask from annotation points using SAM2"""
        points, _ = self._get_current_annotations() # Get points for current object
        if not points:
            QMessageBox.information(self, "Info", "Please add at least one annotation point for the current object first.")
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
            QMessageBox.information(self, "Info", "Please add at least one annotation point for the current object and generate a mask first.")
            return
            
        # Ask for confirmation
        reply = QMessageBox.question(self, "Confirm", f"This will propagate the mask for Object ID {self.current_object_id.get()} to all slices. It may take some time. Continue?")
        if reply != QMessageBox.Yes:
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
            
            # Use the full range of slices, regardless of demo or real mode
            slices_to_process = slices # slices is already list(range(slice_count))
            print(f"Processing all {len(slices_to_process)} slices for propagation.")

            # Initialize video predictor with the full slice list
            self.queue.put(("status", "Initializing video predictor..."))
            self.queue.put(("progress", 20))
            
            try:
                if use_demo_mode:
                     self.predictor.demo_mode = True # Ensure demo mode is set
                     self.predictor.init_video_predictor(slices_to_process, obj_id) # Pass full list
                else:
                     # Pass full list, remove max_slices limit from predictor if it exists there too
                     self.predictor.init_video_predictor(slices_to_process, obj_id) 
            except Exception as e:
                self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                # Handle fallback if needed (e.g., retry with demo)
                try:
                    print("Falling back to demo mode for initialization due to error.")
                    self.predictor.demo_mode = True
                    self.predictor.init_video_predictor(slices_to_process, obj_id)
                    use_demo_mode = True
                except Exception as retry_e:
                    self.queue.put(("error", f"Also failed with demo mode init: {str(retry_e)}"))
                    self.queue.put(("progress", 0))
                    return
            
            # --- Find starting frame position ---
            # The frame position is simply the index in the full list `slices_to_process`
            # which corresponds directly to the slice index `current_frame_idx`
            # No complex relative positioning needed when using the full list.
            current_frame_pos = current_frame_idx 
            if not (0 <= current_frame_pos < len(slices_to_process)):
                 # This shouldn't happen if slices_to_process covers the full range 0 to slice_count-1
                 self.queue.put(("error", f"Frame position {current_frame_pos} is out of range [0, {len(slices_to_process)-1}]."))
                 self.queue.put(("progress", 0))
                 return
            print(f"Using frame position {current_frame_pos} for starting propagation.")

            # --- Continue with adding points and propagating ---
            # obj_id = self.current_object_id.get() # Already defined earlier

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
        self.ax.imshow(self.current_slice, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
        
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
                    QMessageBox.critical(self, "Error", msg[1])
                    self.status_text.set("Error occurred.")
                    self.progress_var.set(0)
                elif cmd == "update_scale":
                    self.slice_slider.setMaximum(msg[1])
                    self._load_current_slice()
                elif cmd == "display_mask":
                    self.display_mask(msg[1])
                    
                self.queue.task_done()
        except queue.Empty:
            pass
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About",
            "Seismic Interpretation with SAM2\n\n"
            "An application for seismic interpretation using Segment Anything Model 2 (SAM2).\n\n"
            "© 2025"
        )
    
    def _show_instructions(self):
        """Show instructions dialog"""
        QMessageBox.information(
            self,
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
    

    def _open_3d_visualization(self):
        """Open a 3D visualization window using PyVista"""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            QMessageBox.information(self, "Info", "Please load a SEGY file first.")
            return
            
        if not PYVISTA_AVAILABLE:
            QMessageBox.critical(self, "Error", "PyVista is not available. Please install it with: pip install pyvista")
            return
            
        # Warn user about potential memory issues
        if np.prod(self.segy_loader.data.shape) > 100000000:  # If volume is larger than ~100M voxels
            reply = QMessageBox.question(self, "Warning", 
                                      "The seismic volume is very large and may cause memory issues. "
                                      "Continue with visualization?")
            if reply != QMessageBox.Yes:
                return
        
        # Ask how many slices to visualize
        try:
            num_slices, ok = QInputDialog.getInt(self, "Input", "Enter number of slices to visualize (5-100):", 
                                               20, 5, 100)
            if not ok:  # User canceled
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
                reply = QMessageBox.question(self, "Propagation Needed", 
                                     "Masks have not been fully propagated to all slices. "
                                     f"Propagate masks for Object {obj_id} before visualization? (Recommended)")
                if reply == QMessageBox.Yes:
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
            
            # Use the full range of slices, regardless of demo or real mode
            slices_to_process = slices # slices is already list(range(slice_count))
            print(f"Processing all {len(slices_to_process)} slices for propagation (visualization).")

            # Initialize video predictor with the full slice list
            self.queue.put(("status", "Initializing video predictor..."))
            self.queue.put(("progress", 20))

            try:
                if use_demo_mode:
                    self.predictor.demo_mode = True # Ensure demo mode is set
                    self.predictor.init_video_predictor(slices_to_process, obj_id) # Pass full list
                else:
                    # Pass full list, remove max_slices limit from predictor if it exists there too
                    self.predictor.init_video_predictor(slices_to_process, obj_id)
            except Exception as e:
                self.queue.put(("error", f"Error initializing video predictor: {str(e)}"))
                # Handle fallback if needed (e.g., retry with demo)
                try:
                     print("Falling back to demo mode for initialization due to error.")
                     self.predictor.demo_mode = True
                     self.predictor.init_video_predictor(slices_to_process, obj_id)
                     use_demo_mode = True
                except Exception as retry_e:
                     self.queue.put(("error", f"Also failed with demo mode init: {str(retry_e)}"))
                     self.queue.put(("progress", 0))
                     return

            # --- Find starting frame position ---
            current_frame_pos = current_frame_idx
            if not (0 <= current_frame_pos < len(slices_to_process)):
                self.queue.put(("error", f"Frame position {current_frame_pos} is out of range [0, {len(slices_to_process)-1}]."))
                self.queue.put(("progress", 0))
                return
            print(f"Using frame position {current_frame_pos} for starting propagation (visualization).")
            
            # --- Continue with adding points and propagating ---
            # obj_id = self.current_object_id.get() # Already defined earlier

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
                # plotter.add_axes()  # REMOVED
                # plotter.show_grid() # REMOVED
                # plotter.add_legend() # REMOVED (optional, keep if you want object ID colors identified)
                pass # Keep try/except block structure if other finalization needed later
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
            QMessageBox.information(self, "Info", "Please load a SEGY file first.")
            return
            
        if not PYVISTA_AVAILABLE:
            QMessageBox.critical(self, "Error", "PyVista is not available. Please install it with: pip install pyvista")
            return
            
        # Check if any masks exist for ANY object ID
        has_masks = False
        if hasattr(self.predictor, 'has_masks_for_object') and callable(self.predictor.has_masks_for_object):
            # Get all known object IDs
            all_known_obj_ids = set(self.object_annotations.keys())
            if not self.predictor.demo_mode:
                all_known_obj_ids.update(self.predictor.inference_state.keys())
            else:
                all_known_obj_ids.update(self.predictor.demo_state.keys())
                
            # Check if any known object has masks
            for obj_id_check in all_known_obj_ids:
                if self.predictor.has_masks_for_object(obj_id_check):
                    has_masks = True
                    break # Found masks for at least one object
        else:
             # Fallback check if the method somehow doesn't exist
             print("Warning: predictor.has_masks_for_object not found. Limited mask checking.")
             # This fallback is less accurate now with multiple objects
             if hasattr(self.predictor, 'inference_state') and self.predictor.inference_state:
                 has_masks = True # Assume masks exist if state is populated
             elif hasattr(self.predictor, 'demo_state') and self.predictor.demo_state:
                 has_masks = True # Assume masks exist if demo state is populated

        if not has_masks:
            reply = QMessageBox.question(self, "No Masks Detected", 
                                     "No masks have been generated for any object. " # Updated message
                                     "You need to create and propagate masks before generating surfaces. "
                                     "Do you want to generate a surface anyway for the current object (might be empty)?")
            if reply != QMessageBox.Yes:
                return
        
        # Ask the user for the surface generation method
        use_triangulation = True
        if LOOPSTRUCTURAL_AVAILABLE:
            reply = QMessageBox.question(self, "Surface Generation Method", 
                               "LoopStructural is available. Would you like to use it for advanced surface generation?\n\n"
                               "Yes: Use LoopStructural (better for sparse data)\n"
                               "No: Use simple triangulation (more reliable but less smooth)")
            if reply == QMessageBox.Yes:
                use_triangulation = False
        
        # Ask for surface generation resolution
        try:
            num_slices, ok = QInputDialog.getInt(self, "Input", "Number of slices to sample for surface (5-100):", 
                                               20, 5, 100)
            if not ok:
                num_slices = 20  # Default if canceled
                
            if use_triangulation:
                # For triangulation, ask about point reduction
                max_points, ok = QInputDialog.getInt(self, "Input", "Maximum number of points to use (100-5000):", 
                                                   1000, 100, 5000)
                if not ok:
                    max_points = 1000  # Default if canceled
                smoothing = 0  # Not used for triangulation
            else:
                # For LoopStructural, ask about smoothing
                smoothing, ok = QInputDialog.getInt(self, "Input", "Surface smoothing factor (1-50):", 
                                                  10, 1, 50)
                if not ok:
                    smoothing = 10  # Default if canceled
                max_points = 500  # Default for LoopStructural
        except:
            num_slices = 20
            max_points = 1000
            smoothing = 10
        
        # Start the surface generation in a separate thread FOR ALL OBJECTS
        self.status_text.set(f"Preparing surface generation for ALL Objects...") # Update status text
        self.progress_var.set(10)
        
        # REMOVE obj_id from the target function call
        threading.Thread(target=lambda: self._generate_3d_surface(num_slices, smoothing, use_triangulation, max_points), 
                        daemon=True).start()

    def _generate_3d_surface(self, num_slices=20, smoothing=10, use_triangulation=False, max_points=1000): # REMOVED obj_id parameter
        """Generate a 3D surface from mask points for ALL objects using LoopStructural or triangulation"""
        try:
            # Get all known object IDs
            all_known_obj_ids = set(self.object_annotations.keys())
            if self.predictor:
                if not self.predictor.demo_mode:
                    all_known_obj_ids.update(self.predictor.inference_state.keys())
                else:
                    all_known_obj_ids.update(self.predictor.demo_state.keys())
            
            if not all_known_obj_ids:
                 self.queue.put(("error", "No objects defined. Annotate or propagate masks first."))
                 self.queue.put(("progress", 0))
                 return

            print(f"Generating surfaces for Object IDs: {list(all_known_obj_ids)}")
            self.queue.put(("status", f"Generating 3D surfaces for {len(all_known_obj_ids)} objects from {num_slices} slices..."))
            self.queue.put(("progress", 20))
            
            # Get the seismic data dimensions
            seismic_volume = self.segy_loader.data
            ni, nj, nk = seismic_volume.shape
            
            # Current slice type
            slice_type = self.current_slice_type.get()
            
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
            
            # --- Create plotter BEFORE the loop ---
            plotter = pv.Plotter(notebook=False)
            plotter.set_background("white")
            
            # Add seismic data slices to context (only once)
            self.queue.put(("status", "Adding seismic slices for context..."))
            self._add_seismic_slices_to_plot(plotter, slice_type, seismic_volume, indices)
            
            colors = plt.get_cmap('tab10').colors
            
            # --- Loop through each known object ID ---
            total_progress_steps = len(all_known_obj_ids)
            current_progress = 20 # Initial progress after setup

            for i, current_obj_id in enumerate(sorted(list(all_known_obj_ids))):
                 obj_progress_start = current_progress
                 obj_progress_share = (90 - 20) / total_progress_steps # Share remaining progress

                 self.queue.put(("status", f"Processing Object ID: {current_obj_id} ({i+1}/{total_progress_steps})"))
                 self.queue.put(("progress", obj_progress_start))

                 # Check if we need to propagate masks first for this specific object
                 self._ensure_masks_propagated(indices, current_obj_id) 
                 
                 # Collect all mask points for surface generation for this specific object
                 all_points_for_obj = []
                 
                 # Process slices to collect points where masks exist for this object
                 for j, idx in enumerate(indices):
                     # Update progress within object processing
                     progress_within_obj = int( (obj_progress_share * 0.5) * (j / len(indices)) ) # 50% of share for point collection
                     self.queue.put(("progress", obj_progress_start + progress_within_obj))

                     try:
                         # Get mask for this slice and object
                         mask = self.predictor.get_mask_for_frame(idx, current_obj_id) 
                         
                         if mask is not None and np.any(mask):
                             # Process points based on slice type (same logic as before)
                             if slice_type == "inline":
                                 mask_t = mask.T
                                 ys, zs = np.where(mask_t)
                                 for y, z in zip(ys, zs):
                                     all_points_for_obj.append([idx, y, z/8])
                             elif slice_type == "crossline":
                                 mask_t = mask.T
                                 xs, zs = np.where(mask_t)
                                 for x, z in zip(xs, zs):
                                     all_points_for_obj.append([x, idx, z/8])
                             else:
                                 xs, ys = np.where(mask)
                                 for x, y in zip(xs, ys):
                                     all_points_for_obj.append([x, y, idx/8])
                     
                     except Exception as e:
                         print(f"Error processing mask for frame {idx}, Object {current_obj_id}: {e}")
                 
                 # Check if we have enough points for this object
                 if len(all_points_for_obj) < 10:
                     print(f"Not enough mask points found for Object {current_obj_id}. Skipping surface generation for this object.")
                     # Update progress to reflect skipping
                     current_progress += obj_progress_share 
                     self.queue.put(("progress", current_progress))
                     continue # Skip to the next object ID
                     
                 self.queue.put(("status", f"Object {current_obj_id}: Collected {len(all_points_for_obj)} points for surface generation"))
                 # Update progress after point collection
                 current_progress = obj_progress_start + (obj_progress_share * 0.5)
                 self.queue.put(("progress", current_progress )) 
                 
                 # Convert points to numpy array for this object
                 points_array_for_obj = np.array(all_points_for_obj)
                 
                 # Determine color for this object
                 obj_color = colors[ (current_obj_id - 1) % len(colors) ]
                 
                 # Add scatter plot of the points for this object
                 point_cloud = pv.PolyData(points_array_for_obj)
                 plotter.add_mesh(point_cloud, color=obj_color, point_size=3, 
                                  render_points_as_spheres=True, 
                                  label=f"Object {current_obj_id} Points") # Add label
                 
                 # Thin the points if needed for this object
                 if len(points_array_for_obj) > max_points:
                     self.queue.put(("status", f"Object {current_obj_id}: Reducing points from {len(points_array_for_obj)} to {max_points}..."))
                     indices_subset = np.random.choice(len(points_array_for_obj), max_points, replace=False)
                     subset_points = points_array_for_obj[indices_subset]
                 else:
                     subset_points = points_array_for_obj
                     
                 # Generate surface for this object using the chosen method
                 surface_generated = False
                 if use_triangulation:
                     self.queue.put(("status", f"Object {current_obj_id}: Creating surface using triangulation..."))
                     try:
                         self._create_triangulation_surface(subset_points, plotter, obj_color) 
                         surface_generated = True
                     except Exception as e:
                          print(f"Triangulation failed for Object {current_obj_id}: {e}")
                 else: # Use LoopStructural
                     self.queue.put(("status", f"Object {current_obj_id}: Creating surface using LoopStructural..."))
                     try:
                         self._create_loopstructural_surface(subset_points, plotter, smoothing, ni, nj, nk, obj_color) 
                         surface_generated = True
                     except ImportError:
                         self.queue.put(("status", f"Object {current_obj_id}: LoopStructural failed, falling back to triangulation..."))
                         try:
                              self._create_triangulation_surface(subset_points, plotter, obj_color)
                              surface_generated = True
                         except Exception as e:
                              print(f"Fallback triangulation failed for Object {current_obj_id}: {e}")
                     except Exception as e_ls:
                          print(f"LoopStructural failed for Object {current_obj_id}: {e_ls}")
                          # Optionally try triangulation as fallback even if LoopStructural exists but failed
                          reply = QMessageBox.question(self, "LoopStructural Failed", 
                                                f"LoopStructural surface generation failed for Object {current_obj_id}. Try triangulation instead?")
                          if reply == QMessageBox.Yes:
                              try:
                                  self._create_triangulation_surface(subset_points, plotter, obj_color)
                                  surface_generated = True
                              except Exception as e_fallback:
                                  print(f"Fallback triangulation also failed: {e_fallback}")
                 
                 # Update progress after surface generation attempt for this object
                 current_progress += (obj_progress_share * 0.5) # Add remaining share
                 self.queue.put(("progress", current_progress ))

                 if not surface_generated:
                      self.queue.put(("status", f"Object {current_obj_id}: Failed to generate surface."))


            # --- Finalize plotter AFTER the loop ---
            # plotter.add_axes() # REMOVED
            # plotter.show_grid() # REMOVED
            # plotter.add_legend() # REMOVED (optional, keep if you want object ID colors identified)
            
            # Show the plotter
            self.queue.put(("status", "Displaying 3D surfaces..."))
            self.queue.put(("progress", 95))
            
            try:
                plotter.show(title="3D Surfaces from Masks (All Objects)") # Update title
                self.queue.put(("status", "3D surface visualization complete for all objects"))
                self.queue.put(("progress", 100))
            except Exception as e:
                self.queue.put(("error", f"Error displaying 3D surface: {str(e)}"))
                self.queue.put(("progress", 0))
                
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error in multi-surface generation: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))

    # --- Helper function _create_triangulation_surface ---
    # Modify plotter.add_mesh call to include a label
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
                        
                    plotter.add_mesh(mesh, color=color, opacity=0.5, label=f"Surface (Triangulated Planar)") # Add label
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
                        
                        plotter.add_mesh(surf, color=color, opacity=0.5, label=f"Surface (Triangulated 3D)") # Add label
                        self.queue.put(("status", "Created triangulated surface"))
                    else:
                        raise ValueError("Surface generation failed with all methods")
                        
                except Exception as e:
                    self.queue.put(("error", f"Error in 3D triangulation: {str(e)}"))
                    # Create a convex hull as a last resort
                    try:
                        cloud = pv.PolyData(points)
                        hull = cloud.delaunay_3d().extract_surface()
                        plotter.add_mesh(hull, color=color, opacity=0.5, label=f"Surface (Convex Hull Fallback)") # Add label
                        self.queue.put(("status", "Created convex hull surface (fallback)"))
                    except:
                        self.queue.put(("status", "Unable to create surface, showing point cloud only"))
                        
        except Exception as e:
            self.queue.put(("error", f"Triangulation failed: {str(e)}"))
            self.queue.put(("status", "Unable to create surface, showing point cloud only"))

    # --- Helper function _create_loopstructural_surface ---
    # Modify plotter.add_mesh call to include a label
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
                plotter.add_mesh(surf, color=color, opacity=0.5, label=f"Surface (LoopStructural)") # Add label
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
        
        # Count how many have masks for all objects
        mask_count = 0
        for idx in check_indices:
            try:
                masks = []
                for obj_id in self.object_annotations.keys():
                    mask = self.predictor.get_mask_for_frame(idx, obj_id)
                    if mask is not None and np.any(mask):
                        masks.append(mask)
                
                if not masks:
                    self.queue.put(("status", f"No masks found for slice {idx}"))
                    continue
                
                # Check if masks are consistent across objects
                consistent = all(np.array_equal(masks[0], mask) for mask in masks)
                if consistent:
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
            # --- FIX: Add the missing obj_id argument ---
            self.predictor.set_demo_context(current_frame_idx, obj_id, points, labels)
            
            # Force demo mode for more reliable propagation
            use_demo_mode = True
            
            # Initialize video predictor with all necessary indices
            # Using indices directly, not slice_indices from a potentially limited set
            all_slice_indices = list(range(len(self.segy_loader.inlines if slice_type == "inline" 
                                              else self.segy_loader.crosslines if slice_type == "crossline" 
                                              else self.segy_loader.timeslices)))
            try:
                if use_demo_mode:
                    self.predictor.demo_mode = True
                    self.predictor.init_video_predictor(all_slice_indices, obj_id) # Pass obj_id and full list
                else:
                    self.predictor.init_video_predictor(all_slice_indices, obj_id) # Pass obj_id and full list
            except Exception as e:
                print(f"Error initializing video predictor: {e}")
                # Optionally add fallback to demo mode here if needed
                return
            
            # Add points to current frame if they exist for this object
            if points and len(points) > 0:
                try:
                    # Determine frame position (should be the index itself if using full list)
                    frame_pos = current_frame_idx 
                    if not (0 <= frame_pos < len(all_slice_indices)):
                         print(f"Error: Frame position {frame_pos} out of range during ensure_masks_propagated")
                         return # Cannot add points if position is invalid

                    # Add points to frame
                    self.predictor.add_point_to_video(frame_pos, obj_id, points, labels) 
                except Exception as e:
                    print(f"Error adding points to video: {e}")
                    # Continue propagation even if adding points fails? Or return? Decide based on desired behavior.
                    # For now, we'll let it continue to try propagation.
                    pass # Or return if adding points is critical

            # Propagate masks for this object (forward and backward)
            try:
                print(f"Propagating forward for Object {obj_id} in ensure_masks...")
                self.predictor.propagate_masks(obj_id=obj_id, start_frame_idx=frame_pos, reverse=False) 
                print(f"Propagating backward for Object {obj_id} in ensure_masks...")
                self.predictor.propagate_masks(obj_id=obj_id, start_frame_idx=frame_pos, reverse=True) 
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

    def _on_object_id_change(self, value=None):
        """Handle change in object ID."""
        if value is not None:
            self._current_object_id = value
        print(f"Object ID changed to: {self.current_object_id.get()}")
        # Reload the current slice view to show annotations for the new object ID
        self._load_current_slice()

    def _prompt_and_save_3d_view(self):
        """Prompt user for filename and format, then save the 3D view."""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            QMessageBox.information(self, "Info", "Please load a SEGY file first.")
            return
        if not PYVISTA_AVAILABLE:
            QMessageBox.critical(self, "Error", "PyVista is not available.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D View As...",
            "",
            "PNG Image (*.png);;SVG Image (Experimental) (*.svg);;All Files (*.*)"
        )

        if not save_path:
            return # User cancelled

        file_ext = os.path.splitext(save_path)[1].lower()
        save_format = "svg" if file_ext == ".svg" else "png"
        
        # Get the number of slices used in the last interactive view (or default)
        # For simplicity, we'll just use a default here. Could store last used value if needed.
        num_slices = 20 

        self.status_text.set(f"Saving 3D view to {os.path.basename(save_path)}...")
        self.progress_var.set(10)
        
        # Start saving in a thread
        threading.Thread(target=self._save_3d_view_thread, 
                         args=(save_path, save_format, num_slices), 
                         daemon=True).start()

    def _prompt_and_save_3d_surface(self):
        """Prompt user for filename and format, then save the 3D surface."""
        if not hasattr(self.segy_loader, 'data') or self.segy_loader.data is None:
            QMessageBox.information(self, "Info", "Please load a SEGY file first.")
            return
        if not PYVISTA_AVAILABLE:
            QMessageBox.critical(self, "Error", "PyVista is not available.")
            return
            
        # Check if any masks exist (reuse logic from _open_3d_surface_generation)
        has_masks = False
        if hasattr(self.predictor, 'has_masks_for_object') and callable(self.predictor.has_masks_for_object):
            all_known_obj_ids = set(self.object_annotations.keys())
            if not self.predictor.demo_mode: all_known_obj_ids.update(self.predictor.inference_state.keys())
            else: all_known_obj_ids.update(self.predictor.demo_state.keys())
            for obj_id_check in all_known_obj_ids:
                if self.predictor.has_masks_for_object(obj_id_check): has_masks = True; break
        if not has_masks:
             reply = QMessageBox.question(self, "No Masks", "No masks found to generate surfaces. Save an empty plot?")
             if reply != QMessageBox.Yes:
                 return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Surface As...",
            "",
            "PNG Image (*.png);;SVG Image (Experimental) (*.svg);;All Files (*.*)"
        )

        if not save_path:
            return # User cancelled

        file_ext = os.path.splitext(save_path)[1].lower()
        save_format = "svg" if file_ext == ".svg" else "png"

        # Use default parameters for surface generation for saving
        # Could potentially ask the user again or store last used values
        num_slices = 20
        smoothing = 10
        use_triangulation = not LOOPSTRUCTURAL_AVAILABLE # Default to triangulation if LS not available
        max_points = 1000

        self.status_text.set(f"Saving 3D surface to {os.path.basename(save_path)}...")
        self.progress_var.set(10)

        # Start saving in a thread
        threading.Thread(target=self._save_3d_surface_thread, 
                         args=(save_path, save_format, num_slices, smoothing, use_triangulation, max_points), 
                         daemon=True).start()

    # --- Add Thread Methods for Saving ---
    def _save_3d_view_thread(self, save_path, save_format, num_slices):
        """Generates the 3D view off-screen and saves it."""
        try:
            self.queue.put(("status", f"Generating 3D view for saving ({save_format})..."))
            self.queue.put(("progress", 30))
            
            # --- Replicate the core logic of _create_3d_visualization ---
            seismic_volume = self.segy_loader.data
            ni, nj, nk = seismic_volume.shape
            slice_type = self.current_slice_type.get()
            
            # Initialize OFF-SCREEN plotter
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 800]) # Use a fixed size for consistency
            plotter.set_background("white")
            
            ds_i = max(1, ni // 150); ds_j = max(1, nj // 150); ds_k = max(1, nk // 150)
            
            if slice_type == "inline": indices = np.linspace(0, ni-1, num_slices, dtype=int)
            elif slice_type == "crossline": indices = np.linspace(0, nj-1, num_slices, dtype=int)
            else: indices = np.linspace(0, nk-1, num_slices, dtype=int)
                
            all_known_obj_ids = set(self.object_annotations.keys())
            if self.predictor:
                if not self.predictor.demo_mode: all_known_obj_ids.update(self.predictor.inference_state.keys())
                else: all_known_obj_ids.update(self.predictor.demo_state.keys())
            colors = plt.get_cmap('tab10').colors

            # --- Loop to add slices and masks (Simplified from _create_3d_visualization) ---
            for i, idx in enumerate(indices):
                 self.queue.put(("progress", 30 + int(60 * i / num_slices)))
                 # Add seismic slice data (copy relevant parts from _create_3d_visualization)
                 slice_data = None; grid = None
                 # ... [Logic to get slice_data and points/grid for the slice type] ...
                 if slice_type == "inline":
                      slice_data = seismic_volume[idx, ::ds_j, ::ds_k]; s_nj, s_nk = slice_data.shape
                      x_coords = np.ones(s_nj * s_nk) * idx; y_coords = np.repeat(np.arange(0, s_nj * ds_j, ds_j), s_nk); z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_nj) / 8
                      points = np.column_stack((x_coords, y_coords, z_coords)); grid = pv.PolyData(points)
                 elif slice_type == "crossline":
                      slice_data = seismic_volume[::ds_i, idx, ::ds_k]; s_ni, s_nk = slice_data.shape
                      x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nk); y_coords = np.ones(s_ni * s_nk) * idx; z_coords = np.tile(np.arange(0, s_nk * ds_k, ds_k), s_ni) / 8
                      points = np.column_stack((x_coords, y_coords, z_coords)); grid = pv.PolyData(points)
                 else: # timeslice
                      slice_data = seismic_volume[::ds_i, ::ds_j, idx]; s_ni, s_nj = slice_data.shape
                      x_coords = np.repeat(np.arange(0, s_ni * ds_i, ds_i), s_nj); y_coords = np.tile(np.arange(0, s_nj * ds_j, ds_j), s_ni); z_coords = np.ones(s_ni * s_nj) * idx / 8
                      points = np.column_stack((x_coords, y_coords, z_coords)); grid = pv.PolyData(points)

                 if grid is not None and slice_data is not None:
                     vmin, vmax = np.percentile(slice_data, [5, 95]); norm_data = np.clip(slice_data, vmin, vmax)
                     norm_data = (norm_data - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-6 else np.zeros_like(slice_data)
                     grid.point_data["intensity"] = norm_data.flatten()
                     plotter.add_mesh(grid, cmap="seismic", point_size=3, render_points_as_spheres=True, scalars="intensity")

                 # Add masks for all objects (copy relevant parts from _create_3d_visualization)
                 for current_obj_id_to_display in all_known_obj_ids:
                     try:
                         mask = self.predictor.get_mask_for_frame(idx, current_obj_id_to_display) 
                         if mask is not None and np.any(mask):
                             obj_color = colors[ (current_obj_id_to_display - 1) % len(colors) ]
                             mask_points = []
                             # ... [Logic to get mask_points based on slice type, identical to _create_3d_visualization] ...
                             if slice_type == "inline":
                                 mask_t = mask.T
                                 if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                     mask_ds = mask_t[::ds_j, ::ds_k]
                                     if mask_ds.shape == slice_data.shape:
                                         s_nj, s_nk = slice_data.shape # Get dimensions again here
                                         for j in range(s_nj):
                                             for k in range(s_nk):
                                                 if mask_ds[j, k]: mask_points.append([idx, j*ds_j, k*ds_k/8])
                             elif slice_type == "crossline":
                                 mask_t = mask.T
                                 if mask_t.shape[0] > 1 and mask_t.shape[1] > 1:
                                     mask_ds = mask_t[::ds_i, ::ds_k]
                                     if mask_ds.shape == slice_data.shape:
                                         s_ni, s_nk = slice_data.shape # Get dimensions again here
                                         for i_ in range(s_ni):
                                             for k in range(s_nk):
                                                 if mask_ds[i_, k]: mask_points.append([i_*ds_i, idx, k*ds_k/8])
                             else: # timeslice
                                 if mask.shape[0] > 1 and mask.shape[1] > 1:
                                     mask_ds = mask[::ds_i, ::ds_j]
                                     if mask_ds.shape == slice_data.shape:
                                         s_ni, s_nj = slice_data.shape # Get dimensions again here
                                         for i_ in range(s_ni):
                                             for j in range(s_nj):
                                                 if mask_ds[i_, j]: mask_points.append([i_*ds_i, j*ds_j, idx/8])

                             if mask_points:
                                 mask_poly = pv.PolyData(np.array(mask_points))
                                 plotter.add_mesh(mask_poly, color=obj_color, point_size=5, render_points_as_spheres=True) 
                     except Exception as mask_err:
                         print(f"Error processing mask for Object {current_obj_id_to_display} on slice {idx} during save: {mask_err}")
                         
            # --- Save the plot ---
            self.queue.put(("status", f"Saving plot to {os.path.basename(save_path)}..."))
            self.queue.put(("progress", 95))
            
            if save_format == "png":
                 plotter.screenshot(save_path)
            elif save_format == "svg":
                 try:
                     # SVG export might be limited for complex scenes
                     plotter.save_graphic(save_path)
                     print("Note: SVG export quality may vary depending on scene complexity.")
                 except Exception as svg_err:
                      self.queue.put(("error", f"Failed to save as SVG: {svg_err}. Try PNG instead."))
                      self.queue.put(("progress", 0))
                      plotter.close() # Close plotter to free resources
                      return
            
            plotter.close() # Close plotter to free resources
            self.queue.put(("status", f"3D View saved successfully to {os.path.basename(save_path)}."))
            self.queue.put(("progress", 100))

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error saving 3D view: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
            if 'plotter' in locals() and plotter: plotter.close()

    def _save_3d_surface_thread(self, save_path, save_format, num_slices, smoothing, use_triangulation, max_points):
        """Generates the 3D surface(s) off-screen and saves it."""
        try:
            self.queue.put(("status", f"Generating 3D surface(s) for saving ({save_format})..."))
            self.queue.put(("progress", 20))

            # --- Replicate the core logic of _generate_3d_surface ---
            all_known_obj_ids = set(self.object_annotations.keys())
            if self.predictor:
                if not self.predictor.demo_mode: all_known_obj_ids.update(self.predictor.inference_state.keys())
                else: all_known_obj_ids.update(self.predictor.demo_state.keys())
            
            if not all_known_obj_ids: self.queue.put(("status", "No objects found for surface saving.")); return

            seismic_volume = self.segy_loader.data
            ni, nj, nk = seismic_volume.shape
            slice_type = self.current_slice_type.get()
            
            if slice_type == "inline": indices = np.linspace(0, ni-1, num_slices, dtype=int)
            elif slice_type == "crossline": indices = np.linspace(0, nj-1, num_slices, dtype=int)
            else: indices = np.linspace(0, nk-1, num_slices, dtype=int)
            
            # Initialize OFF-SCREEN plotter
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
            plotter.set_background("white")
            
            # Add seismic slices (call helper, which adds to plotter)
            self._add_seismic_slices_to_plot(plotter, slice_type, seismic_volume, indices)
            
            colors = plt.get_cmap('tab10').colors
            total_progress_steps = len(all_known_obj_ids)
            current_progress = 20

            # Loop through objects (Simplified from _generate_3d_surface)
            for i, current_obj_id in enumerate(sorted(list(all_known_obj_ids))):
                 obj_progress_start = current_progress
                 obj_progress_share = (90 - 20) / total_progress_steps
                 self.queue.put(("status", f"Processing Object {current_obj_id} for save..."))
                 self.queue.put(("progress", obj_progress_start))

                 # Collect points (copy relevant parts from _generate_3d_surface)
                 all_points_for_obj = []
                 # ... [Logic to get mask and append points to all_points_for_obj, identical to _generate_3d_surface] ...
                 for j, idx in enumerate(indices):
                     progress_within_obj = int( (obj_progress_share * 0.5) * (j / len(indices)) )
                     self.queue.put(("progress", obj_progress_start + progress_within_obj))
                     try:
                         mask = self.predictor.get_mask_for_frame(idx, current_obj_id) 
                         if mask is not None and np.any(mask):
                             if slice_type == "inline":
                                 mask_t = mask.T; ys, zs = np.where(mask_t)
                                 for y, z in zip(ys, zs): all_points_for_obj.append([idx, y, z/8])
                             elif slice_type == "crossline":
                                 mask_t = mask.T; xs, zs = np.where(mask_t)
                                 for x, z in zip(xs, zs): all_points_for_obj.append([x, idx, z/8])
                             else:
                                 xs, ys = np.where(mask)
                                 for x, y in zip(xs, ys): all_points_for_obj.append([x, y, idx/8])
                     except Exception as e:
                         print(f"Error processing mask frame {idx}, Obj {current_obj_id} during save: {e}")

                 if len(all_points_for_obj) < 10: continue # Skip object if no points
                 
                 current_progress = obj_progress_start + (obj_progress_share * 0.5)
                 self.queue.put(("progress", current_progress )) 

                 points_array_for_obj = np.array(all_points_for_obj)
                 obj_color = colors[ (current_obj_id - 1) % len(colors) ]
                 
                 # Add points cloud
                 # point_cloud = pv.PolyData(points_array_for_obj) # Optionally add points
                 # plotter.add_mesh(point_cloud, color=obj_color, point_size=3, render_points_as_spheres=True)

                 # Thin points
                 if len(points_array_for_obj) > max_points:
                     indices_subset = np.random.choice(len(points_array_for_obj), max_points, replace=False)
                     subset_points = points_array_for_obj[indices_subset]
                 else: subset_points = points_array_for_obj

                 # Generate surface (call helpers, which add to plotter)
                 try:
                     if use_triangulation:
                         self._create_triangulation_surface(subset_points, plotter, obj_color)
                     else:
                         self._create_loopstructural_surface(subset_points, plotter, smoothing, ni, nj, nk, obj_color)
                 except Exception as surface_err:
                      print(f"Could not generate surface for Object {current_obj_id} during save: {surface_err}")
                      # Optionally add fallback here if needed

                 current_progress += (obj_progress_share * 0.5)
                 self.queue.put(("progress", current_progress ))
            
            # --- Save the plot ---
            self.queue.put(("status", f"Saving plot to {os.path.basename(save_path)}..."))
            self.queue.put(("progress", 95))

            if save_format == "png":
                 plotter.screenshot(save_path)
            elif save_format == "svg":
                 try:
                     # SVG export might be limited
                     plotter.save_graphic(save_path)
                     print("Note: SVG export quality may vary depending on scene complexity.")
                 except Exception as svg_err:
                      self.queue.put(("error", f"Failed to save as SVG: {svg_err}. Try PNG instead."))
                      self.queue.put(("progress", 0))
                      plotter.close()
                      return

            plotter.close() # Close plotter to free resources
            self.queue.put(("status", f"3D Surface saved successfully to {os.path.basename(save_path)}."))
            self.queue.put(("progress", 100))

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            self.queue.put(("error", f"Error saving 3D surface: {str(e)}\n\n{trace}"))
            self.queue.put(("progress", 0))
            if 'plotter' in locals() and plotter: plotter.close()
            

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
    
    # Set Qt application attributes before creating QApplication (skip deprecated ones for Qt6)
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Deprecated in Qt6
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)     # Deprecated in Qt6
    
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Seismic Interpretation App")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("Seismic Analysis")
        
        # Create the main window
        window = SeismicApp(demo_mode=demo_mode, model_id=model_id)
        window.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: try without high DPI scaling
        try:
            print("Trying without high DPI scaling...")
            app = QApplication(sys.argv)
            window = SeismicApp(demo_mode=demo_mode, model_id=model_id)
            window.show()
            sys.exit(app.exec())
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    main()