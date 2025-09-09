import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import sys

# Add parent directory to path for importing SAM2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.build_sam import build_sam2_hf
    SAM2_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 modules not available. Running in demo mode.")
    SAM2_AVAILABLE = False

class SeismicPredictor:
    def __init__(self, model_id="facebook/sam2-hiera-base-plus", demo_mode=False):
        """Initialize the Seismic Predictor with SAM2 model
        
        Args:
            model_id: HuggingFace model ID for SAM2
                Recommended models: 
                - "facebook/sam2-hiera-base-plus" (base+ model, good balance of speed and accuracy)
                - "facebook/sam2-hiera-large" (large model, most accurate but slower)
                - "facebook/sam2-hiera-small" (small model, faster)
                - "facebook/sam2-hiera-tiny" (tiny model, fastest but less accurate)
                - "facebook/sam2.1-hiera-base-plus" (newer base+ model with improved accuracy)
                - "facebook/sam2.1-hiera-large" (newer large model, most accurate)
            demo_mode: If True, run in demo mode without loading the actual model
        """
        self.model_id = model_id
        self.demo_mode = demo_mode or not SAM2_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Demo mode: {self.demo_mode}")
        
        # Load SAM2 model from pretrained
        self.sam_model = None
        self.image_predictor = None
        self.video_predictor = None
        
        # Seismic data
        self.seismic_volume = None
        self.current_slice_type = None  # "inline", "crossline", or "timeslice"
        self.current_slice_idx = None
        self.current_slice = None
        
        # Prediction results per object ID
        self.object_masks = {}  # Stores masks per object ID: {obj_id: {slice_key: mask}}
        self.inference_state = {} # Stores video inference state per object ID: {obj_id: state}
        self.demo_state = {} # Stores demo mode state per object ID: {obj_id: state_info}
        self.horizon_paths = {} # Stores per-object refined horizon per slice: {obj_id: {slice_key: {"x": [], "y": [], "confidence": float}}}
        self.slice_maps = {} # Maps selected slice indices to frame indices per object for video predictor
        
    def load_model(self):
        """Load the SAM2 model"""
        if self.demo_mode:
            print("Running in demo mode - not loading actual model")
            return True
            
        print(f"Loading SAM2 model: {self.model_id}")
        print(f"This may take a few moments depending on the model size...")
        
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
                gpu_memory = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                print(f"Using CUDA - {gpu_info}, {gpu_memory}")
            else:
                print("CUDA not available, using CPU. This may be slow for large models.")
            
            # Make sure the model ID is valid
            supported_models = [
                "facebook/sam2-hiera-tiny",
                "facebook/sam2-hiera-small",
                "facebook/sam2-hiera-base-plus",
                "facebook/sam2-hiera-large",
                "facebook/sam2.1-hiera-tiny",
                "facebook/sam2.1-hiera-small",
                "facebook/sam2.1-hiera-base-plus",
                "facebook/sam2.1-hiera-large"
            ]
            
            if self.model_id not in supported_models:
                print(f"Warning: Model ID '{self.model_id}' may not be supported.")
                print(f"Supported models: {', '.join(supported_models)}")
                
            # Load the model with better error handling
            print("Building SAM2 model...")
            try:
                self.sam_model = build_sam2_hf(self.model_id)
                self.sam_model.to(self.device)
            except Exception as model_error:
                print(f"Error building SAM2 model: {model_error}")
                print("Trying alternative loading method...")
                
                try:
                    # Try with specific device parameter
                    self.sam_model = build_sam2_hf(self.model_id, device=self.device)
                except Exception as alt_error:
                    print(f"Alternative loading also failed: {alt_error}")
                    print("Falling back to demo mode")
                    self.demo_mode = True
                    return False
            
            # Initialize image predictor
            print("Initializing image predictor...")
            try:
                self.image_predictor = SAM2ImagePredictor(self.sam_model)
            except Exception as img_error:
                print(f"Error initializing image predictor: {img_error}")
                self.demo_mode = True
                return False
            
            # Initialize video predictor - handle compatible or missing arguments errors
            print("Initializing video predictor...")
            video_mode_available = False
            try:
                # Attempt standard initialization
                self.video_predictor = SAM2VideoPredictor(self.sam_model)
                video_mode_available = True
            except TypeError as e:
                if "missing 3 required positional arguments" in str(e):
                    print("The SAM2 video predictor requires additional arguments that aren't compatible with this model.")
                    print("Video propagation will not be available, but image segmentation will still work.")
                    
                    # Create mock video predictor for demo mode for compatibility
                    self._setup_mock_video_predictor()
                else:
                    print(f"Error initializing video predictor: {e}")
                    print("Video propagation may not be available, but image segmentation should work")
            except Exception as vid_error:
                print(f"Error initializing video predictor: {vid_error}")
                print("Video propagation may not be available, but image segmentation should work")
            
            # Print summary
            if video_mode_available:
                print("SAM2 model loaded successfully with full video propagation support")
            else:
                print("SAM2 model loaded successfully with image segmentation only")
                
            return True
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            print("Details:", str(e))
            import traceback
            traceback.print_exc()
            self.demo_mode = True  # Fall back to demo mode
            print("Falling back to demo mode")
            return False
    
    def store_mask_for_object(self, slice_type, slice_idx, obj_id, mask):
        """Store a generated mask for a specific object and slice."""
        if obj_id not in self.object_masks:
            self.object_masks[obj_id] = {}
        slice_key = f"{slice_type}_{slice_idx}"
        self.object_masks[obj_id][slice_key] = mask
        print(f"Stored mask for Object {obj_id} on slice {slice_key}")

    def has_masks_for_object(self, obj_id):
        """Check if any masks (generated or propagated) exist for the given object ID."""
        # Check generated masks
        if obj_id in self.object_masks and self.object_masks[obj_id]:
             return True
             
        # Check propagated masks (real predictor)
        if obj_id in self.inference_state and self.inference_state[obj_id] and \
           "output" in self.inference_state[obj_id] and self.inference_state[obj_id]["output"]:
            return True
            
        # Check propagated masks (demo mode)
        if obj_id in self.demo_state and self.demo_state[obj_id] and \
           "inference_state" in self.demo_state[obj_id] and \
           "output" in self.demo_state[obj_id]["inference_state"] and \
           self.demo_state[obj_id]["inference_state"]["output"]:
            return True
            
        return False

    def _setup_mock_video_predictor(self):
        """Set up a mock video predictor for compatibility when the real one fails"""
        class MockVideoPredictor:
            def __init__(self):
                pass
                
            def init_state(self, frames, offload_video_to_cpu=True):
                # Return a minimal inference state that will work with our demo mode
                return {"output": {}, "object_ids": []}
                
            def add_new_points(self, inference_state, frame_idx, obj_id, points, labels, clear_old_points=True):
                # Just store minimal info in the inference state
                if "object_ids" not in inference_state:
                    inference_state["object_ids"] = []
                if obj_id not in inference_state["object_ids"]:
                    inference_state["object_ids"].append(obj_id)
                return True
                
            def propagate_in_video(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
                # Do nothing, the application will fall back to demo mode for propagation
                return True
                
        self.video_predictor = MockVideoPredictor()
        print("Created mock video predictor for compatibility")
    
    def set_seismic_volume(self, seismic_volume):
        """Set the seismic volume data"""
        self.seismic_volume = seismic_volume
        # Reset masks
        self.object_masks = {}
        
    def _normalize_slice(self, slice_data):
        """Normalize the slice data to 0-255 range for SAM2 input"""
        # Clip to percentiles to handle outliers
        p_low, p_high = 1, 99
        low, high = np.percentile(slice_data, [p_low, p_high])
        slice_norm = np.clip(slice_data, low, high)
        
        # Normalize to 0-255 range
        slice_norm = ((slice_norm - low) / (high - low) * 255).astype(np.uint8)
        
        # Convert to RGB by repeating the channel
        slice_rgb = np.stack([slice_norm, slice_norm, slice_norm], axis=2)
        
        return slice_rgb
    
    def set_current_slice(self, slice_type, slice_idx, slice_data):
        """Set the current working slice"""
        self.current_slice_type = slice_type
        self.current_slice_idx = slice_idx
        self.current_slice = slice_data
    
    def predict_masks_from_points(self, points, point_labels, multimask_output=True):
        """Predict masks from points on the current slice using SAM2 image predictor
        
        Args:
            points: Nx2 array of point coordinates (x, y)
            point_labels: N array of point labels (1 for foreground, 0 for background)
            multimask_output: Whether to return multiple mask predictions
            
        Returns:
            masks: Array of predicted masks
            scores: Confidence scores for masks
            logits: Raw mask logits
        """
        if self.current_slice is None:
            raise ValueError("No slice data set")
            
        # For demo mode, generate a simple mask based on points
        if self.demo_mode:
            return self._generate_demo_mask(points, point_labels)
            
        # Normalize slice for SAM2 input
        slice_rgb = self._normalize_slice(self.current_slice)
        
        # Set image for prediction
        self.image_predictor.set_image(slice_rgb)
        
        # Predict masks
        masks, scores, logits = self.image_predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(point_labels),
            multimask_output=multimask_output
        )
        
        # Store masks for current slice - REMOVED, handled by store_mask_for_object in App
        # slice_key = f"{self.current_slice_type}_{self.current_slice_idx}"
        # self.masks[slice_key] = masks 
        
        return masks, scores, logits
    
    def _generate_demo_mask(self, points, point_labels):
        """Generate a demo mask based on points for testing without the model"""
        h, w = self.current_slice.shape
        
        # Create a simple mask centered at foreground points
        masks = []
        scores = []
        
        # Build a more representative mask - especially for fault lines
        # For fault lines, we want to connect points instead of just making circles
        base_mask = np.zeros((h, w), dtype=bool)
        
        # Collect foreground points
        fg_points = []
        for i, point in enumerate(points):
            if point_labels[i] == 1:  # Foreground point
                y, x = int(point[1]), int(point[0])
                fg_points.append((y, x))
        
        # If we have multiple foreground points, connect them with a line to simulate fault lines
        if len(fg_points) >= 2:
            from scipy import ndimage
            
            # For horizontal fault structures in our pre-transposed data
            # Sort points by x-coordinate to connect them horizontally
            fg_points.sort(key=lambda p: p[1])  # Sort by x-coordinate
            
            # Connect points with lines
            for i in range(len(fg_points) - 1):
                y1, x1 = fg_points[i]
                y2, x2 = fg_points[i + 1]
                
                # Create a line between the two points using Bresenham's algorithm
                # We'll use a simple approach here
                steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
                if steps == 0:
                    continue
                    
                for j in range(steps + 1):
                    t = j / steps
                    y = int(y1 * (1 - t) + y2 * t)
                    x = int(x1 * (1 - t) + x2 * t)
                    
                    # Make sure coordinates are within bounds
                    if 0 <= y < h and 0 <= x < w:
                        # Add a small circle at this point
                        y_indices, x_indices = np.ogrid[:h, :w]
                        # Use a thinner width for fault line
                        width = np.random.randint(4, 8)  # Reduced from 8-15 to 4-8
                        dist = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
                        base_mask = np.logical_or(base_mask, dist <= width)
            
            # Apply a very light dilation to create a more continuous feature
            base_mask = ndimage.binary_dilation(base_mask, iterations=1)
            
        else:
            # For single points or no connectivity, create circular masks - using smaller circles
            for i, point in enumerate(points):
                if point_labels[i] == 1:  # Foreground point
                    y, x = int(point[1]), int(point[0])
                    
                    # Create a circular mask
                    y_indices, x_indices = np.ogrid[:h, :w]
                    # Random radius for variation - reduce size
                    radius = np.random.randint(10, 25)  # Reduced from 20-50 to 10-25
                    dist_from_center = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
                    circle_mask = dist_from_center <= radius
                    base_mask = np.logical_or(base_mask, circle_mask)
        
        # Create three variations of the mask for multi-mask output
        # First is the base mask
        masks.append(base_mask)
        scores.append(0.95)
        
        # Second is a slightly dilated version
        try:
            from scipy import ndimage
            mask2 = ndimage.binary_dilation(base_mask, iterations=1)
            masks.append(mask2)
            scores.append(0.85)
        except:
            # Fallback if ndimage is not available
            mask2 = base_mask.copy()
            masks.append(mask2)
            scores.append(0.85)
        
        # Third is a slightly eroded version
        try:
            from scipy import ndimage
            # Only erode if we have enough True pixels
            if np.count_nonzero(base_mask) > 100:
                mask3 = ndimage.binary_erosion(base_mask, iterations=1)
            else:
                mask3 = base_mask.copy()
            masks.append(mask3)
            scores.append(0.75)
        except:
            # Fallback
            mask3 = base_mask.copy()
            masks.append(mask3)
            scores.append(0.75)
        
        # Convert to the expected format
        masks = np.array(masks)
        scores = np.array(scores)
        logits = np.zeros((3, h, w))  # Fake logits
        
        # Store masks for current slice - REMOVED, handled by store_mask_for_object in App
        # slice_key = f"{self.current_slice_type}_{self.current_slice_idx}"
        # self.masks[slice_key] = masks 
        
        return masks, scores, logits
    
    def set_demo_context(self, frame_idx, obj_id, points, labels):
        """Set context for demo mode operations for a specific object."""
        if obj_id not in self.demo_state:
            self.demo_state[obj_id] = {}
        self.demo_state[obj_id]['frame_idx'] = frame_idx
        self.demo_state[obj_id]['points'] = points
        self.demo_state[obj_id]['labels'] = labels
        print(f"Set demo context for Object {obj_id}: frame={frame_idx}, {len(points)} points")

    def init_video_predictor(self, slice_indices, obj_id):
        """Initialize the video predictor with a sequence of slices for a specific object
        
        Args:
            slice_indices: List of slice indices to use for propagation
            obj_id: The object ID this initialization is for
        """
        if self.demo_mode:
            # In demo mode, store indices and other context per object ID
            if obj_id not in self.demo_state:
                 self.demo_state[obj_id] = {}
            self.demo_state[obj_id]['slice_indices'] = slice_indices
            # Ensure inference_state exists for demo
            if 'inference_state' not in self.demo_state[obj_id]:
                self.demo_state[obj_id]['inference_state'] = {"output": {}, "object_ids": [obj_id]}
                
            print(f"Demo mode: Initialized video predictor for Object {obj_id} with {len(slice_indices)} frames")
            return True
            
        # --- Real Predictor Path ---
        # REMOVED max_slices logic - process all provided slice_indices
        self.selected_slice_indices = slice_indices 
        print(f"Initializing predictor for {len(self.selected_slice_indices)} slices.")
            
        # Create a "video" from the sequence of slices
        frames = []
        
        # First determine the maximum dimensions across all slices to ensure consistency
        max_h, max_w = 0, 0
        print("Scanning slices to determine maximum dimensions...")
        for idx in self.selected_slice_indices:
            if self.current_slice_type == "inline":
                slice_data = self.seismic_volume.get_inline_slice(idx)
            elif self.current_slice_type == "crossline":
                slice_data = self.seismic_volume.get_crossline_slice(idx)
            elif self.current_slice_type == "timeslice":
                slice_data = self.seismic_volume.get_timeslice(idx)
            else:
                raise ValueError(f"Invalid slice type: {self.current_slice_type}")
            
            h, w = slice_data.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        
        print(f"Maximum slice dimensions: {max_h}x{max_w}")
        target_shape = (max_h, max_w)
        self.common_slice_shape = target_shape  # Store for future reference
        
        # Now process all slices and pad to the maximum dimensions
        print("Processing slices...")
        for i, idx in enumerate(self.selected_slice_indices):
            print(f"Processing slice {i+1}/{len(self.selected_slice_indices)}: {idx}")
            if self.current_slice_type == "inline":
                slice_data = self.seismic_volume.get_inline_slice(idx)
            elif self.current_slice_type == "crossline":
                slice_data = self.seismic_volume.get_crossline_slice(idx)
            elif self.current_slice_type == "timeslice":
                slice_data = self.seismic_volume.get_timeslice(idx)
            else:
                raise ValueError(f"Invalid slice type: {self.current_slice_type}")
                
            # Ensure consistent dimensions by padding to maximum dimensions
            if slice_data.shape != target_shape:
                slice_data = self._pad_slice(slice_data, target_shape)
                
            # Normalize slice for SAM2 input
            slice_rgb = self._normalize_slice(slice_data)
            frames.append(slice_rgb)
            
        # Initialize video predictor
        print(f"Initializing video predictor with {len(frames)} frames...")
        self.inference_state[obj_id] = self.video_predictor.init_state(frames, 
                                                              offload_video_to_cpu=True)
        
        # Store a mapping from selected slice indices to frame indices
        self.slice_maps[obj_id] = {idx: i for i, idx in enumerate(self.selected_slice_indices)}
        
        print(f"Video predictor initialization complete for Object {obj_id}")
        return True
        
    def _pad_slice(self, slice_data, target_shape):
        """Pad a slice to target shape to ensure consistency"""
        h, w = slice_data.shape
        target_h, target_w = target_shape
        
        # Create a new array with target shape
        padded = np.zeros(target_shape, dtype=slice_data.dtype)
        
        # Copy data
        padded[:h, :w] = slice_data
        
        return padded
        
    def add_point_to_video(self, frame_idx, obj_id, points, labels):
        """Add points to a specific frame in the video sequence
        
        Args:
            frame_idx: Index of the frame to add points to
            obj_id: Object ID for tracking
            points: Nx2 array of point coordinates (x, y)
            labels: N array of point labels (1 for foreground, 0 for background)
            
        Returns:
            True if successful
        """
        if self.demo_mode:
            # In demo mode, just store the points and labels per object
            if obj_id not in self.demo_state:
                self.demo_state[obj_id] = {} # Should be initialized by init_video_predictor
            self.demo_state[obj_id]['frame_idx'] = frame_idx # Store starting frame
            self.demo_state[obj_id]['points'] = points
            self.demo_state[obj_id]['labels'] = labels
            print(f"Demo mode: Added points for Object {obj_id} at frame {frame_idx}")
            return True
            
        if obj_id not in self.inference_state:
            raise ValueError(f"Video predictor not initialized for Object ID {obj_id}")
            
        current_state = self.inference_state[obj_id]
        
        # Add the object ID to the state if it's not there (important for mock predictor)
        if "object_ids" not in current_state:
            current_state["object_ids"] = []
        if obj_id not in current_state["object_ids"]:
             current_state["object_ids"].append(obj_id)
             
        self.video_predictor.add_new_points(
            current_state,
            frame_idx,
            obj_id,
            points=np.array(points),
            labels=np.array(labels),
            clear_old_points=True
        )
        
        return True
        
    def propagate_masks(self, obj_id, start_frame_idx=None, max_frames=None, reverse=False): # Add obj_id
        """Propagate masks through the video sequence for a specific object
        
        Args:
            obj_id: The object ID to propagate masks for
            start_frame_idx: Starting frame index
            max_frames: Maximum number of frames to process
            reverse: Whether to propagate backward
            
        Returns:
            True if successful
        """
        if self.demo_mode:
            # In demo mode, create fake propagated masks for the specified object
            self._generate_demo_propagated_masks(obj_id, reverse) # Pass obj_id
            return True
            
        if obj_id not in self.inference_state:
            raise ValueError(f"Video predictor not initialized for Object ID {obj_id}")
            
        current_state = self.inference_state[obj_id]
            
        self.video_predictor.propagate_in_video(
            current_state, # Use state for this object
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frames,
            reverse=reverse
        )
        
        return True
    
    def _generate_demo_propagated_masks(self, obj_id, reverse=False): # Accept obj_id
        """Generate fake propagated masks for demo mode for a specific object"""
        if obj_id not in self.demo_state or 'slice_indices' not in self.demo_state[obj_id] or \
           'points' not in self.demo_state[obj_id] or 'frame_idx' not in self.demo_state[obj_id]:
            raise ValueError(f"Demo video predictor context not fully set for Object ID {obj_id}")
            
        state_info = self.demo_state[obj_id]
        slice_indices = state_info['slice_indices']
        points = state_info['points']
        labels = state_info['labels']
        start_frame = state_info['frame_idx'] # This is the frame position
        
        # Get slice dimensions from current slice
        if self.current_slice is None:
            # Attempt to load the slice corresponding to the start frame
            try:
                 slice_idx = slice_indices[start_frame]
                 if self.current_slice_type == "inline":
                     slice_data = self.seismic_volume.get_inline_slice(slice_idx)
                 elif self.current_slice_type == "crossline":
                     slice_data = self.seismic_volume.get_crossline_slice(slice_idx)
                 else: # timeslice
                     slice_data = self.seismic_volume.get_timeslice(slice_idx)
                 base_h, base_w = slice_data.shape
            except:
                 print("Warning: Cannot determine slice shape for demo mask generation. Using default 500x500.")
                 base_h, base_w = 500, 500 # Default if no slice loaded
        else:
            base_h, base_w = self.current_slice.shape
        
        # Use the inference_state stored within demo_state for this object
        demo_inference_state = state_info.setdefault('inference_state', {"output": {}, "object_ids": [obj_id]})
        
        # Note: slice_indices are position indices (0 to len-1)
        slice_positions = list(range(len(slice_indices)))
        
        # Generate masks for all slices
        direction = -1 if reverse else 1
        
        # Make sure start_frame is an integer (frame position)
        if not isinstance(start_frame, int):
            print(f"Warning: demo_frame_idx ({start_frame}) is not an integer for Object {obj_id}, using 0 instead")
            start_frame = 0
            state_info['frame_idx'] = 0 # Correct stored value
            
        # Define frame ranges for propagation
        if reverse:
            frame_range = range(start_frame, -1, -1)
        else:
            frame_range = range(start_frame, len(slice_positions))

        # First, generate the initial mask at the start frame
        print(f"Generating demo masks for Object {obj_id} from frame position {start_frame}")
        
        # Create the initial mask
        try:
            # Get slice data based on the position index
            position = slice_indices[start_frame]
            print(f"Initial mask for Object {obj_id}: Using position {position}")
            
            if self.current_slice_type == "inline":
                slice_data = self.seismic_volume.get_inline_slice(position)
            elif self.current_slice_type == "crossline":
                slice_data = self.seismic_volume.get_crossline_slice(position)
            elif self.current_slice_type == "timeslice":
                slice_data = self.seismic_volume.get_timeslice(position)
            
            mask_h, mask_w = slice_data.shape
            print(f"Initial slice shape for Object {obj_id}: {mask_h}x{mask_w}")
        except Exception as e:
            print(f"Error loading slice for initial mask (Object {obj_id}): {e}, using base dimensions")
            mask_h, mask_w = base_h, base_w
        
        # For the initial frame, use the mask generated from points for this object
        # Temporarily set current slice for _generate_demo_mask
        original_slice = self.current_slice
        try:
            if 'slice_data' in locals():
                 self.current_slice = slice_data
            else: # Fallback if slice loading failed
                 self.current_slice = np.zeros((base_h, base_w))
                 
            masks, _, _ = self._generate_demo_mask(points, labels)
            initial_mask = masks[0].astype(bool)  # Use the first mask
            print(f"Generated initial mask for Object {obj_id} with shape {initial_mask.shape}")
        finally:
            self.current_slice = original_slice # Restore original slice
        
        # Store the initial mask - use frame position as key
        demo_inference_state["output"][start_frame] = {"mask": [initial_mask]}
        
        # Get the reference mask structure and size for this object
        reference_mask = initial_mask.copy()
        reference_area = np.count_nonzero(reference_mask)
        state_info['reference_mask_area'] = reference_area # Store per object
        print(f"Reference mask area for Object {obj_id}: {reference_area} pixels")
        
        # Calculate max allowable mask size (to prevent runaway growth)
        max_mask_size = min(50000, reference_area * 3)
        
        # Now propagate masks by transforming the previous mask
        prev_frame = start_frame
        prev_mask = reference_mask
        
        # Process all other frames
        for frame_pos in frame_range:
            # Skip the start frame as we already processed it
            if frame_pos == start_frame:
                continue
                
            # Get slice position for this frame position
            try:
                position = slice_indices[frame_pos]
                print(f"Processing frame {frame_pos}, position {position} for Object {obj_id}")
            except (IndexError, TypeError) as e:
                print(f"Error getting position for frame {frame_pos} (Object {obj_id}): {e}")
                continue
                
            try:
                # Get the current slice
                if self.current_slice_type == "inline":
                    curr_slice = self.seismic_volume.get_inline_slice(position)
                elif self.current_slice_type == "crossline":
                    curr_slice = self.seismic_volume.get_crossline_slice(position)
                elif self.current_slice_type == "timeslice":
                    curr_slice = self.seismic_volume.get_timeslice(position)
                
                curr_h, curr_w = curr_slice.shape
                
                # Create a new mask by transforming the previous mask
                # First, ensure previous mask size matches current slice
                if prev_mask.shape != (curr_h, curr_w):
                    temp_mask = np.zeros((curr_h, curr_w), dtype=bool)
                    min_h = min(prev_mask.shape[0], curr_h)
                    min_w = min(prev_mask.shape[1], curr_w)
                    temp_mask[:min_h, :min_w] = prev_mask[:min_h, :min_w]
                    prev_mask = temp_mask
                
                # Apply very minimal transformations to the mask - just slight shifts
                from scipy import ndimage
                
                # Use different random shifts per object potentially? (Keep simple for now)
                if self.current_slice_type in ["inline", "crossline"]:
                    shift_x = np.random.randint(-2, 3) * 0.5
                    shift_y = np.random.randint(-1, 2) * 0.2
                else:
                    shift_x = np.random.randint(-1, 2)
                    shift_y = np.random.randint(-1, 2)
                
                new_mask = ndimage.shift(prev_mask.astype(float), (shift_y, shift_x), order=0, mode='constant', cval=0.0) > 0.5
                
                # Enforce strict size constraints based on this object's reference area
                current_reference_area = state_info.get('reference_mask_area', 10000) # Use object's ref area
                new_area = np.count_nonzero(new_mask)
                area_ratio = new_area / current_reference_area if current_reference_area > 0 else 1.0
                
                # If mask deviates even slightly from reference size, adjust it
                if area_ratio > 1.1 or area_ratio < 0.9:
                    print(f"Object {obj_id}: Adjusting mask size, area ratio: {area_ratio:.2f}")
                    if area_ratio > 1.1:
                        iterations = 0
                        temp_mask = new_mask.copy()
                        while np.count_nonzero(temp_mask) > 1.1 * current_reference_area and iterations < 5:
                            temp_mask = ndimage.binary_erosion(temp_mask)
                            iterations += 1
                        new_mask = temp_mask
                    elif area_ratio < 0.9:
                        iterations = 0
                        temp_mask = new_mask.copy()
                        while np.count_nonzero(temp_mask) < 0.9 * current_reference_area and iterations < 5:
                            temp_mask = ndimage.binary_dilation(temp_mask)
                            iterations += 1
                        new_mask = temp_mask
                    new_area = np.count_nonzero(new_mask)
                    new_ratio = new_area / current_reference_area
                    print(f"Object {obj_id}: After adjustment: area={new_area}, ratio={new_ratio:.2f}")
                
                # If mask is still too big or there's another issue, restore from reference
                if np.count_nonzero(new_mask) > max_mask_size:
                    print(f"WARNING (Object {obj_id}): Mask too large ({np.count_nonzero(new_mask)} pixels) - using reference mask")
                    clean_shift_x = shift_x * 0.8
                    clean_shift_y = 0
                    new_mask = ndimage.shift(reference_mask.astype(float), 
                                            (clean_shift_y, clean_shift_x), 
                                            order=0, mode='constant', cval=0.0) > 0.5
                
                # Store the mask for this frame - use position as key
                demo_inference_state["output"][frame_pos] = {"mask": [new_mask]}
                prev_mask = new_mask
                prev_frame = frame_pos
                
            except Exception as e:
                print(f"Error creating mask for frame {frame_pos}, position {position} (Object {obj_id}): {e}")
                try:
                    if self.current_slice_type == "inline":
                        curr_slice = self.seismic_volume.get_inline_slice(position)
                    elif self.current_slice_type == "crossline":
                        curr_slice = self.seismic_volume.get_crossline_slice(position)
                    elif self.current_slice_type == "timeslice":
                        curr_slice = self.seismic_volume.get_timeslice(position)
                    empty_mask = np.zeros(curr_slice.shape, dtype=bool)
                except Exception:
                    empty_mask = np.zeros((base_h, base_w), dtype=bool)
                
                demo_inference_state["output"][frame_pos] = {"mask": [empty_mask]}
    
    def _add_noise_to_mask_edges(self, mask):
        """Add noise to mask edges to simulate realistic SAM2 propagation"""
        # Find mask boundaries
        from scipy import ndimage
        
        try:
            # Dilate and erode to find boundary
            dilated = ndimage.binary_dilation(mask)
            eroded = ndimage.binary_erosion(mask)
            boundary = np.logical_xor(dilated, eroded)
            
            # Add random noise to boundary
            noise = np.random.random(boundary.shape) > 0.5
            noise_boundary = np.logical_and(boundary, noise)
            
            # Either add or remove the noisy boundary
            add_noise = np.random.random() > 0.5
            if add_noise:
                noisy_mask = np.logical_or(mask, noise_boundary)
            else:
                noisy_mask = np.logical_and(mask, np.logical_not(noise_boundary))
                
            return noisy_mask
        except Exception as e:
            # If there's an error, just return the original mask
            print(f"Error adding noise to mask edges: {e}, returning original mask")
            return mask
        
    def get_mask_for_frame(self, frame_idx, obj_id):
        """Get the propagated mask for a specific frame and object
        
        Args:
            frame_idx: Frame index in the processed sequence
            obj_id: Object ID
            
        Returns:
            Mask array or None if no mask available
        """
        if self.demo_mode:
            try:
                # Use the demo state specific to this object ID
                if obj_id not in self.demo_state or 'inference_state' not in self.demo_state[obj_id]:
                    print(f"Demo masks not generated yet for Object {obj_id}")
                    return None
                    
                state_info = self.demo_state[obj_id]
                demo_inference_state = state_info.get('inference_state')
                
                if demo_inference_state is None or "output" not in demo_inference_state:
                     print(f"Demo inference state invalid for Object {obj_id}")
                     return None

                # Debug info about available frames for this object
                output_frames = list(demo_inference_state["output"].keys())
                print(f"Object {obj_id}: Looking for frame {frame_idx} in available frames: {output_frames[:10]}{'...' if len(output_frames) > 10 else ''}")
                
                # Get reference area for this object
                reference_area = state_info.get('reference_mask_area', 10000) 
                
                # First try exact frame index
                if frame_idx in demo_inference_state["output"]:
                    # Demo mode currently only supports one object ID per state
                    # The obj_id check is implicit as we fetched the state for this obj_id
                    mask = demo_inference_state["output"][frame_idx]["mask"][0]
                    print(f"Found exact mask for frame {frame_idx} (Object {obj_id})")
                else:
                    # Try to find the closest frame
                    print(f"No mask for exact frame {frame_idx} in demo mode (Object {obj_id}), looking for nearby frames")
                    
                    # Find closest available frame
                    closest_frame = None
                    min_dist = float('inf')
                    search_range = 30
                    
                    for avail_frame in demo_inference_state["output"].keys():
                        dist = abs(avail_frame - frame_idx)
                        if dist < min_dist and dist <= search_range:
                            min_dist = dist
                            closest_frame = avail_frame
                    
                    if closest_frame is not None:
                        print(f"Using nearby frame {closest_frame} (distance: {min_dist}) for Object {obj_id}")
                        mask = demo_inference_state["output"][closest_frame]["mask"][0]
                    else:
                        print(f"No suitable nearby frame found for Object {obj_id} within range {search_range}")
                        if self.current_slice is not None:
                            return np.zeros(self.current_slice.shape, dtype=bool)
                        return None
                
                # Check if mask has grown too large - apply emergency size correction
                mask_size = np.count_nonzero(mask)
                mask_ratio = mask_size / reference_area if reference_area > 0 else 1.0
                
                if mask_size > 50000 or mask_ratio > 5.0:
                    print(f"WARNING (Object {obj_id}): Mask too large ({mask_size} pixels, ratio: {mask_ratio:.1f}). Emergency correction applied.")
                    from scipy import ndimage
                    
                    # If it's a huge mask, try to find the reference mask and shift it
                    ref_mask = None
                    start_frame = state_info.get('frame_idx', 0)
                    if start_frame in demo_inference_state["output"]:
                        ref_mask = demo_inference_state["output"][start_frame]["mask"][0]
                        
                    if ref_mask is not None:
                         # Shift reference mask slightly instead of using corrupted mask
                         shift_x = np.random.randint(-1, 2)
                         shift_y = 0
                         mask = ndimage.shift(ref_mask.astype(float), (shift_y, shift_x), order=0, mode='constant', cval=0.0) > 0.5
                         print(f"After correction (Object {obj_id}): {np.count_nonzero(mask)} pixels")
                    else:
                         # If reference not found, just erode current mask aggressively
                         while np.count_nonzero(mask) > 1.5 * reference_area and np.count_nonzero(mask) > 100:
                             mask = ndimage.binary_erosion(mask)
                         print(f"After erosion correction (Object {obj_id}): {np.count_nonzero(mask)} pixels")

                # Make sure the mask matches the current slice dimensions
                if self.current_slice is not None and mask.shape != self.current_slice.shape:
                    print(f"Resizing demo mask from {mask.shape} to {self.current_slice.shape} (Object {obj_id})")
                    h, w = self.current_slice.shape
                    mask_h, mask_w = mask.shape
                    resized_mask = np.zeros(self.current_slice.shape, dtype=bool)
                    min_h = min(h, mask_h)
                    min_w = min(w, mask_w)
                    resized_mask[:min_h, :min_w] = mask[:min_h, :min_w]
                    return resized_mask
                    
                return mask
            except Exception as e:
                print(f"Error getting demo mask for frame {frame_idx}, Object {obj_id}: {e}")
                if self.current_slice is not None:
                    return np.zeros(self.current_slice.shape, dtype=bool)
                return None
            
        # --- Real Predictor Logic ---
        if obj_id not in self.inference_state:
             print(f"Video predictor not initialized for Object ID {obj_id}")
             # Check if a manually generated mask exists for this slice/object
             slice_key = f"{self.current_slice_type}_{self.current_slice_idx}"
             if obj_id in self.object_masks and slice_key in self.object_masks[obj_id]:
                 print(f"Returning generated mask for Object {obj_id} on slice {slice_key}")
                 return self.object_masks[obj_id][slice_key]
             return None # No state and no generated mask
             
        current_state = self.inference_state[obj_id]
        
        # Get the current slice to determine correct dimensions
        if self.current_slice is None:
            print("No current slice set")
            return None
            
        current_slice_shape = self.current_slice.shape
        
        # Check if the frame index is valid within the state for this object
        try:
            # Get mask from the inference state for this object
            if frame_idx not in current_state["output"]:
                 raise KeyError # Frame not processed for this object
                 
            masks = current_state["output"][frame_idx]["mask"]
            mask_ids = current_state["object_ids"]
            
            # Find index for the requested object ID within this state
            obj_idx = None
            for i, mask_id in enumerate(mask_ids):
                if mask_id == obj_id:
                    obj_idx = i
                    break
                    
            if obj_idx is None:
                print(f"Object ID {obj_id} not found in masks for frame {frame_idx} in its state")
                return np.zeros(current_slice_shape, dtype=bool) # Return empty mask with correct shape
                
            mask = masks[obj_idx]
        except KeyError:
            # Try to find the closest frame *within this object's state*
            print(f"Frame {frame_idx} not in processed frames for Object {obj_id}, looking for nearby frames")
            closest_frame = None
            min_dist = float('inf')
            
            search_range = 30
            for avail_frame in current_state["output"].keys(): # Search keys in this object's state
                dist = abs(avail_frame - frame_idx)
                if dist < min_dist and dist <= search_range:
                    min_dist = dist
                    closest_frame = avail_frame
            
            if closest_frame is not None:
                print(f"Using nearby frame {closest_frame} (distance: {min_dist}) for Object {obj_id}")
                try:
                    masks = current_state["output"][closest_frame]["mask"]
                    mask_ids = current_state["object_ids"]
                    
                    # Find index for the requested object ID
                    obj_idx = None
                    for i, mask_id in enumerate(mask_ids):
                        if mask_id == obj_id:
                            obj_idx = i
                            break
                            
                    if obj_idx is None:
                        print(f"Object ID {obj_id} not found in available masks for nearby frame")
                        return np.zeros(current_slice_shape, dtype=bool)
                        
                    mask = masks[obj_idx]
                except Exception as e:
                    print(f"Error getting mask from nearby frame for Object {obj_id}: {e}")
                    return np.zeros(current_slice_shape, dtype=bool)
                
        # Return a correctly sized mask for this slice
        if mask.shape != current_slice_shape:
            print(f"Resizing mask from {mask.shape} to {current_slice_shape} (Object {obj_id})")
            try:
                # If the mask is larger than the slice, crop it
                h, w = current_slice_shape
                mask_h, mask_w = mask.shape
                
                if mask_h > h or mask_w > w:
                    mask = mask[:h, :w]
                
                # If the mask is smaller than the slice, pad it
                if mask_h < h or mask_w < w:
                    padded_mask = np.zeros(current_slice_shape, dtype=bool)
                    padded_mask[:min(mask_h, h), :min(mask_w, w)] = mask[:min(mask_h, h), :min(mask_w, w)]
                    mask = padded_mask
            except Exception as e:
                print(f"Error resizing mask: {e}")
                # Return an empty mask with the correct dimensions
                return np.zeros(current_slice_shape, dtype=bool)
            
        return mask
    
    def create_mask_overlay(self, slice_data, mask, alpha=0.5, color=[1, 0, 0]):
        """Create a visualization with the mask overlaid on the slice
        
        Args:
            slice_data: The seismic slice data
            mask: Binary mask
            alpha: Transparency of the overlay
            color: RGB color for the mask
            
        Returns:
            Figure with the overlay
        """
        # No need to rotate or transpose as data should be correctly oriented from extraction
        # Just ensure mask has the same shape as slice_data
        if mask.shape != slice_data.shape:
            print(f"Reshaping mask from {mask.shape} to {slice_data.shape}")
            from scipy import ndimage
            # If dimensions don't match, resize the mask to match slice dimensions
            resized_mask = np.zeros(slice_data.shape, dtype=bool)
            min_h = min(slice_data.shape[0], mask.shape[0])
            min_w = min(slice_data.shape[1], mask.shape[1])
            resized_mask[:min_h, :min_w] = mask[:min_h, :min_w]
            mask = resized_mask
        
        # Normalize the slice data for display
        vmin, vmax = np.percentile(slice_data, [5, 95])
        
        # Create figure
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display slice with 'gray' colormap for better contrast
        ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Create mask overlay
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask > 0] = [*color, alpha]
        
        # Display mask overlay
        ax.imshow(mask_overlay, aspect='auto')
        
        # Set proper axis labels
        ax.set_xlabel('Trace Position')
        ax.set_ylabel('Time/Depth')
        
        ax.set_title(f"{self.current_slice_type.capitalize()} {self.current_slice_idx} with Mask")
        
        return fig 

    # --- Horizon refinement utilities ---
    def mask_to_polyline(self, mask):
        """Extract a simple centerline polyline from a binary mask.

        For each x-column, take the median y-position of True pixels as the centerline.

        Returns:
            x_coords: 1D numpy array of x positions (columns with mask coverage)
            y_coords: 1D numpy array of y positions (row index per x)
        """
        if mask is None:
            return None, None
        if mask.ndim != 2:
            mask = np.squeeze(mask)
        if mask.dtype != bool:
            mask = mask.astype(bool)

        h, w = mask.shape
        xs = []
        ys = []
        # Iterate over columns
        for x in range(w):
            y_idx = np.flatnonzero(mask[:, x])
            if y_idx.size == 0:
                continue
            # Use median to be robust to outliers
            y_med = int(np.median(y_idx))
            xs.append(x)
            ys.append(y_med)
        if not xs:
            return None, None
        return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)

    def _smooth_1d(self, v, kernel_size):
        """Simple moving-average smoothing with odd kernel_size."""
        v = np.asarray(v, dtype=float)
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        if k <= 1:
            return v
        kernel = np.ones(k, dtype=float) / float(k)
        # Reflect padding to preserve edges
        pad = k // 2
        vp = np.pad(v, (pad, pad), mode='reflect')
        vs = np.convolve(vp, kernel, mode='valid')
        return vs

    def _reflector_data_score(self, v, mode, kernel_size=1):
        """Compute a per-sample data score for a vertical trace segment v.

        Lower is better (used as cost). Mode can be 'peak', 'trough', or 'zero'.
        Uses smoothed amplitude, gradient and curvature for ridge-aware selection.
        """
        v = np.asarray(v, dtype=float)
        eps = 1e-6
        vs = self._smooth_1d(v, kernel_size)
        g = np.gradient(vs)
        c = np.gradient(g)
        # Normalize signals
        a = vs / (np.mean(np.abs(vs)) + eps)
        g_n = g / (np.mean(np.abs(g)) + eps)
        c_n = c / (np.mean(np.abs(c)) + eps)

        if mode == 'peak':
            # Favor strong positive amplitude (large a), near zero gradient, negative curvature (local max)
            cost = -a + 0.25 * np.abs(g_n) + (-0.5) * np.minimum(c_n, 0.0)
        elif mode == 'trough':
            # Favor strong negative amplitude (large -a), near zero gradient, positive curvature (local min)
            cost = a + 0.25 * np.abs(g_n) + (-0.5) * np.maximum(c_n, 0.0)  # a is negative for troughs, so higher a→worse, keep as is
        else:  # 'zero'
            # Favor amplitude near zero but steep gradient; curvature neutral
            cost = np.abs(a) - 0.6 * np.abs(g_n) + 0.1 * np.abs(c_n)

        return cost

    def snap_polyline_to_reflector(self, slice_data, x_coords, y_coords, mode='peak', window=8, kernel_size=7):
        """Snap an input polyline to the nearest reflector (peak/trough/zero) within a vertical window.

        Returns:
            y_snapped: numpy array of y positions after snapping
            point_conf: numpy array of confidence per point in [0, 1]
            mean_conf: float average confidence
        """
        if x_coords is None or y_coords is None:
            return None, None, 0.0
        h, w = slice_data.shape
        y_snapped = np.copy(y_coords).astype(float)
        point_conf = np.zeros_like(y_snapped, dtype=float)
        eps = 1e-6

        for i, x in enumerate(x_coords):
            y0 = int(round(y_coords[i]))
            y_min = max(0, y0 - int(window))
            y_max = min(h - 1, y0 + int(window))
            if y_max < y_min:
                continue
            v = slice_data[y_min:y_max + 1, int(x)]

            # Build score (lower is better)
            cost = self._reflector_data_score(v, mode, kernel_size=kernel_size)
            # Choose best index
            j = int(np.argmin(cost))
            y_best = y_min + j
            y_snapped[i] = y_best

            # Confidence: normalize relative improvement vs. median cost
            c_best = float(cost[j])
            c_med = float(np.median(cost))
            c_std = float(np.std(cost)) + eps
            # Map to [0,1]: higher improvement => higher confidence
            z = (c_med - c_best) / c_std
            point_conf[i] = 1.0 / (1.0 + np.exp(-z))  # sigmoid

        mean_conf = float(np.nanmean(point_conf)) if point_conf.size else 0.0
        return y_snapped, point_conf, mean_conf

    def _split_contiguous(self, x_coords):
        """Split indices into contiguous segments where x increments by 1."""
        if x_coords is None or len(x_coords) == 0:
            return []
        segments = []
        start = 0
        for i in range(1, len(x_coords)):
            if x_coords[i] != x_coords[i - 1] + 1:
                segments.append((start, i))
                start = i
        segments.append((start, len(x_coords)))
        return segments

    def dynamic_programming_refine(self, slice_data, x_coords, y_seed, mode='peak', dp_window=8, lam_smooth=1.0, kernel_size=7):
        """Refine a horizon path with a simple dynamic programming smoother.

        Args:
            slice_data: 2D numpy array (H, W)
            x_coords: 1D array of x positions (assumed increasing)
            y_seed: 1D array of initial y positions (same length as x_coords)
            mode: 'peak' | 'trough' | 'zero'
            dp_window: vertical search half-window around y_seed
            lam_smooth: smoothness penalty weight (|y_i - y_{i-1}|)

        Returns:
            y_refined: 1D array of refined y positions
        """
        if x_coords is None or y_seed is None or len(x_coords) == 0:
            return None
        h, w = slice_data.shape
        x_coords = np.asarray(x_coords, dtype=int)
        y_seed = np.asarray(y_seed, dtype=float)

        y_out = np.copy(y_seed)

        # Process contiguous x segments to keep DP small
        for s, e in self._split_contiguous(x_coords):
            xs = x_coords[s:e]
            ys0 = y_seed[s:e]
            if len(xs) == 0:
                continue

            # Build candidate sets per position
            cand_list = []
            cost_list = []
            for i, x in enumerate(xs):
                y0 = int(round(ys0[i]))
                y_min = max(0, y0 - int(dp_window))
                y_max = min(h - 1, y0 + int(dp_window))
                if y_max < y_min:
                    # Fallback to current y
                    y_min = y0
                    y_max = y0
                cand_y = np.arange(y_min, y_max + 1, dtype=int)
                v = slice_data[cand_y, int(x)]
                data_cost = self._reflector_data_score(v, mode, kernel_size=kernel_size)
                cand_list.append(cand_y)
                cost_list.append(data_cost)

            # DP tables
            # Initialize with first column
            J0 = cost_list[0]
            prev_cost = np.array(J0, dtype=float)
            back_ptrs = []

            # Forward pass
            for i in range(1, len(xs)):
                y_prev = cand_list[i - 1]
                y_curr = cand_list[i]
                data_cost = cost_list[i]
                new_cost = np.full_like(data_cost, np.inf, dtype=float)
                back_idx = np.full_like(data_cost, -1, dtype=int)

                for j, yc in enumerate(y_curr):
                    # Smoothness term relative to all previous candidates
                    smooth = lam_smooth * np.abs(y_prev - yc)
                    c = prev_cost + smooth
                    k = int(np.argmin(c))
                    new_cost[j] = float(c[k]) + float(data_cost[j])
                    back_idx[j] = k
                prev_cost = new_cost
                back_ptrs.append(back_idx)

            # Backtrack
            y_ref = np.zeros(len(xs), dtype=float)
            k = int(np.argmin(prev_cost))
            y_ref[-1] = float(cand_list[-1][k])
            for i in range(len(xs) - 2, -1, -1):
                k = int(back_ptrs[i][k])
                y_ref[i] = float(cand_list[i][k])

            # Write back
            y_out[s:e] = y_ref

        return y_out

    def refine_horizon(self, obj_id, slice_type, slice_idx, slice_data, seed_mask=None, prev_path=None,
                       mode='peak', snap_window=8, dp_window=8, dp_lambda=1.0, snap_kernel=7):
        """Refine a horizon from a seed (mask or previous path), snap to reflector, and smooth.

        Stores the result internally and returns (x, y, confidence).
        """
        # Decide seed polyline
        x_seed, y_seed = (None, None)
        if seed_mask is not None:
            x_seed, y_seed = self.mask_to_polyline(seed_mask)

        # If no mask-based seed, try previous stored path
        if (x_seed is None or y_seed is None) and prev_path is not None:
            x_prev, y_prev = prev_path
            x_seed, y_seed = np.asarray(x_prev, dtype=int), np.asarray(y_prev, dtype=float)

        if x_seed is None or y_seed is None or len(x_seed) == 0:
            return None

        # Snap to nearest reflector
        y_snapped, conf_pts, conf_mean = self.snap_polyline_to_reflector(
            slice_data, x_seed, y_seed, mode=mode, window=snap_window, kernel_size=snap_kernel
        )
        if y_snapped is None:
            return None

        # Smooth with DP around snapped path
        y_refined = self.dynamic_programming_refine(
            slice_data, x_seed, y_snapped, mode=mode, dp_window=dp_window, lam_smooth=dp_lambda, kernel_size=snap_kernel
        )
        if y_refined is None:
            return None

        # Confidence: combine snapping confidence and local consistency (stability wrt snapped)
        if conf_pts is None or len(conf_pts) != len(y_refined):
            conf = float(conf_mean) if conf_pts is not None else 0.0
        else:
            delta = np.abs(y_refined - y_snapped)
            stable = np.clip(1.0 - (delta / (dp_window + 1e-6)), 0.0, 1.0)
            conf = float(0.6 * conf_mean + 0.4 * float(np.nanmean(stable)))

        # Store
        slice_key = f"{slice_type}_{slice_idx}"
        if obj_id not in self.horizon_paths:
            self.horizon_paths[obj_id] = {}
        self.horizon_paths[obj_id][slice_key] = {"x": x_seed.astype(int), "y": y_refined.astype(float), "confidence": conf}

        return x_seed, y_refined, conf

    def get_prev_horizon(self, obj_id, slice_type, slice_idx):
        """Get the previous slice's refined horizon path if available."""
        prev_key = f"{slice_type}_{slice_idx - 1}"
        if obj_id in self.horizon_paths and prev_key in self.horizon_paths[obj_id]:
            p = self.horizon_paths[obj_id][prev_key]
            return p["x"], p["y"]
        return None