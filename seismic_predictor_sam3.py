import numpy as np
import torch
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import threading
import shutil
import tempfile
import cv2
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

# Try to import skimage, provide fallback if not available
SKIMAGE_AVAILABLE = False
try:
    from skimage.morphology import skeletonize, thin
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Using fallback for skeletonization.")
    # Simple fallback functions
    def skeletonize(mask):
        """Simple fallback skeletonization using erosion."""
        from scipy.ndimage import binary_erosion
        result = mask.copy()
        while True:
            eroded = binary_erosion(result)
            if not np.any(eroded):
                break
            result = eroded
        return result
    
    def thin(mask):
        return skeletonize(mask)
    
    def label(mask):
        from scipy.ndimage import label as scipy_label
        return scipy_label(mask)
    
    def regionprops(labeled):
        return []

# Add parent directory to path for importing SAM3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock triton if not available (Windows compatibility)
try:
    import triton
except ImportError:
    import sys
    from unittest.mock import MagicMock
    from types import ModuleType
    
    # Create a mock module with a spec to satisfy importlib/torch
    triton_mock = MagicMock()
    triton_mock.__spec__ = MagicMock()
    triton_mock.__spec__.name = "triton"
    triton_mock.__spec__.loader = MagicMock()
    triton_mock.__path__ = []
    triton_mock.__file__ = "mock_triton.py"
    
    sys.modules["triton"] = triton_mock
    sys.modules["triton.language"] = MagicMock()
    print("Warning: 'triton' module not found. Mocking it for Windows compatibility.")

try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM3 modules not available: {e}")
    # Try adding the potential path directly
    sam3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sam3"))
    if os.path.exists(sam3_path):
        sys.path.append(sam3_path)
        try:
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor
            SAM3_AVAILABLE = True
            print(f"Successfully imported SAM3 from {sam3_path}")
        except ImportError as e2:
            print(f"Still failed to import SAM3 after adding path: {e2}")
            SAM3_AVAILABLE = False
    else:
        SAM3_AVAILABLE = False

# Import autotracking modules
AUTOTRACKING_AVAILABLE = False
try:
    from seismic_attributes import SeismicAttributes
    from autotracker import Autotracker, TrackingMode
    AUTOTRACKING_AVAILABLE = True
except ImportError:
    print("Warning: Autotracking modules not available. Using fallback mode.")
    AUTOTRACKING_AVAILABLE = False

# Fallback autotracker class (copied from seismic_predictor.py for consistency)
if not AUTOTRACKING_AVAILABLE:
    from enum import Enum
    class TrackingMode(Enum):
        ATTRIBUTE_GUIDED = "attribute_guided"
        EDGE_BASED = "edge_based"
        PHASE_GUIDED = "phase_guided"
        SIMILARITY_GUIDED = "similarity_guided"
        HYBRID = "hybrid"

class FallbackAutotracker:
    """Simple fallback autotracker using existing SAM3 propagation."""
    def __init__(self, seismic_volume=None):
        self.seismic_volume = seismic_volume
    def set_seismic_volume(self, volume):
        self.seismic_volume = volume
    def auto_detect_seeds(self, slice_data, slice_idx, n_seeds=10, method='from_points', existing_points=None, existing_masks=None):
        return existing_points[:n_seeds] if existing_points else []
    def track_horizon_dp(self, start_slice_idx, start_points, direction='forward', max_slices=50, mode=None):
        return {'paths': {}, 'confidences': {}, 'start_slice': start_slice_idx, 'direction': direction, 'mode': str(mode)}
    def track_multiple_horizons(self, start_slice_idx, n_horizons=3, direction='forward', max_slices=50):
        return {'horizons': [], 'n_tracked': 0}
    def compute_tracking_quality(self, tracking_results):
        return {'mean_confidence': 0.5, 'path_smoothness': 0.5, 'attribute_consistency': 0.5, 'overall_quality': 0.5}


class SeismicPredictorSAM3:
    def __init__(self, demo_mode=False):
        """Initialize the Seismic Predictor with SAM3 model"""
        self.demo_mode = demo_mode or not SAM3_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"SAM3 Demo mode: {self.demo_mode}")

        # SAM3 models
        self.sam_model = None
        self.image_processor = None
        self.video_predictor = None

        # Seismic data
        self.seismic_volume = None
        self.current_slice_type = None
        self.current_slice_idx = None
        self.current_slice = None

        # State storage
        self.object_masks = {}  # {obj_id: {slice_key: mask}}
        self.inference_state = {} # {obj_id: state} for image inference
        self.video_sessions = {} # {obj_id: session_id} for video inference
        self.video_masks = {} # {frame_idx: {obj_id: mask}} - Storage for propagated video masks
        self.demo_state = {}
        
        # Horizon interpretation mode
        self.horizon_mode = True  # Default to horizon mode for seismic interpretation
        self.horizon_lines = {}   # {obj_id: {slice_key: [(x,y), ...]}} - Store horizon lines
        self.horizon_thickness = 3  # Thickness for horizon line display

        # Autotracking
        self.attributes_processor = None
        self.autotracker = None
        self.auto_tracked_horizons = {}
        
        self._init_autotracker()

    def set_horizon_mode(self, enabled: bool):
        """Enable or disable horizon interpretation mode."""
        self.horizon_mode = enabled
        print(f"Horizon mode: {'enabled' if enabled else 'disabled'}")

    def _init_autotracker(self):
        if AUTOTRACKING_AVAILABLE:
            try:
                self.attributes_processor = SeismicAttributes(use_gpu=torch.cuda.is_available())
                self.autotracker = Autotracker(seismic_volume=self.seismic_volume,
                                             attributes_processor=self.attributes_processor)
                print("Full autotracking system initialized")
            except Exception as e:
                print(f"Failed to initialize full autotracking: {e}, using fallback")
                self.autotracker = FallbackAutotracker(seismic_volume=self.seismic_volume)
        else:
            self.autotracker = FallbackAutotracker(seismic_volume=self.seismic_volume)

    def load_model(self):
        """Load the SAM3 model"""
        if self.demo_mode:
            print("Running in demo mode - not loading actual SAM3 model")
            return True

        print("Loading SAM3 model...")
        try:
            # Load Image Model
            self.sam_model = build_sam3_image_model()
            self.sam_model.to(self.device)
            # Ensure model is in float32 to avoid bfloat16 issues
            self.sam_model = self.sam_model.float()
            self.image_processor = Sam3Processor(self.sam_model)
            
            # Load Video Predictor
            self.video_predictor = build_sam3_video_predictor()
            
            # Try to convert video predictor to float32 as well
            if hasattr(self.video_predictor, 'model'):
                try:
                    self.video_predictor.model = self.video_predictor.model.float()
                except Exception as e:
                    print(f"Warning: Could not convert video predictor to float32: {e}")
            
            print("SAM3 model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            if "401 Client Error" in str(e) or "GatedRepoError" in str(e):
                print("\n" + "="*80)
                print("AUTHENTICATION REQUIRED FOR SAM3 MODEL")
                print("="*80)
                print("The SAM3 model is gated and requires Hugging Face authentication.")
                print("Please run the following command in your terminal to log in:")
                print("    huggingface-cli login")
                print("\nAlternatively, if you have the checkpoint locally, you can modify")
                print("seismic_predictor_sam3.py to pass the 'checkpoint_path' argument")
                print("to build_sam3_image_model() and build_sam3_video_predictor().")
                print("="*80 + "\n")
            
            import traceback
            traceback.print_exc()
            self.demo_mode = True
            return False

    def set_seismic_volume(self, seismic_volume):
        """Set the seismic volume data"""
        self.seismic_volume = seismic_volume
        self.object_masks = {}
        self.video_masks = {} # Reset video masks when volume changes
        if self.autotracker and hasattr(self.autotracker, 'set_seismic_volume'):
            self.autotracker.set_seismic_volume(seismic_volume)
        elif self.autotracker:
            self.autotracker.seismic_volume = seismic_volume

    def set_current_slice(self, slice_type, slice_idx, slice_data):
        """Set the current working slice"""
        self.current_slice_type = slice_type
        self.current_slice_idx = slice_idx
        self.current_slice = slice_data

    def _normalize_slice(self, slice_data):
        """Normalize slice for SAM3 (expects PIL Image or similar, but we'll pass numpy array converted to RGB)"""
        # Clip to percentiles
        p_low, p_high = 1, 99
        low, high = np.percentile(slice_data, [p_low, p_high])
        slice_norm = np.clip(slice_data, low, high)
        
        # Normalize to 0-255
        slice_norm = ((slice_norm - low) / (high - low) * 255).astype(np.uint8)
        
        # Convert to RGB
        slice_rgb = np.stack([slice_norm, slice_norm, slice_norm], axis=2)
        
        # SAM3 processor expects PIL Image usually, but let's check if it accepts numpy
        from PIL import Image
        return Image.fromarray(slice_rgb)

    def predict_masks_from_points(self, points, point_labels, multimask_output=True):
        """Predict masks using SAM3 image processor"""
        if self.current_slice is None:
            raise ValueError("No slice data set")

        if self.demo_mode:
            return self._generate_demo_mask(points, point_labels)

        try:
            image = self._normalize_slice(self.current_slice)
            
            # SAM3 Image Processor Workflow
            inference_state = self.image_processor.set_image(image)
            
            # Prepare points
            # SAM3 expects points normalized to [0, 1]
            h, w = self.current_slice.shape
            points_norm = []
            for p in points:
                points_norm.append([p[0] / w, p[1] / h])
            
            points_tensor = torch.tensor(points_norm, dtype=torch.float32, device=self.device).unsqueeze(1) # (N, 1, 2)
            labels_tensor = torch.tensor(point_labels, dtype=torch.long, device=self.device).unsqueeze(1) # (N, 1)
            
            # Add points to geometric prompt
            # We need to access the internal geometric prompt object
            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = self.sam_model._get_dummy_prompt()
                
            # Use append_points from Prompt class
            inference_state["geometric_prompt"].append_points(points_tensor, labels_tensor)
            
            # Ensure text prompt is handled (dummy visual)
            if "language_features" not in inference_state["backbone_out"]:
                 dummy_text_outputs = self.sam_model.backbone.forward_text(
                    ["visual"], device=self.device
                )
                 inference_state["backbone_out"].update(dummy_text_outputs)
            
            # Run forward grounding
            output_state = self.image_processor._forward_grounding(inference_state)
            
            # Extract masks
            masks = output_state["masks"] # (N, H, W)
            scores = output_state["scores"]
            logits = output_state["masks_logits"]
            
            # Convert to numpy
            masks_np = masks.squeeze(1).cpu().numpy() # (N, H, W) or (1, H, W)?
            scores_np = scores.cpu().numpy()
            logits_np = logits.squeeze(1).cpu().numpy()
            
            # If we have multiple masks (ambiguity), SAM3 usually returns them.
            # Check shape.
            if len(masks_np.shape) == 2:
                masks_np = masks_np[None, :, :]
                logits_np = logits_np[None, :, :]
                
            return masks_np, scores_np, logits_np

        except Exception as e:
            print(f"SAM3 prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_demo_mask(points, point_labels)

    def _generate_demo_mask(self, points, point_labels):
        """Generate a demo mask (same as SAM2 predictor)"""
        h, w = self.current_slice.shape
        base_mask = np.zeros((h, w), dtype=bool)
        
        fg_points = [p for i, p in enumerate(points) if point_labels[i] == 1]
        
        if len(fg_points) >= 2:
            # Connect points for fault simulation
            fg_points.sort(key=lambda p: p[1])
            for i in range(len(fg_points) - 1):
                y1, x1 = int(fg_points[i][1]), int(fg_points[i][0])
                y2, x2 = int(fg_points[i+1][1]), int(fg_points[i+1][0])
                
                # Simple line drawing
                rr, cc = self._line(y1, x1, y2, x2)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                base_mask[rr[valid], cc[valid]] = True
                
            # Dilate
            from scipy import ndimage
            base_mask = ndimage.binary_dilation(base_mask, iterations=2)
        else:
            # Circles
            y_idx, x_idx = np.ogrid[:h, :w]
            for p in fg_points:
                y, x = int(p[1]), int(p[0])
                mask = ((y_idx - y)**2 + (x_idx - x)**2) <= 15**2
                base_mask |= mask

        masks = [base_mask, base_mask, base_mask] # 3 masks
        scores = [0.95, 0.85, 0.75]
        logits = np.zeros((3, h, w))
        
        return np.array(masks), np.array(scores), logits

    def _line(self, r0, c0, r1, c1):
        # Bresenham's line algorithm or similar
        # Using skimage.draw.line logic simplified
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        num = max(abs(r1 - r0), abs(c1 - c0)) + 1
        return np.linspace(r0, r1, num).astype(int), np.linspace(c0, c1, num).astype(int)

    # ===================== HORIZON INTERPRETATION METHODS =====================
    
    def predict_horizon_from_points(self, points: List[Tuple[float, float]], 
                                   point_labels: List[int],
                                   use_seismic_guidance: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
        """
        Predict a seismic horizon line from input points.
        
        Unlike standard SAM segmentation that produces blobs, this method:
        1. Uses SAM3 to understand the seismic context around points
        2. Fits a smooth horizon line through the foreground points
        3. Uses seismic attributes (edges, phase) to refine the path
        
        Args:
            points: List of (x, y) point coordinates
            point_labels: List of labels (1=foreground on horizon, 0=background)
            use_seismic_guidance: Whether to use seismic attributes for guidance
            
        Returns:
            Tuple of (horizon_mask, horizon_line_points, confidence)
        """
        if self.current_slice is None:
            raise ValueError("No slice data set")
            
        h, w = self.current_slice.shape
        
        # Get foreground points (horizon picks)
        fg_points = [(p[0], p[1]) for i, p in enumerate(points) if point_labels[i] == 1]
        
        if len(fg_points) < 2:
            # Not enough points for a line - use SAM3 blob and extract centerline
            masks, scores, logits = self.predict_masks_from_points(points, point_labels)
            best_mask = masks[np.argmax(scores)]
            horizon_line = self._extract_horizon_from_mask(best_mask)
            horizon_mask = self._create_line_mask(horizon_line, h, w)
            return horizon_mask, horizon_line, float(np.max(scores))
        
        # Sort points by x-coordinate for proper horizon interpolation
        fg_points.sort(key=lambda p: p[0])
        
        print(f"Horizon from {len(fg_points)} points: {fg_points}")
        
        # Fit a smooth spline through the points
        if use_seismic_guidance and self.attributes_processor is not None:
            # Use seismic-guided interpolation
            horizon_line = self._seismic_guided_interpolation(fg_points, self.current_slice)
        else:
            # Simple spline interpolation
            horizon_line = self._spline_interpolation(fg_points, w)
        
        print(f"Generated horizon line with {len(horizon_line)} points")
        
        # Create thin line mask from horizon line
        horizon_mask = self._create_line_mask(horizon_line, h, w)
        
        # Store horizon line for this object
        confidence = 0.95  # High confidence when user provides points
        
        return horizon_mask, horizon_line, confidence
    
    def _spline_interpolation(self, points: List[Tuple[float, float]], 
                              width: int, smoothing: float = 0.1) -> List[Tuple[int, int]]:
        """
        Fit a smooth spline through the given points and extend across full width.
        
        Args:
            points: List of (x, y) coordinates - picked horizon points
            width: Image width - horizon will extend across this
            smoothing: Spline smoothing factor (lower = closer to points)
            
        Returns:
            List of (x, y) coordinates along the horizon
        """
        if len(points) < 2:
            # Single point - create horizontal line at that y
            if len(points) == 1:
                y = int(points[0][1])
                return [(x, y) for x in range(width)]
            return []
        
        points = np.array(points, dtype=float)
        x = points[:, 0]
        y = points[:, 1]
        
        # Sort by x coordinate
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Remove duplicate x values (keep average y for duplicates)
        unique_x = []
        unique_y = []
        i = 0
        while i < len(x_sorted):
            curr_x = x_sorted[i]
            same_x_ys = [y_sorted[i]]
            j = i + 1
            while j < len(x_sorted) and abs(x_sorted[j] - curr_x) < 1:
                same_x_ys.append(y_sorted[j])
                j += 1
            unique_x.append(curr_x)
            unique_y.append(np.mean(same_x_ys))
            i = j
        
        x_unique = np.array(unique_x)
        y_unique = np.array(unique_y)
        
        if len(x_unique) < 2:
            y_val = int(y_unique[0])
            return [(x, y_val) for x in range(width)]
        
        print(f"Spline: {len(x_unique)} unique points from x={x_unique[0]:.0f} to x={x_unique[-1]:.0f}")
        
        try:
            # Use cubic spline with very low smoothing to pass near the points
            from scipy.interpolate import CubicSpline, interp1d
            
            # Use CubicSpline for natural interpolation through points
            # 'natural' boundary conditions for smooth extension
            spline = CubicSpline(x_unique, y_unique, bc_type='natural')
            
            # Generate points across FULL width (extend horizon)
            # Extend slightly beyond the data range
            x_min = max(0, int(x_unique[0]) - 20)
            x_max = min(width, int(x_unique[-1]) + 20)
            
            # For extrapolation beyond picked points, use linear extension
            x_new = np.arange(x_min, x_max)
            y_new = spline(x_new)
            
            # Extend to full width using the slope at the endpoints
            horizon_line = []
            
            # Left extension (before first point)
            if x_min > 0:
                left_slope = (y_unique[1] - y_unique[0]) / max(1, x_unique[1] - x_unique[0])
                for xi in range(0, x_min):
                    yi = y_unique[0] - left_slope * (x_unique[0] - xi)
                    yi = np.clip(yi, 0, self.current_slice.shape[0] - 1)
                    horizon_line.append((xi, int(yi)))
            
            # Main interpolated section
            y_new = np.clip(y_new, 0, self.current_slice.shape[0] - 1)
            for xi, yi in zip(x_new, y_new):
                horizon_line.append((int(xi), int(yi)))
            
            # Right extension (after last point)
            if x_max < width:
                right_slope = (y_unique[-1] - y_unique[-2]) / max(1, x_unique[-1] - x_unique[-2])
                for xi in range(x_max, width):
                    yi = y_unique[-1] + right_slope * (xi - x_unique[-1])
                    yi = np.clip(yi, 0, self.current_slice.shape[0] - 1)
                    horizon_line.append((xi, int(yi)))
            
            return horizon_line
            
        except Exception as e:
            print(f"Spline interpolation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple linear interpolation
            horizon_line = []
            for i in range(len(x_unique) - 1):
                x1, y1 = int(x_unique[i]), int(y_unique[i])
                x2, y2 = int(x_unique[i+1]), int(y_unique[i+1])
                for xi in range(x1, x2 + 1):
                    yi = y1 + (y2 - y1) * (xi - x1) / max(1, x2 - x1)
                    horizon_line.append((xi, int(yi)))
        
        return horizon_line
    
    def _seismic_guided_interpolation(self, points: List[Tuple[float, float]], 
                                       slice_data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Interpolate horizon using seismic attribute guidance.
        
        First creates a spline through points, then refines using edge detection
        to snap to nearby strong reflectors.
        
        Args:
            points: List of (x, y) horizon pick coordinates
            slice_data: 2D seismic slice data
            
        Returns:
            List of (x, y) coordinates along the guided horizon
        """
        if len(points) < 2:
            # Single point - extend horizontally
            if len(points) == 1:
                y = int(points[0][1])
                return [(x, y) for x in range(slice_data.shape[1])]
            return []
        
        h, w = slice_data.shape
        
        # First, get the basic spline interpolation
        base_horizon = self._spline_interpolation(points, w)
        
        if not base_horizon:
            return []
        
        # Try to refine using seismic attributes
        try:
            attributes = self.attributes_processor.compute_slice_attributes(
                slice_data, ['edge_strength']
            )
            edge_strength = attributes.get('edge_strength', np.zeros_like(slice_data))
            
            # Normalize
            edge_norm = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-10)
            
            # Refine horizon by snapping to nearby strong edges
            refined_horizon = []
            search_window = 5  # Search +/- 5 pixels vertically
            
            for x, y in base_horizon:
                if x < 0 or x >= w:
                    continue
                    
                y_min = max(0, y - search_window)
                y_max = min(h, y + search_window + 1)
                
                if y_min >= y_max:
                    refined_horizon.append((x, y))
                    continue
                
                # Find strongest edge in the search window
                window = edge_norm[y_min:y_max, x]
                best_offset = np.argmax(window)
                new_y = y_min + best_offset
                
                # Apply smoothness constraint
                if refined_horizon:
                    prev_y = refined_horizon[-1][1]
                    max_jump = 3
                    if abs(new_y - prev_y) > max_jump:
                        new_y = prev_y + np.sign(new_y - prev_y) * max_jump
                
                refined_horizon.append((x, int(new_y)))
            
            return refined_horizon if refined_horizon else base_horizon
            
        except Exception as e:
            print(f"Seismic-guided refinement failed: {e}")
            return base_horizon
    
    def _dp_path_finding(self, x1: int, y1: int, x2: int, y2: int, 
                         cost_map: np.ndarray, h: int, 
                         max_slope: int = 5) -> List[Tuple[int, int]]:
        """
        Find optimal path between two points using dynamic programming.
        
        Args:
            x1, y1: Start point
            x2, y2: End point  
            cost_map: 2D array where lower values indicate better horizon path
            h: Height of the image
            max_slope: Maximum allowed vertical change per pixel
            
        Returns:
            List of (x, y) coordinates along the optimal path
        """
        if x1 == x2:
            # Vertical line
            return [(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)]
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        width = x2 - x1 + 1
        
        # DP table: cost[x_offset][y] = minimum cost to reach position (x1+x_offset, y)
        # We use inverted cost (1 - edge_strength) so that strong edges have low cost
        inverted_cost = 1.0 - cost_map
        
        # Initialize
        INF = float('inf')
        dp = np.full((width, h), INF)
        parent = np.full((width, h), -1, dtype=int)
        
        # Start position
        dp[0, y1] = inverted_cost[y1, x1]
        
        # Fill DP table
        for dx in range(1, width):
            x = x1 + dx
            if x >= cost_map.shape[1]:
                break
                
            for y in range(h):
                # Look at previous column within slope constraint
                for prev_y in range(max(0, y - max_slope), min(h, y + max_slope + 1)):
                    if dp[dx-1, prev_y] < INF:
                        new_cost = dp[dx-1, prev_y] + inverted_cost[y, x]
                        if new_cost < dp[dx, y]:
                            dp[dx, y] = new_cost
                            parent[dx, y] = prev_y
        
        # Backtrack from end point
        path = []
        x_offset = width - 1
        y = y2
        
        # If exact end point not reachable, find closest reachable y
        if dp[x_offset, y2] == INF:
            valid_ys = np.where(dp[x_offset] < INF)[0]
            if len(valid_ys) == 0:
                # Fallback to straight line
                return [(x, int(y1 + (y2 - y1) * (x - x1) / max(1, x2 - x1))) 
                        for x in range(x1, x2 + 1)]
            y = valid_ys[np.argmin(np.abs(valid_ys - y2))]
        
        while x_offset >= 0:
            path.append((x1 + x_offset, y))
            if x_offset == 0:
                break
            y = parent[x_offset, y]
            if y < 0:
                break
            x_offset -= 1
        
        path.reverse()
        return path
    
    def _extract_horizon_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract a horizon line from a blob mask using skeletonization.
        
        This converts SAM's blob output into a thin line suitable for 
        seismic horizon interpretation.
        
        Args:
            mask: 2D boolean mask from SAM
            
        Returns:
            List of (x, y) coordinates along the horizon line
        """
        if not np.any(mask):
            return []
        
        h, w = mask.shape
        
        try:
            # Skeletonize the mask to get centerline
            skeleton = skeletonize(mask.astype(bool))
            
            # Get skeleton coordinates
            y_coords, x_coords = np.where(skeleton)
            
            if len(x_coords) == 0:
                # Fallback: use center of mass for each column
                return self._extract_centerline_simple(mask)
            
            # Sort by x-coordinate for ordered horizon line
            sorted_indices = np.argsort(x_coords)
            horizon_line = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
            
            # Remove duplicate x values (keep the one closest to median y)
            x_to_ys = {}
            for x, y in horizon_line:
                if x not in x_to_ys:
                    x_to_ys[x] = []
                x_to_ys[x].append(y)
            
            # For each x, pick the median y
            unique_line = []
            for x in sorted(x_to_ys.keys()):
                y = int(np.median(x_to_ys[x]))
                unique_line.append((x, y))
            
            return unique_line
            
        except Exception as e:
            print(f"Skeletonization failed: {e}")
            return self._extract_centerline_simple(mask)
    
    def _extract_centerline_simple(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Simple centerline extraction using column-wise center of mass.
        
        Args:
            mask: 2D boolean mask
            
        Returns:
            List of (x, y) coordinates along the centerline
        """
        h, w = mask.shape
        horizon_line = []
        
        for x in range(w):
            column = mask[:, x]
            if np.any(column):
                y_indices = np.where(column)[0]
                y_center = int(np.mean(y_indices))
                horizon_line.append((x, y_center))
        
        return horizon_line
    
    def _create_line_mask(self, horizon_line: List[Tuple[int, int]], 
                          height: int, width: int) -> np.ndarray:
        """
        Create a thin line mask from horizon coordinates.
        
        Args:
            horizon_line: List of (x, y) coordinates
            height: Mask height
            width: Mask width
            
        Returns:
            2D boolean mask with thin horizon line
        """
        mask = np.zeros((height, width), dtype=bool)
        
        if not horizon_line:
            return mask
        
        # Draw the horizon line
        for x, y in horizon_line:
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = True
        
        # Optionally dilate for visibility (controlled by horizon_thickness)
        if self.horizon_thickness > 1:
            from scipy.ndimage import binary_dilation
            iterations = (self.horizon_thickness - 1) // 2
            if iterations > 0:
                mask = binary_dilation(mask, iterations=iterations)
        
        return mask
    
    def get_horizon_line(self, obj_id: int, slice_type: str, slice_idx: int) -> Optional[List[Tuple[int, int]]]:
        """
        Get stored horizon line for an object on a specific slice.
        
        Args:
            obj_id: Object ID
            slice_type: Type of slice (inline, crossline, timeslice)
            slice_idx: Slice index
            
        Returns:
            List of (x, y) coordinates or None if not found
        """
        slice_key = f"{slice_type}_{slice_idx}"
        
        # First check explicit horizon_lines storage
        if obj_id in self.horizon_lines:
            if slice_key in self.horizon_lines[obj_id]:
                return self.horizon_lines[obj_id][slice_key]
        
        # Check demo state for propagated horizon lines - use slice_idx directly
        if obj_id in self.demo_state:
            state = self.demo_state[obj_id]
            if 'inference_state' in state and 'horizon_lines' in state['inference_state']:
                # horizon_lines now uses slice_idx as key directly
                if slice_idx in state['inference_state']['horizon_lines']:
                    return state['inference_state']['horizon_lines'][slice_idx]
        
        # Try to extract from mask if available
        if obj_id in self.object_masks and slice_key in self.object_masks[obj_id]:
            mask = self.object_masks[obj_id][slice_key]
            return self._extract_horizon_from_mask(mask)
        
        # Try to extract from video_masks
        if hasattr(self, 'video_masks') and slice_idx in self.video_masks:
            if obj_id in self.video_masks[slice_idx]:
                mask = self.video_masks[slice_idx][obj_id]
                return self._extract_horizon_from_mask(mask)
        
        return None
    
    def store_horizon_line(self, obj_id: int, slice_type: str, slice_idx: int, 
                          horizon_line: List[Tuple[int, int]]):
        """Store a horizon line for later retrieval."""
        if obj_id not in self.horizon_lines:
            self.horizon_lines[obj_id] = {}
        slice_key = f"{slice_type}_{slice_idx}"
        self.horizon_lines[obj_id][slice_key] = horizon_line
    
    def mask_to_horizon_line(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Public method to convert any mask to a horizon line.
        Useful for converting SAM blob output to seismic horizon.
        """
        return self._extract_horizon_from_mask(mask)

    # ===================== END HORIZON INTERPRETATION METHODS =====================

    def store_mask_for_object(self, slice_type, slice_idx, obj_id, mask):
        if obj_id not in self.object_masks:
            self.object_masks[obj_id] = {}
        slice_key = f"{slice_type}_{slice_idx}"
        self.object_masks[obj_id][slice_key] = mask

    def has_masks_for_object(self, obj_id):
        if obj_id in self.object_masks and self.object_masks[obj_id]:
            return True
        # Check video sessions
        if obj_id in self.video_sessions:
            return True
        return False

    def init_video_predictor(self, slice_indices, obj_id):
        """Initialize video predictor for SAM3"""
        if self.demo_mode:
            if obj_id not in self.demo_state: self.demo_state[obj_id] = {}
            self.demo_state[obj_id]['slice_indices'] = slice_indices
            return True

        # SAM3 Video Predictor expects a path to video or folder of images.
        # Since we have in-memory slices, we might need to dump them to a temp folder.
        
        temp_dir = tempfile.mkdtemp(prefix=f"sam3_seismic_obj{obj_id}_")
        self.video_sessions[obj_id] = {"temp_dir": temp_dir, "slice_indices": slice_indices}
        
        print(f"Dumping slices to {temp_dir} for SAM3...")
        for i, idx in enumerate(slice_indices):
            if self.current_slice_type == "inline":
                slice_data = self.seismic_volume.get_inline_slice(idx)
            # ... handle other types ...
            elif self.current_slice_type == "crossline":
                slice_data = self.seismic_volume.get_crossline_slice(idx)
            else:
                slice_data = self.seismic_volume.get_timeslice(idx)
                
            # Normalize and save
            img = self._normalize_slice(slice_data)
            img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))

        # Start session
        try:
            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_dir,
                )
            )
            self.video_sessions[obj_id]["session_id"] = response["session_id"]
            print(f"SAM3 Video Session started: {response['session_id']}")
            return True
        except Exception as e:
            print(f"SAM3 video session initialization failed: {e}")
            # Clean up temp directory if session failed to start
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

    def add_point_to_video(self, frame_idx, obj_id, points, labels):
        """
        Add a point prompt to a specific frame in the video sequence.
        
        For SAM3, we use a two-step approach:
        1. First use TEXT prompt to detect horizons (SAM3's strength)
        2. Then propagate through the video
        
        Points are used to help localize the horizon of interest.
        """
        if self.demo_mode:
            return self._demo_add_point_to_video(frame_idx, obj_id, points, labels)

        if obj_id not in self.video_sessions:
            print(f"Error: No video session for object {obj_id}")
            return False
            
        session_id = self.video_sessions[obj_id]["session_id"]
        
        try:
            # Get image dimensions for coordinate conversion
            h, w = self.current_slice.shape
            
            # Convert pixel coordinates to relative coordinates (0-1 range) as SAM3 expects
            points_rel = [[float(p[0]) / w, float(p[1]) / h] for p in points]
            
            # Strategy: Use TEXT prompt first (SAM3's detection capability)
            # Then propagate, which will track the detected objects
            
            # Calculate approximate y-position of the horizon from points
            fg_points = [(pt[0], pt[1]) for pt, lbl in zip(points, labels) if lbl == 1]
            if not fg_points:
                print("No foreground points provided")
                return False
                
            # Get average Y position to help SAM3 locate the horizon area
            avg_y = np.mean([p[1] for p in fg_points])
            avg_y_rel = avg_y / h
            
            print(f"SAM3: Using text prompt for horizon detection on frame {frame_idx}")
            print(f"  Horizon approximate position: y={avg_y:.0f} ({avg_y_rel:.2%} from top)")
            
            # Use text prompt - SAM3 excels at understanding natural language
            # "horizontal line" or "seismic reflector" should work well
            request = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": "horizontal line",  # SAM3 text prompt
            }
            
            print(f"SAM3 text request: {request}")
            response = self.video_predictor.handle_request(request)
            print(f"SAM3 response: {response.keys() if response else 'None'}")
            
            if response and "outputs" in response:
                outputs = response["outputs"]
                out_obj_ids = outputs.get("out_obj_ids", [])
                out_masks = outputs.get("out_binary_masks", [])
                
                print(f"SAM3 detected {len(out_obj_ids)} objects")
                
                # Find the object closest to our picked points
                if len(out_obj_ids) > 0 and len(out_masks) > 0:
                    # Find which detected mask best matches our horizon points
                    best_match_idx = self._find_best_matching_mask(
                        out_masks, fg_points, h, w
                    )
                    
                    if best_match_idx >= 0:
                        matched_obj_id = out_obj_ids[best_match_idx]
                        matched_mask = out_masks[best_match_idx]
                        
                        # Convert mask to numpy
                        if hasattr(matched_mask, 'cpu'):
                            matched_mask = matched_mask.cpu().numpy()
                        if matched_mask.ndim > 2:
                            matched_mask = matched_mask.squeeze()
                        
                        # Store this mask
                        if frame_idx not in self.video_masks:
                            self.video_masks[frame_idx] = {}
                        self.video_masks[frame_idx][obj_id] = matched_mask.astype(bool)
                        
                        # Map our obj_id to SAM3's detected obj_id
                        self.video_sessions[obj_id]["sam3_obj_id"] = matched_obj_id
                        
                        # Extract horizon line
                        horizon_line = self._extract_horizon_from_mask(matched_mask)
                        if horizon_line:
                            if obj_id not in self.horizon_lines:
                                self.horizon_lines[obj_id] = {}
                            slice_key = f"{self.current_slice_type}_{frame_idx}"
                            self.horizon_lines[obj_id][slice_key] = horizon_line
                            print(f"SAM3 matched object {matched_obj_id}, extracted {len(horizon_line)} horizon points")
                        
                        return True
                    else:
                        print("SAM3 detected objects but none matched the picked points")
                else:
                    print("SAM3 did not detect any objects with text prompt")
            
            # If text prompt failed, fall back to demo mode
            print("Falling back to demo mode for this object")
            self.demo_mode = True
            return self._demo_add_point_to_video(frame_idx, obj_id, points, labels)
            
        except Exception as e:
            print(f"Error with SAM3 add_prompt: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to demo mode
            print("Falling back to demo mode")
            self.demo_mode = True
            return self._demo_add_point_to_video(frame_idx, obj_id, points, labels)
    
    def _find_best_matching_mask(self, masks, points, h, w):
        """Find which mask best matches the user-picked points."""
        best_idx = -1
        best_score = -1
        
        for idx, mask in enumerate(masks):
            if hasattr(mask, 'cpu'):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # Resize mask if needed
            if mask_np.shape != (h, w):
                from scipy.ndimage import zoom
                zoom_h = h / mask_np.shape[0]
                zoom_w = w / mask_np.shape[1]
                mask_np = zoom(mask_np.astype(float), (zoom_h, zoom_w), order=0) > 0.5
            
            # Score: how many points are near the mask
            score = 0
            for px, py in points:
                px, py = int(px), int(py)
                # Check if point is on or near the mask
                for dy in range(-10, 11):
                    for dx in range(-10, 11):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if mask_np[ny, nx]:
                                score += 1
                                break
                    else:
                        continue
                    break
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _demo_add_point_to_video(self, frame_idx, obj_id, points, labels):
        """Demo mode add point - creates horizon line from points."""
        if frame_idx not in self.video_masks:
            self.video_masks[frame_idx] = {}
        
        h, w = self.current_slice.shape
        
        # Get foreground points
        fg_points = [(pt[0], pt[1]) for pt, lbl in zip(points, labels) if lbl == 1]
        
        if len(fg_points) >= 2:
            # Create a horizon line through the points
            horizon_line = self._spline_interpolation(fg_points, w)
            mask = self._create_line_mask(horizon_line, h, w)
            
            # Store both mask and horizon line
            self.video_masks[frame_idx][obj_id] = mask
                
            # Store horizon line for later retrieval
            if obj_id not in self.horizon_lines:
                self.horizon_lines[obj_id] = {}
            slice_key = f"{self.current_slice_type}_{frame_idx}"
            self.horizon_lines[obj_id][slice_key] = horizon_line
            print(f"Demo mode: stored horizon line with {len(horizon_line)} points for frame {frame_idx}")
        
        return True

    def propagate_masks(self, obj_id, start_frame_idx=None, max_frames=None, reverse=False):
        """
        Propagate masks through the video sequence.
        """
        print(f"propagate_masks called: obj_id={obj_id}, reverse={reverse}, demo_mode={self.demo_mode}")
        
        if self.demo_mode:
            print("  -> Using demo mode propagation")
            return self._generate_demo_propagated_masks(obj_id, reverse)

        if obj_id not in self.video_sessions:
            print(f"No video session for object {obj_id}, using demo propagation")
            return self._generate_demo_propagated_masks(obj_id, reverse)
            
        try:
            direction = "backward" if reverse else "forward"
            session_id = self.video_sessions[obj_id]["session_id"]
            
            print(f"Starting SAM3 propagation {direction} from frame {start_frame_idx}, session={session_id}")
            
            request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
            }
            
            if start_frame_idx is not None:
                request["start_frame_index"] = start_frame_idx
                
            if max_frames is not None:
                request["max_frame_num_to_track"] = max_frames
            
            print(f"SAM3 propagation request: {request}")
            
            # Use autocast to handle dtype issues
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                generator = self.video_predictor.handle_stream_request(request)
                
                count = 0
                for item in generator:
                    frame_idx = item.get("frame_index", -1)
                    outputs = item.get("outputs")
                    if outputs is not None:
                        self._update_masks_from_outputs(frame_idx, outputs, obj_id)
                        count += 1
                
            print(f"SAM3 Propagated {count} frames with masks")
            return True
            
        except RuntimeError as e:
            if "BFloat16" in str(e) or "Input type" in str(e):
                print(f"SAM3 dtype error during propagation - falling back to demo mode: {e}")
            else:
                print(f"Error propagating masks: {e}")
                import traceback
                traceback.print_exc()
            
            # Fall back to demo propagation
            print("Falling back to demo propagation...")
            self.demo_mode = True
            
            # Initialize demo state if needed
            if obj_id not in self.demo_state:
                self.demo_state[obj_id] = {}
            if 'slice_indices' not in self.demo_state[obj_id]:
                # Get slice indices from video session if available
                if obj_id in self.video_sessions and 'slice_indices' in self.video_sessions[obj_id]:
                    self.demo_state[obj_id]['slice_indices'] = self.video_sessions[obj_id]['slice_indices']
                else:
                    # Generate full range of slices
                    if self.current_slice_type == "inline" and self.seismic_volume:
                        slice_count = len(self.seismic_volume.inlines) if hasattr(self.seismic_volume, 'inlines') else 651
                    elif self.current_slice_type == "crossline" and self.seismic_volume:
                        slice_count = len(self.seismic_volume.crosslines) if hasattr(self.seismic_volume, 'crosslines') else 951
                    else:
                        slice_count = 462  # Default for timeslice
                    self.demo_state[obj_id]['slice_indices'] = list(range(slice_count))
            
            return self._generate_demo_propagated_masks(obj_id, reverse)
            
        except Exception as e:
            print(f"Error propagating masks: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to demo propagation
            print("Falling back to demo propagation...")
            self.demo_mode = True
            
            # Initialize demo state if needed
            if obj_id not in self.demo_state:
                self.demo_state[obj_id] = {}
            if 'slice_indices' not in self.demo_state[obj_id]:
                if obj_id in self.video_sessions and 'slice_indices' in self.video_sessions[obj_id]:
                    self.demo_state[obj_id]['slice_indices'] = self.video_sessions[obj_id]['slice_indices']
                else:
                    if self.current_slice_type == "inline" and self.seismic_volume:
                        slice_count = len(self.seismic_volume.inlines) if hasattr(self.seismic_volume, 'inlines') else 651
                    elif self.current_slice_type == "crossline" and self.seismic_volume:
                        slice_count = len(self.seismic_volume.crosslines) if hasattr(self.seismic_volume, 'crosslines') else 951
                    else:
                        slice_count = 462
                    self.demo_state[obj_id]['slice_indices'] = list(range(slice_count))
            
            return self._generate_demo_propagated_masks(obj_id, reverse)

    def _update_masks_from_outputs(self, frame_idx, outputs, target_obj_id):
        """Helper to extract masks from SAM3 outputs and store them.
        
        Maps SAM3's detected object IDs to our target object ID.
        Also extracts horizon lines from masks for better visualization.
        """
        if outputs is None:
            return

        out_obj_ids = outputs.get("out_obj_ids", [])
        out_binary_masks = outputs.get("out_binary_masks", [])
        
        if len(out_obj_ids) == 0:
            return
            
        # Initialize video masks storage if needed
        if not hasattr(self, 'video_masks'):
            self.video_masks = {}
            
        if frame_idx not in self.video_masks:
            self.video_masks[frame_idx] = {}
        
        # Get the SAM3 object ID that was matched to our target_obj_id
        sam3_obj_id = None
        if target_obj_id in self.video_sessions and "sam3_obj_id" in self.video_sessions[target_obj_id]:
            sam3_obj_id = self.video_sessions[target_obj_id]["sam3_obj_id"]
            
        for i, obj_id in enumerate(out_obj_ids):
            if i < len(out_binary_masks):
                mask = out_binary_masks[i]
                # Convert to numpy bool array if it's a tensor
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                mask = mask.astype(bool)
                
                # Map SAM3 obj_id to our target_obj_id
                store_obj_id = target_obj_id if (sam3_obj_id is not None and obj_id == sam3_obj_id) else obj_id
                self.video_masks[frame_idx][store_obj_id] = mask
                
                # Extract horizon line from mask for horizon mode display
                if self.horizon_mode:
                    horizon_line = self._extract_horizon_from_mask(mask)
                    if horizon_line and len(horizon_line) > 1:
                        if store_obj_id not in self.horizon_lines:
                            self.horizon_lines[store_obj_id] = {}
                        slice_key = f"{self.current_slice_type}_{frame_idx}"
                        self.horizon_lines[store_obj_id][slice_key] = horizon_line

    def get_mask_for_frame(self, frame_idx, obj_id):
        """Get the propagated mask for a specific frame and object.
        
        Args:
            frame_idx: The actual slice index (e.g., inline number)
            obj_id: Object ID
            
        Returns:
            Boolean mask or None
        """
        # First check video_masks (used by both real SAM3 and fallback)
        if hasattr(self, 'video_masks') and frame_idx in self.video_masks:
            if obj_id in self.video_masks[frame_idx]:
                return self.video_masks[frame_idx][obj_id]
        
        # Check demo state
        if obj_id in self.demo_state:
            state = self.demo_state[obj_id]
            
            # Try to generate if not already done
            if 'inference_state' not in state or 'output' not in state.get('inference_state', {}):
                if 'slice_indices' in state:
                    self._generate_demo_propagated_masks(obj_id)
            
            if 'inference_state' in state and 'output' in state['inference_state']:
                output = state['inference_state']['output']
                
                # The output uses SLICE INDEX as key (not position)
                if frame_idx in output:
                    mask = output[frame_idx]
                    # Ensure mask matches current slice dimensions
                    if self.current_slice is not None and mask.shape != self.current_slice.shape:
                        from scipy.ndimage import zoom
                        h, w = self.current_slice.shape
                        zoom_h = h / mask.shape[0]
                        zoom_w = w / mask.shape[1]
                        mask = zoom(mask.astype(float), (zoom_h, zoom_w), order=0) > 0.5
                    return mask
        
        return None

    def _generate_demo_propagated_masks(self, obj_id, reverse=False):
        """Generate propagated horizon masks for demo mode.
        
        Instead of generating random blobs, this propagates the initial horizon line
        across slices using seismic attribute guidance when available.
        """
        print(f"_generate_demo_propagated_masks called for obj_id={obj_id}, reverse={reverse}")
        
        # Initialize demo state if not present
        if obj_id not in self.demo_state:
            print(f"  Creating demo_state for object {obj_id}")
            self.demo_state[obj_id] = {}
            
        state = self.demo_state[obj_id]
        
        # If no slice_indices, try to get them
        if 'slice_indices' not in state:
            print(f"  No slice_indices in demo_state, initializing...")
            if obj_id in self.video_sessions and 'slice_indices' in self.video_sessions[obj_id]:
                state['slice_indices'] = self.video_sessions[obj_id]['slice_indices']
            else:
                # Generate full range
                if self.current_slice_type == "inline" and self.seismic_volume:
                    slice_count = len(self.seismic_volume.inlines) if hasattr(self.seismic_volume, 'inlines') else 651
                elif self.current_slice_type == "crossline" and self.seismic_volume:
                    slice_count = len(self.seismic_volume.crosslines) if hasattr(self.seismic_volume, 'crosslines') else 951
                else:
                    slice_count = 462
                state['slice_indices'] = list(range(slice_count))
                print(f"  Generated {slice_count} slice indices")
            
        slice_indices = state['slice_indices']
        print(f"  Processing {len(slice_indices)} slices")
        
        # Create propagated masks storage
        if 'inference_state' not in state:
            state['inference_state'] = {'output': {}, 'horizon_lines': {}}
        
        # Get the initial horizon line from stored data
        initial_horizon_line = None
        initial_slice_idx = None
        
        # Find the initial horizon line from horizon_lines storage
        if obj_id in self.horizon_lines:
            for slice_key, horizon_line in self.horizon_lines[obj_id].items():
                if horizon_line and len(horizon_line) > 1:
                    initial_horizon_line = horizon_line
                    try:
                        initial_slice_idx = int(slice_key.split('_')[-1])
                    except:
                        initial_slice_idx = 0
                    print(f"  Found horizon line in horizon_lines: {len(horizon_line)} points from slice {initial_slice_idx}")
                    break
        
        # If no horizon line found, try to extract from object_masks
        if initial_horizon_line is None and obj_id in self.object_masks:
            for slice_key, mask in self.object_masks[obj_id].items():
                if mask is not None and np.any(mask):
                    initial_horizon_line = self._extract_horizon_from_mask(mask)
                    if initial_horizon_line and len(initial_horizon_line) > 1:
                        try:
                            initial_slice_idx = int(slice_key.split('_')[-1])
                        except:
                            initial_slice_idx = 0
                        print(f"  Extracted horizon line from object_masks: {len(initial_horizon_line)} points")
                        break
        
        if initial_horizon_line is None or len(initial_horizon_line) < 2:
            print(f"  No initial horizon line found for object {obj_id}, using fallback horizontal line")
            h, w = self.current_slice.shape if self.current_slice is not None else (500, 500)
            initial_horizon_line = [(x, h // 2) for x in range(w)]
            initial_slice_idx = 0
        
        h, w = self.current_slice.shape if self.current_slice is not None else (500, 500)
        
        print(f"Propagating horizon with {len(initial_horizon_line)} points from slice {initial_slice_idx}")
        
        # CRITICAL: Store the template waveform from the ORIGINAL slice
        # This is what allows accurate tracking across large changes
        try:
            if self.seismic_volume is not None:
                if self.current_slice_type == "inline":
                    original_slice_data = self.seismic_volume.get_inline_slice(initial_slice_idx)
                elif self.current_slice_type == "crossline":
                    original_slice_data = self.seismic_volume.get_crossline_slice(initial_slice_idx)
                else:
                    original_slice_data = self.seismic_volume.get_timeslice(initial_slice_idx)
                
                # Store templates for waveform correlation tracking
                self._tracking_templates = {
                    'slice_data': original_slice_data.copy(),
                    'horizon': initial_horizon_line,
                    'slice_idx': initial_slice_idx,
                    'obj_id': obj_id
                }
                print(f"  Stored waveform templates from slice {initial_slice_idx}")
            else:
                self._tracking_templates = None
        except Exception as e:
            print(f"  Could not store tracking templates: {e}")
            self._tracking_templates = None
        
        # Store the initial horizon at its slice index
        if initial_slice_idx in slice_indices:
            horizon_mask = self._create_line_mask(initial_horizon_line, h, w)
            state['inference_state']['output'][initial_slice_idx] = horizon_mask
            state['inference_state']['horizon_lines'] = state['inference_state'].get('horizon_lines', {})
            state['inference_state']['horizon_lines'][initial_slice_idx] = initial_horizon_line
        
        # Sort slices by distance from initial
        forward_slices = [idx for idx in slice_indices if idx > initial_slice_idx]
        backward_slices = [idx for idx in slice_indices if idx < initial_slice_idx]
        forward_slices.sort()
        backward_slices.sort(reverse=True)
        
        # SIMPLE INCREMENTAL TRACKING - each slice uses previous as base
        # This allows natural drift following the horizon
        
        # Process forward direction
        print(f"  Processing {len(forward_slices)} forward slices...")
        current_horizon = initial_horizon_line
        
        for i, idx in enumerate(forward_slices):
            try:
                if self.current_slice_type == "inline":
                    slice_data = self.seismic_volume.get_inline_slice(idx)
                elif self.current_slice_type == "crossline":
                    slice_data = self.seismic_volume.get_crossline_slice(idx)
                else:
                    slice_data = self.seismic_volume.get_timeslice(idx)
                
                # Propagate from PREVIOUS slice (incremental)
                propagated_horizon = self._propagate_horizon_to_slice(
                    current_horizon, 1, slice_data, h, w
                )
                current_horizon = propagated_horizon  # Update for next iteration
                
            except Exception as e:
                propagated_horizon = current_horizon
            
            # Store result
            horizon_mask = self._create_line_mask(propagated_horizon, h, w)
            state['inference_state']['output'][idx] = horizon_mask
            state['inference_state']['horizon_lines'][idx] = propagated_horizon
            
            # Progress update every 100 slices
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(forward_slices)} forward slices")
        
        # Process backward direction
        print(f"  Processing {len(backward_slices)} backward slices...")
        current_horizon = initial_horizon_line
        
        for i, idx in enumerate(backward_slices):
            try:
                if self.current_slice_type == "inline":
                    slice_data = self.seismic_volume.get_inline_slice(idx)
                elif self.current_slice_type == "crossline":
                    slice_data = self.seismic_volume.get_crossline_slice(idx)
                else:
                    slice_data = self.seismic_volume.get_timeslice(idx)
                
                propagated_horizon = self._propagate_horizon_to_slice(
                    current_horizon, -1, slice_data, h, w
                )
                current_horizon = propagated_horizon
                
            except Exception as e:
                propagated_horizon = current_horizon
            
            # Store result
            horizon_mask = self._create_line_mask(propagated_horizon, h, w)
            state['inference_state']['output'][idx] = horizon_mask
            state['inference_state']['horizon_lines'][idx] = propagated_horizon
            
            # Progress update every 100 slices
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(backward_slices)} backward slices")
        
        print(f"Demo propagation complete: generated masks for {len(state['inference_state']['output'])} slices")
        return True

    def _propagate_horizon_to_slice(self, base_horizon: List[Tuple[int, int]], 
                                    slice_offset: int,
                                    slice_data: np.ndarray,
                                    height: int, width: int) -> List[Tuple[int, int]]:
        """
        Propagate a horizon line to a new slice using seismic guidance.
        
        Args:
            base_horizon: The reference horizon line [(x, y), ...]
            slice_offset: Number of slices from the reference (positive = forward)
            slice_data: The seismic data for the target slice (can be None)
            height, width: Dimensions of the target slice
            
        Returns:
            Propagated horizon line for the target slice
        """
        if not base_horizon:
            return []
        
        # Convert to numpy for easier manipulation
        base_pts = np.array(base_horizon)
        
        # Use FAST and SIMPLE edge-snapping tracking
        if slice_data is not None:
            try:
                return self._fast_edge_snap_tracking(base_pts, slice_data, height, width)
            except Exception as e:
                print(f"Fast tracking failed: {e}")
        
        # Simple fallback: just copy the horizon
        return [(int(x), int(y)) for x, y in base_horizon]
    
    def _fast_edge_snap_tracking(self, base_pts: np.ndarray,
                                  slice_data: np.ndarray,
                                  height: int, width: int) -> List[Tuple[int, int]]:
        """
        FAST horizon tracking using simple edge snapping.
        
        This is optimized for speed while still being accurate:
        1. Compute edge strength once (fast)
        2. For each point, snap to nearest strong edge
        3. Apply smoothness constraint
        """
        from scipy.ndimage import sobel, gaussian_filter, gaussian_filter1d
        
        # Compute edge strength ONCE (vertical Sobel = horizontal edges = reflectors)
        smoothed = gaussian_filter(slice_data.astype(float), sigma=1)
        edges = np.abs(sobel(smoothed, axis=0))
        
        # Normalize
        edges = edges / (edges.max() + 1e-10)
        
        # Parameters
        search_window = 20  # Search ±20 pixels
        
        propagated = []
        prev_y = None
        
        for x, y in base_pts:
            x, y = int(x), int(y)
            if x < 0 or x >= width:
                continue
            
            # Search window
            y_min = max(0, y - search_window)
            y_max = min(height, y + search_window + 1)
            
            if y_max <= y_min:
                new_y = y
            else:
                # Get edge strength in window
                window = edges[y_min:y_max, x]
                
                # Weight by distance from expected position (prefer staying close)
                distances = np.abs(np.arange(len(window)) - (y - y_min))
                weights = np.exp(-distances / 8.0)  # Decay factor
                
                # Combined score
                scores = window * weights
                
                # Find best
                best_idx = np.argmax(scores)
                new_y = y_min + best_idx
            
            # Smoothness: limit jump from previous point
            if prev_y is not None:
                max_jump = 5
                if abs(new_y - prev_y) > max_jump:
                    new_y = int(prev_y + np.sign(new_y - prev_y) * max_jump)
            
            new_y = int(np.clip(new_y, 0, height - 1))
            propagated.append((x, new_y))
            prev_y = new_y
        
        # Light smoothing
        if len(propagated) > 5:
            xs = [p[0] for p in propagated]
            ys = np.array([p[1] for p in propagated])
            ys_smooth = gaussian_filter1d(ys.astype(float), sigma=2)
            propagated = [(int(x), int(np.clip(y, 0, height-1))) for x, y in zip(xs, ys_smooth)]
        
        return propagated
    
    def _professional_horizon_tracking(self, base_pts: np.ndarray,
                                        slice_data: np.ndarray,
                                        height: int, width: int) -> List[Tuple[int, int]]:
        """
        Professional-grade horizon tracking using multiple seismic attributes.
        
        This implements techniques used in commercial seismic interpretation software:
        1. Instantaneous phase for phase-consistent tracking
        2. Multi-trace semblance for lateral coherence
        3. Normalized cross-correlation with quality thresholds
        4. Dip estimation for structural guidance
        5. Robust outlier rejection
        """
        from scipy.ndimage import gaussian_filter1d, median_filter, sobel
        from scipy.signal import hilbert, correlate
        
        # Get template data
        if not hasattr(self, '_tracking_templates') or self._tracking_templates is None:
            return self._stable_horizon_tracking(base_pts, slice_data, height, width)
        
        template_data = self._tracking_templates.get('slice_data')
        template_horizon = self._tracking_templates.get('horizon')
        
        if template_data is None or template_horizon is None:
            return self._stable_horizon_tracking(base_pts, slice_data, height, width)
        
        # ============ STEP 1: Compute Seismic Attributes ============
        
        # Compute instantaneous attributes using Hilbert transform
        template_analytic = self._compute_analytic_signal(template_data)
        current_analytic = self._compute_analytic_signal(slice_data)
        
        # Instantaneous phase (most important for horizon tracking!)
        template_phase = np.angle(template_analytic)
        current_phase = np.angle(current_analytic)
        
        # Instantaneous envelope (amplitude envelope)
        template_envelope = np.abs(template_analytic)
        current_envelope = np.abs(current_analytic)
        
        # Compute local dip field for structural guidance
        current_dip = self._compute_local_dip(slice_data)
        
        # ============ STEP 2: Multi-attribute correlation ============
        
        base_dict = {int(x): int(y) for x, y in base_pts}
        template_dict = {int(x): int(y) for x, y in template_horizon}
        
        # Parameters
        template_half_height = 25  # Larger window for more context
        search_window = 50  # Reasonable search range
        multi_trace_width = 5  # Use 5 traces for semblance
        
        # First pass: estimate shifts with quality scores
        raw_shifts = []
        quality_scores = []
        x_positions = []
        
        for x in range(0, width, 3):  # Sample every 3rd trace for efficiency
            if x not in base_dict:
                continue
            
            base_y = base_dict[x]
            template_y = template_dict.get(x, base_y)
            
            # Get multi-trace windows
            x_min = max(0, x - multi_trace_width // 2)
            x_max = min(width, x + multi_trace_width // 2 + 1)
            
            t_ymin = max(0, template_y - template_half_height)
            t_ymax = min(height, template_y + template_half_height + 1)
            
            # Extract multi-trace template (amplitude + phase)
            template_amp = template_envelope[t_ymin:t_ymax, x_min:x_max]
            template_ph = template_phase[t_ymin:t_ymax, x_min:x_max]
            
            if template_amp.size < 10:
                continue
            
            # Search in current slice
            search_ymin = max(0, base_y - search_window)
            search_ymax = min(height, base_y + search_window + 1)
            
            best_shift = 0
            best_score = -1
            
            for dy in range(-search_window, search_window + 1, 1):
                test_y = base_y + dy
                test_ymin = max(0, test_y - template_half_height)
                test_ymax = min(height, test_y + template_half_height + 1)
                
                if test_ymax - test_ymin != t_ymax - t_ymin:
                    continue
                
                # Extract current window
                current_amp = current_envelope[test_ymin:test_ymax, x_min:x_max]
                current_ph = current_phase[test_ymin:test_ymax, x_min:x_max]
                
                if current_amp.shape != template_amp.shape:
                    continue
                
                # Compute similarity score combining amplitude and phase
                amp_score = self._normalized_correlation(template_amp.flatten(), current_amp.flatten())
                phase_score = self._phase_similarity(template_ph.flatten(), current_ph.flatten())
                
                # Combined score (phase is more important for horizons)
                combined_score = 0.4 * amp_score + 0.6 * phase_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_shift = dy
            
            if best_score > 0.2:  # Lowered threshold for more matches
                raw_shifts.append(best_shift)
                quality_scores.append(best_score)
                x_positions.append(x)
        
        # ============ STEP 3: Robust shift estimation ============
        
        if len(raw_shifts) < 3:
            # Not enough quality matches - use simple copy with no shift
            print(f"    Low quality matches ({len(raw_shifts)}), using minimal shift")
            return [(int(x), int(y)) for x, y in base_pts]
        
        # Weight shifts by quality
        raw_shifts = np.array(raw_shifts)
        quality_scores = np.array(quality_scores)
        x_positions = np.array(x_positions)
        
        # Use MEDIAN shift (most robust) instead of polynomial
        median_shift = np.median(raw_shifts)
        
        # Only use polynomial if we have many good matches
        if len(raw_shifts) >= 10 and np.std(raw_shifts) < 10:
            try:
                weights = quality_scores ** 2
                coeffs = np.polyfit(x_positions, raw_shifts, deg=1, w=weights)
                shift_trend = np.polyval(coeffs, np.arange(width))
            except:
                shift_trend = np.full(width, median_shift)
        else:
            # Use constant median shift
            shift_trend = np.full(width, median_shift)
        
        # STRICT limit on shift - prevent runaway
        max_shift = 15  # Very conservative
        shift_trend = np.clip(shift_trend, -max_shift, max_shift)
        
        # ============ STEP 4: Apply shifts with local phase snapping ============
        
        propagated_horizon = []
        prev_y = None
        base_avg_y = np.mean([y for x, y in base_pts])
        
        for x, y in base_pts:
            x, y = int(x), int(y)
            if x < 0 or x >= width:
                continue
            
            # Get the trend-based shift
            trend_shift = int(round(shift_trend[x]))
            expected_y = y + trend_shift
            
            # Bound check - don't let expected_y go out of valid range
            expected_y = int(np.clip(expected_y, 10, height - 10))
            
            # Local phase snapping - find nearest phase match in small window
            snap_window = 6  # Reduced from 8
            snap_ymin = max(0, expected_y - snap_window)
            snap_ymax = min(height, expected_y + snap_window + 1)
            
            template_y = template_dict.get(x, y)
            if 0 <= template_y < height and snap_ymax > snap_ymin:
                target_phase = template_phase[template_y, x]
                
                # Find position with most similar phase
                best_snap_y = expected_y
                best_phase_diff = float('inf')
                
                for snap_y in range(snap_ymin, snap_ymax):
                    if 0 <= snap_y < height:
                        phase_diff = abs(self._phase_difference(current_phase[snap_y, x], target_phase))
                        if phase_diff < best_phase_diff:
                            best_phase_diff = phase_diff
                            best_snap_y = snap_y
                
                new_y = best_snap_y
            else:
                new_y = expected_y
            
            # Smoothness constraint
            if prev_y is not None:
                max_local_jump = 4
                if abs(new_y - prev_y) > max_local_jump:
                    # Use dip to guide the jump direction
                    if 0 <= prev_y < height and x > 0:
                        local_dip = current_dip[prev_y, max(0, x-1)]
                        expected_change = local_dip  # Dip tells us expected vertical change per trace
                        new_y = int(prev_y + np.clip(expected_change, -max_local_jump, max_local_jump))
                    else:
                        new_y = int(prev_y + np.sign(new_y - prev_y) * max_local_jump)
            
            new_y = int(np.clip(new_y, 0, height - 1))
            propagated_horizon.append((x, new_y))
            prev_y = new_y
        
        # ============ STEP 5: Final smoothing ============
        
        if len(propagated_horizon) > 10:
            xs = [p[0] for p in propagated_horizon]
            ys = np.array([p[1] for p in propagated_horizon])
            
            # Median filter to remove outliers
            ys_median = median_filter(ys, size=5)
            # Gentle Gaussian smooth
            ys_smooth = gaussian_filter1d(ys_median.astype(float), sigma=2)
            
            propagated_horizon = [(int(x), int(np.clip(y, 0, height-1))) for x, y in zip(xs, ys_smooth)]
        
        return propagated_horizon
    
    def _compute_analytic_signal(self, data: np.ndarray) -> np.ndarray:
        """Compute analytic signal using Hilbert transform along vertical axis."""
        from scipy.signal import hilbert
        # Apply Hilbert transform column by column (vertical traces)
        analytic = np.zeros(data.shape, dtype=complex)
        for x in range(data.shape[1]):
            analytic[:, x] = hilbert(data[:, x].astype(float))
        return analytic
    
    def _compute_local_dip(self, data: np.ndarray) -> np.ndarray:
        """
        Compute local structural dip (vertical change per horizontal sample).
        Uses gradient-based estimation.
        """
        from scipy.ndimage import sobel, gaussian_filter
        
        # Smooth the data first
        smoothed = gaussian_filter(data.astype(float), sigma=2)
        
        # Compute gradients
        dz = sobel(smoothed, axis=0)  # Vertical gradient
        dx = sobel(smoothed, axis=1)  # Horizontal gradient
        
        # Dip = -dx/dz (negative because we want vertical change per horizontal step)
        dip = np.zeros_like(data, dtype=float)
        valid = np.abs(dz) > 1e-6
        dip[valid] = -dx[valid] / dz[valid]
        
        # Clip extreme dips
        dip = np.clip(dip, -5, 5)
        
        return dip
    
    def _normalized_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized cross-correlation coefficient."""
        a = a - np.mean(a)
        b = b - np.mean(b)
        
        std_a = np.std(a)
        std_b = np.std(b)
        
        if std_a < 1e-6 or std_b < 1e-6:
            return 0.0
        
        return np.sum(a * b) / (len(a) * std_a * std_b)
    
    def _phase_similarity(self, phase_a: np.ndarray, phase_b: np.ndarray) -> float:
        """Compute phase similarity (accounts for circular nature of phase)."""
        # Phase difference accounting for wrap-around
        diff = np.abs(phase_a - phase_b)
        diff = np.minimum(diff, 2*np.pi - diff)  # Handle wrap-around
        
        # Convert to similarity (0 = opposite phase, 1 = same phase)
        similarity = 1.0 - diff / np.pi
        return np.mean(similarity)
    
    def _phase_difference(self, phase_a: float, phase_b: float) -> float:
        """Compute minimum phase difference accounting for wrap-around."""
        diff = abs(phase_a - phase_b)
        return min(diff, 2*np.pi - diff)
    
    def _stable_horizon_tracking(self, base_pts: np.ndarray,
                                  slice_data: np.ndarray,
                                  height: int, width: int) -> List[Tuple[int, int]]:
        """
        Stable horizon tracking that prevents wild jumps.
        
        Key improvements:
        1. Uses MEDIAN shift estimation from multiple sample points
        2. Limits per-trace jumps strictly
        3. Global consistency check to prevent runaway drift
        """
        from scipy.ndimage import gaussian_filter1d, median_filter
        from scipy.signal import correlate
        
        # Get template data
        if not hasattr(self, '_tracking_templates') or self._tracking_templates is None:
            return self._edge_based_tracking(base_pts, slice_data, height, width)
        
        template_data = self._tracking_templates.get('slice_data')
        template_horizon = self._tracking_templates.get('horizon')
        
        if template_data is None or template_horizon is None:
            return self._edge_based_tracking(base_pts, slice_data, height, width)
        
        # Parameters - MUCH MORE CONSERVATIVE
        template_half_height = 20  # Larger template for more reliable matching
        search_window = 40  # Smaller search window to limit jumps
        
        # STEP 1: Estimate global vertical shift by sampling multiple locations
        sample_positions = [int(width * f) for f in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]]
        shifts = []
        
        base_dict = {int(x): int(y) for x, y in base_pts}
        template_dict = {int(x): int(y) for x, y in template_horizon}
        
        for x in sample_positions:
            if x not in base_dict or x >= width:
                continue
            
            base_y = base_dict[x]
            template_y = template_dict.get(x, base_y)
            
            # Extract template waveform
            t_ymin = max(0, template_y - template_half_height)
            t_ymax = min(template_data.shape[0], template_y + template_half_height + 1)
            template_waveform = template_data[t_ymin:t_ymax, x].astype(float)
            
            if len(template_waveform) < 10:
                continue
            
            # Normalize
            template_waveform = template_waveform - np.mean(template_waveform)
            template_std = np.std(template_waveform)
            if template_std < 1e-6:
                continue
            template_waveform = template_waveform / template_std
            
            # Search in new slice
            search_ymin = max(0, base_y - search_window)
            search_ymax = min(height, base_y + search_window + 1)
            search_trace = slice_data[search_ymin:search_ymax, x].astype(float)
            
            if len(search_trace) < len(template_waveform):
                continue
            
            # Normalize
            search_trace = search_trace - np.mean(search_trace)
            search_std = np.std(search_trace)
            if search_std < 1e-6:
                continue
            search_trace = search_trace / search_std
            
            # Cross-correlation
            correlation = correlate(search_trace, template_waveform, mode='valid')
            
            if len(correlation) > 0:
                best_offset = np.argmax(correlation)
                new_y = search_ymin + best_offset + len(template_waveform) // 2
                shift = new_y - base_y
                shifts.append(shift)
        
        # Use MEDIAN shift to be robust to outliers
        if shifts:
            global_shift = int(np.median(shifts))
        else:
            global_shift = 0
        
        # Limit global shift to prevent runaway
        max_global_shift = 20  # Maximum allowed shift per slice
        global_shift = np.clip(global_shift, -max_global_shift, max_global_shift)
        
        # STEP 2: Apply global shift and local refinement
        propagated_horizon = []
        prev_y = None
        
        for x, y in base_pts:
            x, y = int(x), int(y)
            if x < 0 or x >= width:
                continue
            
            # Start with globally shifted position
            expected_y = y + global_shift
            
            # Local refinement with TIGHT constraint
            template_y = template_dict.get(x, y)
            t_ymin = max(0, template_y - template_half_height)
            t_ymax = min(template_data.shape[0], template_y + template_half_height + 1)
            template_waveform = template_data[t_ymin:t_ymax, x].astype(float)
            
            local_search = 10  # Very tight local search
            search_ymin = max(0, expected_y - local_search)
            search_ymax = min(height, expected_y + local_search + 1)
            
            if len(template_waveform) >= 5 and search_ymax > search_ymin:
                search_trace = slice_data[search_ymin:search_ymax, x].astype(float)
                
                if len(search_trace) >= len(template_waveform):
                    # Normalize
                    tw = template_waveform - np.mean(template_waveform)
                    tw_std = np.std(tw)
                    st = search_trace - np.mean(search_trace)
                    st_std = np.std(st)
                    
                    if tw_std > 1e-6 and st_std > 1e-6:
                        tw = tw / tw_std
                        st = st / st_std
                        correlation = correlate(st, tw, mode='valid')
                        
                        if len(correlation) > 0:
                            best_offset = np.argmax(correlation)
                            new_y = search_ymin + best_offset + len(template_waveform) // 2
                        else:
                            new_y = expected_y
                    else:
                        new_y = expected_y
                else:
                    new_y = expected_y
            else:
                new_y = expected_y
            
            # STRICT smoothness constraint
            if prev_y is not None:
                max_local_jump = 3  # Very strict - max 3 pixels between adjacent traces
                if abs(new_y - prev_y) > max_local_jump:
                    new_y = int(prev_y + np.sign(new_y - prev_y) * max_local_jump)
            
            new_y = int(np.clip(new_y, 0, height - 1))
            propagated_horizon.append((x, new_y))
            prev_y = new_y
        
        # STEP 3: Smooth the result
        if len(propagated_horizon) > 10:
            xs = [p[0] for p in propagated_horizon]
            ys = np.array([p[1] for p in propagated_horizon])
            
            # Use median filter first to remove spikes
            ys_median = median_filter(ys, size=7)
            # Then Gaussian smooth
            ys_smooth = gaussian_filter1d(ys_median.astype(float), sigma=3)
            
            propagated_horizon = [(int(x), int(np.clip(y, 0, height-1))) for x, y in zip(xs, ys_smooth)]
        
        return propagated_horizon
    
    def _edge_based_tracking(self, base_pts: np.ndarray,
                             slice_data: np.ndarray,
                             height: int, width: int) -> List[Tuple[int, int]]:
        """
        Fallback edge-based tracking when waveform correlation is not available.
        """
        from scipy.ndimage import sobel, gaussian_filter1d
        
        # Compute edge strength
        edge_v = np.abs(sobel(slice_data.astype(float), axis=0))
        amplitude = np.abs(slice_data.astype(float))
        tracking_weight = 0.7 * edge_v / (edge_v.max() + 1e-10) + 0.3 * amplitude / (amplitude.max() + 1e-10)
        
        search_window = 50  # Larger search window
        propagated_horizon = []
        prev_y = None
        
        for x, y in base_pts:
            x, y = int(x), int(y)
            if x < 0 or x >= width:
                continue
            
            y_min = max(0, y - search_window)
            y_max = min(height, y + search_window + 1)
            
            if y_min >= y_max:
                new_y = y
            else:
                window = tracking_weight[y_min:y_max, x]
                distance_from_expected = np.abs(np.arange(len(window)) - (y - y_min))
                distance_weight = np.exp(-distance_from_expected / 15.0)
                combined_score = window * distance_weight
                best_offset = np.argmax(combined_score)
                new_y = y_min + best_offset
                
                if prev_y is not None:
                    max_jump = 10
                    if abs(new_y - prev_y) > max_jump:
                        new_y = int(prev_y + np.sign(new_y - prev_y) * max_jump)
            
            propagated_horizon.append((x, int(new_y)))
            prev_y = new_y
        
        # Smooth the result
        if len(propagated_horizon) > 5:
            xs = [p[0] for p in propagated_horizon]
            ys = np.array([p[1] for p in propagated_horizon])
            ys_smooth = gaussian_filter1d(ys.astype(float), sigma=2)
            propagated_horizon = [(int(x), int(np.clip(y, 0, height-1))) for x, y in zip(xs, ys_smooth)]
        
        return propagated_horizon
    
    def _seismic_guided_propagation(self, base_pts: np.ndarray, 
                                     slice_data: np.ndarray,
                                     slice_offset: int,
                                     height: int, width: int) -> List[Tuple[int, int]]:
        """
        Propagate horizon using seismic attributes for guidance.
        
        Uses edge detection to snap the horizon to the nearest strong reflector.
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Compute edge strength for the target slice
        try:
            attributes = self.attributes_processor.compute_slice_attributes(
                slice_data, ['edge_strength', 'amplitude']
            )
            edge_strength = attributes.get('edge_strength', np.zeros_like(slice_data))
        except Exception as e:
            print(f"Could not compute attributes: {e}")
            # Fallback to simple sobel edge detection
            from scipy.ndimage import sobel
            edge_strength = np.sqrt(sobel(slice_data.astype(float), axis=0)**2 + sobel(slice_data.astype(float), axis=1)**2)
        
        # Normalize edge strength
        edge_norm = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-10)
        
        raw_horizon = []
        search_window = 30  # Increased search window
        prev_y = None
        
        for x, y in base_pts:
            x = int(x)
            y = int(y)
            
            if x < 0 or x >= width:
                continue
            
            # Search for strongest edge near the expected position
            y_min = max(0, y - search_window)
            y_max = min(height, y + search_window + 1)
            
            if y_min >= y_max:
                raw_horizon.append((x, y))
                prev_y = y
                continue
            
            # Get edge strength in search window
            window = edge_norm[y_min:y_max, x]
            
            # Weight by distance from expected position
            distance_from_expected = np.abs(np.arange(len(window)) - (y - y_min))
            distance_weight = np.exp(-distance_from_expected / 15.0)
            
            combined_score = window * distance_weight
            
            # Find the position with best combined score
            best_offset = np.argmax(combined_score)
            new_y = y_min + best_offset
            
            # Apply smoothness constraint
            if prev_y is not None:
                max_jump = 8
                if abs(new_y - prev_y) > max_jump:
                    # Find best within constraint
                    valid_ys = range(max(y_min, prev_y - max_jump), min(y_max, prev_y + max_jump + 1))
                    if valid_ys:
                        best_in_range = max(valid_ys, key=lambda yy: combined_score[yy - y_min] if 0 <= yy - y_min < len(combined_score) else 0)
                        new_y = best_in_range
            
            raw_horizon.append((x, int(new_y)))
            prev_y = new_y
        
        # Apply smoothing
        if len(raw_horizon) > 5:
            xs = [p[0] for p in raw_horizon]
            ys = np.array([p[1] for p in raw_horizon])
            ys_smooth = gaussian_filter1d(ys.astype(float), sigma=2)
            return [(int(x), int(np.clip(y, 0, height-1))) for x, y in zip(xs, ys_smooth)]
        
        return raw_horizon

    # Autotracking methods
    def auto_detect_horizon_seeds(self, slice_data: np.ndarray, slice_idx: int,
                                 n_seeds: int = 10, method: str = 'from_points') -> List[Tuple[int, int]]:
        """Automatically detect seed points for horizon tracking."""
        if not self.autotracker:
            print("Autotracking not available")
            return []

        # Get existing foreground points for current object
        existing_points = None
        existing_masks = {}
        # We can check self.object_masks
        for mask_obj_id in self.object_masks.keys():
            slice_key = f"{self.current_slice_type}_{slice_idx}"
            if slice_key in self.object_masks[mask_obj_id]:
                existing_masks[mask_obj_id] = self.object_masks[mask_obj_id][slice_key]

        return self.autotracker.auto_detect_seeds(
            slice_data, slice_idx, n_seeds, method,
            existing_points=existing_points,
            existing_masks=existing_masks
        )

    def track_horizon_automatically(self, start_slice_idx: int, seed_points: List[Tuple[int, int]],
                                  direction: str = 'forward', max_slices: int = 50,
                                  mode: str = 'hybrid', horizon_id: str = None) -> Dict:
        """Track a horizon automatically."""
        if not self.autotracker:
            print("Autotracking not available")
            return {}

        # Convert string mode to enum
        mode_map = {
            'attribute_guided': TrackingMode.ATTRIBUTE_GUIDED,
            'edge_based': TrackingMode.EDGE_BASED,
            'phase_guided': TrackingMode.PHASE_GUIDED,
            'similarity_guided': TrackingMode.SIMILARITY_GUIDED,
            'hybrid': TrackingMode.HYBRID
        }
        tracking_mode = mode_map.get(mode, TrackingMode.HYBRID)

        return self.autotracker.track_horizon_dp(
            start_slice_idx=start_slice_idx,
            start_points=seed_points,
            direction=direction,
            max_slices=max_slices,
            mode=tracking_mode
        )

    def cleanup(self):
        """Clean up temporary files and resources"""
        print("Cleaning up SAM3 resources...")
        if hasattr(self, 'video_sessions'):
            for obj_id, session in self.video_sessions.items():
                if 'temp_dir' in session and os.path.exists(session['temp_dir']):
                    try:
                        shutil.rmtree(session['temp_dir'])
                        print(f"Removed temp dir: {session['temp_dir']}")
                    except Exception as e:
                        print(f"Error removing temp dir {session['temp_dir']}: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
