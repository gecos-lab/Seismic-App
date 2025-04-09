# Seismic Interpretation with SAM2

A GUI application for seismic data visualization and interpretation using the Segment Anything Model 2 (SAM2).

## Features

- Load and visualize SEGY seismic data files
- View inline, crossline, and time/depth slices
- Annotate features with point prompts (foreground/background)
- Generate segmentation masks using SAM2
- Propagate segmentations across slices using SAM2's video capabilities
- Support for multiple object tracking and segmentation

## Requirements

- Python 3.8+
- PyTorch
- segyio
- matplotlib
- numpy
- scipy
- huggingface_hub (for loading SAM2 models)
- tkinter (for the GUI)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/seismic-sam2.git
cd seismic-sam2
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download SAM2 model (optional, app will run in demo mode if not available)
```
pip install huggingface_hub
```

## Usage

Run the application:

```
python run_seismic_app.py
```

### Basic Workflow

1. **Load a SEGY file**: Use File > Open SEGY to load your seismic data
2. **Select slice type**: Choose between inline, crossline, or time/depth slices
3. **Navigate**: Use the slider to navigate through the volume
4. **Annotate**: Click on the image to add annotation points
   - Red points (foreground): Areas you want to segment
   - Blue points (background): Areas you want to exclude
5. **Generate mask**: Click "Generate Mask" to create a segmentation mask
6. **Propagate**: Click "Propagate" to extend the segmentation to adjacent slices

### Demo Mode

If SAM2 modules are not available, the application will run in demo mode. This mode simulates SAM2 functionality for testing and demonstration purposes.

## Structure

- `app.py`: Main GUI application
- `segy_loader.py`: SEGY file handling and slice extraction
- `seismic_predictor.py`: SAM2 integration for seismic data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI Research for the SAM2
- The segyio team for their excellent SEGY file handling library 