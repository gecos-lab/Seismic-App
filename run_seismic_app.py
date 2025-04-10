#!/usr/bin/env python
"""
Seismic Interpretation App with SAM2
Run this script to launch the application

Available command line arguments:
  --demo            Run in demo mode without loading the SAM2 model (fallback option if no GPU available)
  --model MODEL     Choose the SAM2 model size to use. Options:
                    - base-plus : Default, good balance of speed and accuracy (80.8M parameters)
                    - large     : Most accurate but slower (224.4M parameters)
                    - small     : Faster with good accuracy (46M parameters) 
                    - tiny      : Fastest, but less accurate (38.9M parameters)
                    - 2.1-base-plus : Newer version with improved accuracy
                    - 2.1-large : Newest version, most accurate

Examples:
  python run_seismic_app.py                       # Run with default base-plus model 
  python run_seismic_app.py --model large         # Run with large model for best accuracy
  python run_seismic_app.py --model tiny          # Run with tiny model for better performance
  python run_seismic_app.py --demo                # Run in demo mode (no model loading)

Troubleshooting:
  If you encounter model loading errors:
  
  1. Make sure the SAM2 repository is properly installed:
     pip install -e .
  
  2. If you see "missing positional arguments" errors, try using demo mode:
     python run_seismic_app.py --demo
     
  3. If you see "Error initializing video predictor" messages, don't worry:
     - The application now automatically creates a mock video predictor
     - Mask propagation will automatically use demo mode
     - Image segmentation still works with the real SAM2 model
  
  4. If you have GPU issues, try a smaller model:
     python run_seismic_app.py --model tiny
  
  5. The application will automatically fall back to demo mode if model loading fails
  
  Note: In the current version, certain versions of the SAM2 model may have incompatible
  video predictor interfaces. The application will automatically detect this and switch
  to using the demo mode for video propagation while still using the real model for
  image segmentation.
"""

import sys
from app import main

if __name__ == "__main__":
    main() 