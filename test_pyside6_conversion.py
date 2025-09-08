#!/usr/bin/env python3
"""
Test script to verify the PySide6 conversion of the Seismic App
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt, Signal
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        print("✓ All PySide6 imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_app_creation():
    """Test that the app can be created without errors"""
    try:
        from app import SeismicApp
        from PySide6.QtWidgets import QApplication
        
        # Create QApplication
        app = QApplication([])
        
        # Create the main window in demo mode to avoid model loading
        window = SeismicApp(demo_mode=True, model_id="facebook/sam2-hiera-base-plus")
        
        print("✓ SeismicApp creation successful")
        
        # Clean up
        window.close()
        app.quit()
        return True
        
    except Exception as e:
        print(f"✗ App creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing PySide6 conversion...")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 2
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test app creation
    if test_app_creation():
        tests_passed += 1
    
    print("-" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! PySide6 conversion appears successful.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
