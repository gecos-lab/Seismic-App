#!/usr/bin/env python3
"""
Basic Qt test to isolate platform issues
"""

import sys
import os

def test_qt_basic():
    """Test basic Qt functionality"""
    try:
        # Set Qt application attributes before creating QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        
        # Set attributes
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Create QApplication
        app = QApplication([])
        
        # Create a simple window
        window = QMainWindow()
        window.setWindowTitle("Qt Test")
        window.setGeometry(100, 100, 400, 300)
        
        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        label = QLabel("Qt is working!")
        layout.addWidget(label)
        
        # Show window
        window.show()
        
        print("✓ Basic Qt window created successfully")
        
        # Don't actually run the event loop, just test creation
        window.close()
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"✗ Qt basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matplotlib_qt():
    """Test matplotlib with Qt backend"""
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        
        print("✓ Matplotlib Qt backend working")
        return True
        
    except Exception as e:
        print(f"✗ Matplotlib Qt test failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("Testing basic Qt functionality...")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 2
    
    if test_qt_basic():
        tests_passed += 1
    
    if test_matplotlib_qt():
        tests_passed += 1
    
    print("-" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ Basic Qt functionality works!")
        return 0
    else:
        print("✗ Some basic tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
