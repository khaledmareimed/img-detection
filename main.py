"""
DIP Object Detector — Entry Point

Digital Image Processing Final Project
An-Najah National University — Spring 25-26

Launch the GUI application for robust object detection
using classical image processing techniques.
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import DIPApp


def main() -> None:
    """Create and run the DIP Object Detector application."""
    app = DIPApp()
    app.run()


if __name__ == "__main__":
    main()
