"""
GUI module for Echolab.

Uses PySide6 and pyqtgraph for interactive visualization.
Strict separation from DSP logic - this module only contains presentation.
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]

