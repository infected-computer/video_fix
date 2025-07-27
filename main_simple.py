"""
PhoenixDRS - Simple Launcher (No Splash Screen)
מפעיל פשוט ללא מסך פתיחה לבדיקות
"""

import sys
import os
from PySide6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from gui.main_app import MainWindow
except ImportError as e:
    print(f"Fatal Error: Failed to import GUI modules: {e}")
    sys.exit(1)

def main():
    """Main function to run the application directly."""
    app = QApplication(sys.argv)
    app.setApplicationName("PhoenixDRS Professional")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("PhoenixDRS Team")
    
    # Create and show main window directly
    main_window = MainWindow()
    main_window.show()
    main_window.raise_()
    main_window.activateWindow()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()