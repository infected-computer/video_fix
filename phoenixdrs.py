#!/usr/bin/env python3
"""
PhoenixDRS Professional - Unified Entry Point
נקודת כניסה מאוחדת - PhoenixDRS מקצועי

Single entry point for both CLI and GUI modes.
"""

import sys
import os
import argparse
from pathlib import Path

# Add integration path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integration'))

def main():
    """Main entry point for PhoenixDRS"""
    parser = argparse.ArgumentParser(
        description='PhoenixDRS Professional - Digital Forensics Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI interface')
    parser.add_argument('--cli', action='store_true', 
                       help='Use command-line interface')
    parser.add_argument('--unified', action='store_true',
                       help='Use unified Python-C++ interface')
    parser.add_argument('--version', action='version', version='PhoenixDRS Professional 2.0.0')
    
    # Parse known args to allow forwarding to subcommands
    args, remaining = parser.parse_known_args()
    
    # Default to unified interface if no specific mode chosen
    if not (args.gui or args.cli):
        args.unified = True
    
    try:
        if args.gui:
            # Launch GUI
            gui_path = os.path.join(os.path.dirname(__file__), 'gui', 'gui_main.py')
            import subprocess
            result = subprocess.run([sys.executable, gui_path] + remaining)
            return result.returncode
            
        elif args.cli:
            # Launch CLI
            cli_path = os.path.join(os.path.dirname(__file__), 'cli', 'main.py')
            import subprocess
            result = subprocess.run([sys.executable, cli_path] + remaining)
            return result.returncode
            
        else:
            # Launch unified interface
            from phoenixdrs_unified import main as unified_main
            return unified_main(remaining)
            
    except Exception as e:
        print(f"Error launching PhoenixDRS: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())