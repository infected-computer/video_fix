#!/bin/bash

# PhoenixDRS Professional Launcher Script
# Compatible with Linux and macOS

echo "========================================================"
echo "  PhoenixDRS Professional v2.0.0"
echo "  Advanced Digital Recovery and Forensics Suite"
echo "========================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        echo "Please install Python 3.8 or later and try again."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $REQUIRED_VERSION or later is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if virtual environment should be used
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
$PYTHON_CMD -c "import PySide6; print('✓ PySide6 GUI framework')" 2>/dev/null || {
    echo "Installing required packages..."
    $PYTHON_CMD -m pip install -r requirements.txt
}

# Launch the application
echo
echo "Starting PhoenixDRS Professional..."
echo
$PYTHON_CMD main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "Application encountered an error. Check the logs above."
    read -p "Press Enter to continue..."
fi