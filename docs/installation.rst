Installation Guide
==================

This guide covers the installation of PhoenixDRS on different operating systems and various deployment scenarios.

System Requirements
-------------------

Before installing PhoenixDRS, ensure your system meets the following requirements:

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

- **Operating System**: Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or later
- **RAM**: 8 GB
- **Storage**: 100 GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

Recommended Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Operating System**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or later
- **RAM**: 32 GB or more
- **Storage**: 1 TB SSD
- **CPU**: High-performance multi-core (8+ cores)
- **Network**: High-speed connection for cloud features

Installation Methods
--------------------

Method 1: From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended method for development and advanced users:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/phoenixdrs/phoenixdrs.git
   cd phoenixdrs

   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\\Scripts\\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Verify installation
   python phoenixdrs.py --help

Method 2: Using pip (Future Release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   This method will be available when PhoenixDRS is published to PyPI.

.. code-block:: bash

   pip install phoenixdrs

Method 3: Binary Distribution (Future Release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-compiled binaries will be available for download from the GitHub releases page.

Platform-Specific Instructions
------------------------------

Windows Installation
~~~~~~~~~~~~~~~~~~~~

1. **Install Python**:
   
   Download Python 3.9+ from `python.org <https://www.python.org/downloads/>`_ and install with "Add to PATH" option.

2. **Install Git**:
   
   Download from `git-scm.com <https://git-scm.com/download/win>`_ if not already installed.

3. **Install Visual Studio Build Tools** (for pytsk3):
   
   Download and install Microsoft C++ Build Tools or Visual Studio Community.

4. **Clone and Install**:

   .. code-block:: cmd

      git clone https://github.com/phoenixdrs/phoenixdrs.git
      cd phoenixdrs
      python -m venv venv
      venv\\Scripts\\activate
      pip install -r requirements.txt

5. **Optional: Install The Sleuth Kit**:
   
   For advanced filesystem analysis, install TSK from `sleuthkit.org <https://www.sleuthkit.org/sleuthkit/>`_.

macOS Installation
~~~~~~~~~~~~~~~~~

1. **Install Homebrew** (if not already installed):

   .. code-block:: bash

      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. **Install dependencies**:

   .. code-block:: bash

      brew install python git sleuthkit

3. **Clone and Install**:

   .. code-block:: bash

      git clone https://github.com/phoenixdrs/phoenixdrs.git
      cd phoenixdrs
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt

Linux Installation (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Update system and install dependencies**:

   .. code-block:: bash

      sudo apt update
      sudo apt install python3 python3-pip python3-venv git build-essential libtsk-dev

2. **Clone and Install**:

   .. code-block:: bash

      git clone https://github.com/phoenixdrs/phoenixdrs.git
      cd phoenixdrs
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt

Linux Installation (RHEL/CentOS/Fedora)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Install dependencies**:

   .. code-block:: bash

      # For RHEL/CentOS:
      sudo yum install python3 python3-pip git gcc sleuthkit-devel

      # For Fedora:
      sudo dnf install python3 python3-pip git gcc sleuthkit-devel

2. **Clone and Install**:

   .. code-block:: bash

      git clone https://github.com/phoenixdrs/phoenixdrs.git
      cd phoenixdrs
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt

Docker Installation
-------------------

For containerized deployment:

.. code-block:: bash

   # Build the Docker image
   docker build -t phoenixdrs .

   # Run PhoenixDRS in a container
   docker run -it --privileged -v /path/to/evidence:/evidence phoenixdrs

Development Installation
------------------------

For developers who want to contribute to PhoenixDRS:

.. code-block:: bash

   # Clone with development dependencies
   git clone https://github.com/phoenixdrs/phoenixdrs.git
   cd phoenixdrs
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows

   # Install development dependencies
   pip install -r requirements.txt
   pip install -e .

   # Install pre-commit hooks
   pre-commit install

   # Run tests to verify installation
   pytest tests/

Optional Dependencies
--------------------

Enhanced Features
~~~~~~~~~~~~~~~~

For additional functionality, you can install optional dependencies:

.. code-block:: bash

   # Machine learning features
   pip install scikit-learn numpy

   # GUI interface
   pip install PySide6

   # Web API
   pip install flask

   # Advanced image processing
   pip install opencv-python

   # Distributed processing
   pip install celery

Forensic Tools Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

PhoenixDRS can integrate with other forensic tools:

- **The Sleuth Kit**: For advanced filesystem analysis
- **Volatility**: For memory analysis
- **Autopsy**: As a plugin or standalone tool

Configuration
-------------

After installation, you may want to configure PhoenixDRS:

1. **Create configuration directory**:

   .. code-block:: bash

      mkdir ~/.phoenixdrs

2. **Copy default configuration**:

   .. code-block:: bash

      cp config/default_config.json ~/.phoenixdrs/config.json

3. **Edit configuration** to match your environment.

Verification
-----------

To verify your installation:

.. code-block:: bash

   # Test basic functionality
   python phoenixdrs.py --version
   python phoenixdrs.py --help

   # Run system check
   python -c "import file_carver, filesystem_parser, disk_imager; print('All modules imported successfully')"

   # Run basic tests
   pytest tests/test_basic.py -v

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'pytsk3'**

Solution:
  - On Windows: Install Visual Studio Build Tools
  - On macOS: ``brew install sleuthkit``
  - On Linux: ``sudo apt install libtsk-dev`` (Ubuntu) or equivalent

**Permission denied when accessing devices**

Solution:
  Run PhoenixDRS with appropriate privileges or configure udev rules on Linux.

**Memory errors with large disk images**

Solution:
  Increase system RAM or adjust chunk sizes in configuration.

**Slow performance**

Solution:
  - Use SSD storage for temporary files
  - Increase available RAM
  - Adjust thread count in configuration

Getting Help
~~~~~~~~~~~~

If you encounter issues during installation:

1. Check the `troubleshooting guide <troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/phoenixdrs/phoenixdrs/issues>`_
3. Create a new issue with system details and error messages
4. Join our community discussions

Updating PhoenixDRS
------------------

To update to the latest version:

.. code-block:: bash

   # If installed from source
   cd phoenixdrs
   git pull origin main
   pip install -r requirements.txt

   # If installed via pip (future)
   pip install --upgrade phoenixdrs

Uninstallation
--------------

To remove PhoenixDRS:

.. code-block:: bash

   # If using virtual environment
   rm -rf venv

   # If installed globally
   pip uninstall phoenixdrs

   # Remove configuration (optional)
   rm -rf ~/.phoenixdrs

Next Steps
----------

After successful installation:

1. Read the `Quick Start Guide <quick_start.html>`_
2. Explore the `CLI Reference <cli_reference.html>`_
3. Review `Best Practices <workflows.html>`_
4. Check out `Advanced Topics <advanced/forensic_analysis.html>`_