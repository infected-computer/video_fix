PhoenixDRS Documentation
========================

.. image:: https://img.shields.io/badge/Version-1.0.0-blue.svg
   :alt: Version

.. image:: https://img.shields.io/badge/Python-3.8%2B-green.svg
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-red.svg
   :alt: License

**PhoenixDRS** is a professional-grade data recovery suite designed for forensic specialists and data recovery professionals. Built with modern Python technologies, it provides comprehensive tools for disk imaging, file carving, filesystem analysis, and data reconstruction.

Features
--------

🔍 **Advanced File Carving**
   Support for 50+ file types with signature-based recovery

🗂️ **Filesystem Analysis**
   Native support for NTFS, EXT4, APFS, and exFAT filesystems

💾 **Professional Disk Imaging**
   Sector-by-sector imaging with error handling and verification

🔧 **Video Reconstruction**
   Specialized algorithms for recovering fragmented video files

📊 **Comprehensive Reporting**
   Detailed forensic reports with chain of custody logging

🚀 **High Performance**
   Parallel processing and memory-mapped I/O for optimal speed

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/phoenixdrs/phoenixdrs.git
   cd phoenixdrs
   pip install -r requirements.txt

Basic Usage
~~~~~~~~~~~

**Create a disk image:**

.. code-block:: bash

   python phoenixdrs.py image create --source /dev/sdb --dest evidence.dd

**Carve files from an image:**

.. code-block:: bash

   python phoenixdrs.py carve --image evidence.dd --output carved_files

**Analyze filesystem:**

.. code-block:: bash

   python phoenixdrs.py analyze --image evidence.dd

**Validate recovered files:**

.. code-block:: bash

   python phoenixdrs.py validate --directory carved_files --output report.txt

Architecture Overview
---------------------

PhoenixDRS follows a modular architecture with the following core components:

.. mermaid::

   graph TB
       CLI[Command Line Interface] --> Core[Core Engine]
       Core --> Imaging[Imaging Module]
       Core --> Analysis[File System Analysis]
       Core --> Carver[File Carver Module]
       Core --> Reconstruction[Fragmented File Reconstruction]
       Core --> Validation[Result Validation & Reporting]
       
       Imaging --> Storage[(Storage Device)]
       Analysis --> FileSystem[File System Parsers]
       Carver --> Signatures[Signature Database]
       Reconstruction --> VideoEngine[Video Rebuilder]
       Validation --> Reports[Client Reports]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quick_start
   cli_reference
   workflows
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/file_carver
   modules/filesystem_parser
   modules/disk_imager
   modules/video_rebuilder
   modules/validation_reporting
   modules/logging_config

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/forensic_analysis
   advanced/performance_tuning
   advanced/custom_signatures
   advanced/plugin_development

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/testing
   development/architecture
   development/changelog

.. toctree::
   :maxdepth: 1
   :caption: Appendices

   appendix/file_signatures
   appendix/filesystem_support
   appendix/legal_considerations
   appendix/glossary

Hebrew Support
--------------

PhoenixDRS includes full Hebrew language support for the Israeli forensic community:

.. note::
   מערכת PhoenixDRS מספקת תמיכה מלאה בשפה העברית עבור קהילת הפורנזיקה הישראלית.
   התוכנה כוללת ממשק משתמש בעברית, דוחות בעברית, ותמיכה בקבצים עם שמות בעברית.

Use Cases
---------

Digital Forensics
~~~~~~~~~~~~~~~~~
- Criminal investigations
- Corporate security incidents
- Intellectual property theft
- Data breach analysis

Data Recovery
~~~~~~~~~~~~~
- Accidental file deletion
- Corrupted filesystems
- Hardware failures
- RAID reconstruction

Compliance & eDiscovery
~~~~~~~~~~~~~~~~~~~~~~~
- Legal hold processing
- Regulatory compliance
- Audit trail documentation
- Chain of custody maintenance

Performance Benchmarks
----------------------

PhoenixDRS has been tested on various hardware configurations:

.. list-table:: Performance Metrics
   :header-rows: 1
   :widths: 30 20 25 25

   * - Operation
     - 1TB HDD
     - 1TB SSD
     - 1TB NVMe
   * - Disk Imaging
     - 2.5 hours
     - 1.8 hours
     - 1.2 hours
   * - File Carving
     - 4.2 hours
     - 2.1 hours
     - 1.4 hours
   * - NTFS Analysis
     - 15 minutes
     - 8 minutes
     - 5 minutes

System Requirements
-------------------

**Minimum Requirements:**
- Python 3.8 or later
- 8 GB RAM
- 100 GB free disk space
- Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)

**Recommended Configuration:**
- Python 3.9 or later
- 32 GB RAM or more
- 1 TB SSD storage
- Multi-core CPU (8+ cores)

Support & Community
-------------------

- **Documentation**: https://phoenixdrs.readthedocs.io
- **Issues**: https://github.com/phoenixdrs/phoenixdrs/issues
- **Discussions**: https://github.com/phoenixdrs/phoenixdrs/discussions
- **Email**: support@phoenixdrs.com

License
-------

PhoenixDRS is released under the MIT License. See the `LICENSE <https://github.com/phoenixdrs/phoenixdrs/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`