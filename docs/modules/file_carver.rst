File Carver Module
==================

The File Carver module is the core component responsible for recovering files from disk images using signature-based detection. It supports over 50 file types and provides high-performance parallel processing capabilities.

.. automodule:: file_carver
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The file carver works by:

1. **Signature Detection**: Scanning disk images for known file headers
2. **Footer Matching**: Locating file endings when available
3. **Size Validation**: Ensuring recovered files meet expected constraints
4. **Quality Assessment**: Determining completeness of recovered files

Core Classes
------------

FileSignature
~~~~~~~~~~~~~

.. autoclass:: file_carver.FileSignature
   :members:
   :undoc-members:

   Represents a file type signature with header/footer patterns and constraints.

   **Example Usage:**

   .. code-block:: python

      signature = FileSignature(
          name="JPEG Image",
          extension=".jpg", 
          header="FFD8FF",
          footer="FFD9",
          max_size=10485760,
          footer_search_offset=1024
      )

CarvedFile
~~~~~~~~~~

.. autoclass:: file_carver.CarvedFile
   :members:
   :undoc-members:

   Represents a successfully carved file with metadata.

SignatureDatabase
~~~~~~~~~~~~~~~~~

.. autoclass:: file_carver.SignatureDatabase
   :members:
   :undoc-members:

   Manages the database of file signatures loaded from JSON configuration.

   **Example Usage:**

   .. code-block:: python

      db = SignatureDatabase("signatures.json")
      headers = db.get_all_headers()
      print(f"Loaded {len(db.signatures)} signatures")

AhoCorasickSearcher
~~~~~~~~~~~~~~~~~~~

.. autoclass:: file_carver.AhoCorasickSearcher
   :members:
   :undoc-members:

   Efficient multi-pattern string searching algorithm implementation.

FileCarver
~~~~~~~~~~

.. autoclass:: file_carver.FileCarver
   :members:
   :undoc-members:

   Main file carving engine that orchestrates the recovery process.

   **Example Usage:**

   .. code-block:: python

      carver = FileCarver(chunk_size=2*1024*1024)  # 2MB chunks
      carved_files = carver.carve(
          "evidence.dd",
          "signatures.json", 
          "recovered_files/"
      )
      print(f"Recovered {len(carved_files)} files")

Supported File Types
--------------------

The file carver currently supports the following categories:

**Images**
- JPEG, PNG, GIF, BMP, TIFF, WebP
- RAW formats: Canon CR2, Nikon NEF, Sony ARW

**Documents** 
- PDF, Microsoft Office (DOC/DOCX, XLS/XLSX, PPT/PPTX)
- OpenDocument formats (ODT, ODS, ODP)
- Rich Text Format (RTF)

**Archives**
- ZIP, RAR, 7-Zip, GZIP, TAR
- Windows CAB, MSI installers

**Media Files**
- Video: MP4, MOV, AVI, MKV, FLV, WMV
- Audio: MP3, WAV, FLAC, AAC, OGG

**Databases**
- SQLite, MySQL, PostgreSQL, Microsoft Access

**System Files**
- Windows executables (EXE, DLL)
- Linux ELF binaries
- Registry hives, Event logs
- Prefetch files, LNK shortcuts

**Other**
- Email (MSG, EML, PST)
- Fonts (TTF, OTF, WOFF)
- Icons and graphics

Configuration
-------------

The file carver is configured through the ``signatures.json`` file:

.. code-block:: json

   {
     "signatures": [
       {
         "name": "JPEG Image",
         "extension": ".jpg",
         "header": "FFD8FF",
         "footer": "FFD9", 
         "max_size": 10485760,
         "footer_search_offset": 1024,
         "description": "JPEG image file"
       }
     ]
   }

**Configuration Parameters:**

- ``name``: Human-readable file type name
- ``extension``: File extension for recovered files
- ``header``: Hexadecimal header signature
- ``footer``: Optional footer signature
- ``max_size``: Maximum expected file size in bytes
- ``footer_search_offset``: Distance to search for footer
- ``description``: Optional description

Performance Tuning
------------------

**Chunk Size Configuration**

The chunk size affects memory usage and performance:

.. code-block:: python

   # For systems with limited RAM
   carver = FileCarver(chunk_size=1024*1024)  # 1MB

   # For high-performance systems  
   carver = FileCarver(chunk_size=10*1024*1024)  # 10MB

**Memory Considerations**

- Larger chunks = faster processing, more RAM usage
- Smaller chunks = slower processing, less RAM usage
- Default 1MB chunks work well for most systems

**Performance Benchmarks**

Typical performance on modern hardware:

.. list-table:: Performance Metrics
   :header-rows: 1

   * - Disk Size
     - Processing Time
     - Files Recovered
     - Throughput
   * - 100 GB
     - 45 minutes
     - ~5,000
     - 37 MB/s
   * - 500 GB  
     - 3.2 hours
     - ~25,000
     - 43 MB/s
   * - 1 TB
     - 6.1 hours  
     - ~50,000
     - 46 MB/s

Error Handling
--------------

The file carver includes comprehensive error handling:

**Common Errors**

- **FileNotFoundError**: Image file doesn't exist
- **PermissionError**: Insufficient file access permissions  
- **MemoryError**: Insufficient RAM for large images
- **JSONDecodeError**: Invalid signature database format

**Recovery Strategies**

- Automatic retry for transient I/O errors
- Graceful degradation for corrupted signatures
- Progress preservation during interruptions
- Detailed error logging for troubleshooting

Best Practices
--------------

**1. Pre-Processing**

.. code-block:: python

   # Verify image integrity before carving
   import hashlib
   
   def verify_image(image_path):
       with open(image_path, 'rb') as f:
           hash_md5 = hashlib.md5()
           for chunk in iter(lambda: f.read(4096), b""):
               hash_md5.update(chunk)
       return hash_md5.hexdigest()

**2. Custom Signatures**

Add specialized signatures for your use case:

.. code-block:: json

   {
     "name": "Custom Database",
     "extension": ".customdb",
     "header": "DEADBEEF", 
     "footer": "CAFEBABE",
     "max_size": 1073741824,
     "footer_search_offset": 8192,
     "description": "Custom application database"
   }

**3. Result Validation**

Always validate carved files:

.. code-block:: python

   for carved_file in carved_files:
       if carved_file.is_complete:
           print(f"✓ {carved_file.output_path}")
       else:
           print(f"⚠ {carved_file.output_path} (incomplete)")

**4. Progress Monitoring**

Monitor carving progress for large images:

.. code-block:: python

   import logging
   from logging_config import setup_logging
   
   # Enable detailed logging
   logger = setup_logging("CASE_001", "Examiner")
   
   # Carving will now include progress updates
   carved_files = carver.carve(image_path, sig_db, output_dir)

Integration Examples
-------------------

**With Disk Imager**

.. code-block:: python

   from disk_imager import DiskImager
   from file_carver import FileCarver
   
   # Create disk image
   imager = DiskImager()
   metadata = imager.create_image("/dev/sdb", "evidence.dd")
   
   # Carve files from image
   carver = FileCarver()
   carved_files = carver.carve("evidence.dd", "signatures.json", "output/")

**With Filesystem Parser**

.. code-block:: python

   from filesystem_parser import FilesystemDetector
   from file_carver import FileCarver
   
   # Analyze filesystem structure
   parser = FilesystemDetector.get_parser("evidence.dd")
   if parser:
       with parser as p:
           fs_info = p.parse_filesystem_info()
           print(f"Filesystem: {fs_info.filesystem_type}")
   
   # Carve files as backup recovery method
   carver = FileCarver()
   carved_files = carver.carve("evidence.dd", "signatures.json", "carved/")

Troubleshooting
--------------

**No Files Recovered**

Check:
- Image file accessibility and size
- Signature database validity
- Sufficient disk space for output
- File permissions

**Poor Performance**

Solutions:
- Increase chunk size for faster systems
- Use SSD storage for temporary files
- Close other memory-intensive applications
- Enable parallel processing if available

**Incomplete Files**

Causes:
- Fragmented file storage
- Overlapping file allocations  
- Corrupted file footers
- Incorrect signature definitions

See Also
--------

- :doc:`filesystem_parser` - For structural file recovery
- :doc:`validation_reporting` - For result verification
- :doc:`../advanced/custom_signatures` - Adding new file types
- :doc:`../advanced/performance_tuning` - Optimization techniques