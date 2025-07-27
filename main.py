#!/usr/bin/env python3
"""
PhoenixDRS Professional - Command Line Interface
מערכת שחזור וידיאו מתקדמת עם ממשק שורת פקודה
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from logic.video_repair_engine import VideoRepairEngine
    from src.python.disk_imager import DiskImager
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

class PhoenixDRS_CLI:
    """PhoenixDRS Command Line Interface"""
    
    def __init__(self):
        self.repair_engine = None
        self.disk_imager = None
        
        # Initialize engines
        try:
            self.repair_engine = VideoRepairEngine()
        except:
            pass
            
        try:
            self.disk_imager = DiskImager()
        except:
            pass
    
    def print_banner(self):
        """הדפסת באנר התוכנה"""
        print("=" * 60)
        print("PhoenixDRS Professional - Command Line Interface")
        print("   Digital Recovery Suite v2.0.0")
        print("   Advanced Video Recovery & Repair System")
        print("=" * 60)
        print()
    
    def print_help(self):
        """הדפסת עזרה"""
        print("Available Commands:")
        print("  repair <input_file> [output_file]  - Repair damaged video file")
        print("  recover <source_path> <output_dir> - Recover deleted videos")
        print("  analyze <file_path>                - Analyze video file structure")
        print("  formats                            - List all supported video formats")
        print("  validate <file_path>               - Validate video file format")
        print("  list-drives                        - List available storage devices")
        print("  help                               - Show this help message")
        print("  exit                               - Exit the application")
        print()
    
    def repair_video(self, input_file, output_file=None):
        """תיקון קובץ וידיאו"""
        if not os.path.exists(input_file):
            print(f"[ERROR] Error: File '{input_file}' not found")
            return False
        
        if not output_file:
            name, ext = os.path.splitext(input_file)
            output_file = f"{name}_repaired{ext}"
        
        print(f"[REPAIR] Repairing video file: {input_file}")
        print(f"[OUTPUT] Output file: {output_file}")
        print()
        
        if self.repair_engine:
            try:
                # Setup progress callback
                def progress_callback(status):
                    progress = status.get('progress', 0)
                    status_text = status.get('status', 'Processing...')
                    print(f"\r[PROGRESS] {progress:.1f}% - {status_text}", end='', flush=True)
                
                def log_callback(message):
                    print(f"\n[LOG] {message}")
                
                self.repair_engine.set_callbacks(progress_callback, log_callback)
                
                # Perform repair
                success = self.repair_engine.repair_video(input_file, output_file, use_ai=True)
                
                print()  # New line after progress
                if success:
                    print(f"[SUCCESS] Video repair completed successfully!")
                    print(f"[FILE] Repaired file saved as: {output_file}")
                else:
                    print("[ERROR] Video repair failed")
                    
                return success
                
            except Exception as e:
                print(f"\n[ERROR] Error during repair: {str(e)}")
                return False
        else:
            print("[ERROR] Video repair engine not available")
            return False
    
    def recover_videos(self, source_path, output_dir):
        """שחזור קבצי וידיאו"""
        if not os.path.exists(source_path):
            print(f"[ERROR] Error: Source path '{source_path}' not found")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[SCAN] Scanning for deleted videos in: {source_path}")
        print(f"[FOLDER] Recovery output directory: {output_dir}")
        print()
        
        # Get comprehensive list of video extensions
        if self.repair_engine:
            video_extensions = self.repair_engine.get_supported_formats_list()
        else:
            # Fallback to basic list
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', 
                              '.mpg', '.mpeg', '.ts', '.m2ts', '.3gp', '.3g2', '.mxf', 
                              '.dv', '.hdv', '.asf', '.rm', '.rmvb', '.ogv', '.y4m']
        
        found_files = []
        
        try:
            if os.path.isfile(source_path):
                print("[FILE] Analyzing single file...")
                found_files = [source_path]
            else:
                print("[DIR] Scanning directory structure...")
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            found_files.append(os.path.join(root, file))
            
            print(f"[VIDEO] Found {len(found_files)} video files")
            
            if found_files:
                print("\nRecoverable videos:")
                for i, file_path in enumerate(found_files, 1):
                    size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"  {i:2d}. {os.path.basename(file_path)} ({size:.1f} MB)")
                
                print(f"\n[DRIVE] Files can be copied to recovery directory: {output_dir}")
                return True
            else:
                print("[ERROR] No video files found")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error during recovery scan: {str(e)}")
            return False
    
    def analyze_video(self, file_path):
        """ניתוח קובץ וידיאו מפורט"""
        if not os.path.exists(file_path):
            print(f"[ERROR] Error: File '{file_path}' not found")
            return False
        
        print(f"[SCAN] Analyzing video file: {file_path}")
        print()
        
        try:
            # Basic file info
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            print("[INFO] Basic File Information:")
            print(f"  Name: {file_name}")
            print(f"  Size: {file_size / (1024*1024):.2f} MB")
            print(f"  Extension: {file_ext}")
            print(f"  Path: {file_path}")
            print()
            
            # Enhanced format detection using repair engine
            if self.repair_engine:
                format_info = self.repair_engine.get_detailed_format_info(file_path)
                
                print("[INFO] Detailed Format Analysis:")
                print(f"  Format Name: {format_info.get('name', 'Unknown')}")
                print(f"  Container: {format_info.get('container', 'Unknown')}")
                print(f"  Description: {format_info.get('description', 'No description')}")
                print(f"  Max Resolution: {format_info.get('max_resolution', 'Unknown')}")
                print(f"  Professional Format: {'Yes' if format_info.get('is_professional', False) else 'No'}")
                print()
                
                print("[INFO] Codec Information:")
                common_codecs = format_info.get('common_codecs', [])
                audio_codecs = format_info.get('audio_codecs', [])
                detected_codec = format_info.get('detected_codec')
                
                print(f"  Supported Video Codecs: {', '.join(common_codecs)}")
                print(f"  Supported Audio Codecs: {', '.join(audio_codecs)}")
                if detected_codec:
                    print(f"  Detected Codec: {detected_codec}")
                print()
                
                print("[INFO] Validation Results:")
                print(f"  File Valid: {'Yes' if format_info.get('is_valid', False) else 'No'}")
                print(f"  Validation Message: {format_info.get('validation_message', 'No validation performed')}")
                print(f"  Repair Priority: {format_info.get('repair_priority', 0)}/10")
                print()
                
                # Support status
                is_supported = self.repair_engine.is_format_supported(file_path)
                print(f"[INFO] PhoenixDRS Support: {'Full Support' if is_supported else 'Limited Support'}")
                
            else:
                # Fallback to basic header analysis
                print("[SCAN] Basic Header Analysis:")
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    print(f"  Header bytes: {header.hex()}")
                    
                    # Basic format detection
                    if header[:4] == b'ftyp':
                        print("  Format: MP4/MOV container detected")
                    elif header[:4] == b'RIFF':
                        print("  Format: AVI container detected")
                    elif header[:4] == b'\x1a\x45\xdf\xa3':
                        print("  Format: MKV/WebM container detected")
                    elif header[:3] == b'FLV':
                        print("  Format: FLV container detected")
                    else:
                        print("  Format: Unknown or corrupted header")
            
            print()
            print("[SUCCESS] File analysis completed")
            return True
                    
        except Exception as e:
            print(f"[ERROR] Error during analysis: {str(e)}")
            return False
    
    def list_drives(self):
        """רשימת כונני האחסון"""
        print("[DRIVE] Available Storage Devices:")
        print()
        
        try:
            import psutil
            
            drives = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    total_gb = usage.total / (1024**3)
                    free_gb = usage.free / (1024**3)
                    used_gb = usage.used / (1024**3)
                    
                    drives.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': total_gb,
                        'used': used_gb,
                        'free': free_gb
                    })
                except:
                    drives.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': 0,
                        'used': 0,
                        'free': 0
                    })
            
            for i, drive in enumerate(drives, 1):
                print(f"  {i}. {drive['device']}")
                print(f"     Mount: {drive['mountpoint']}")
                print(f"     Type: {drive['fstype']}")
                if drive['total'] > 0:
                    print(f"     Size: {drive['total']:.1f} GB")
                    print(f"     Used: {drive['used']:.1f} GB")
                    print(f"     Free: {drive['free']:.1f} GB")
                print()
                
        except ImportError:
            print("[ERROR] psutil module not available. Install with: pip install psutil")
        except Exception as e:
            print(f"[ERROR] Error listing drives: {str(e)}")
    
    def list_supported_formats(self):
        """רשימת כל הפורמטים הנתמכים"""
        print("[INFO] PhoenixDRS Supported Video Formats:")
        print()
        
        if self.repair_engine:
            # Get format statistics
            try:
                if hasattr(self.repair_engine, 'format_manager') and self.repair_engine.format_manager:
                    stats = self.repair_engine.format_manager.get_format_statistics()
                    
                    print(f"Total Supported Formats: {stats['total_formats']}")
                    print(f"Professional Formats: {stats['professional_formats']}")
                    print(f"Consumer Formats: {stats['consumer_formats']}")
                    print(f"Unique Containers: {stats['unique_containers']}")
                    print(f"Unique Codecs: {stats['unique_codecs']}")
                    print()
                    
                    # List all supported extensions
                    supported_formats = self.repair_engine.get_supported_formats_list()
                    
                    print("Supported File Extensions:")
                    for i, ext in enumerate(supported_formats, 1):
                        print(f"  {i:2d}. {ext}")
                        if i % 10 == 0:  # Break every 10 items for readability
                            print()
                    
                    print()
                    print("Popular Container Types:")
                    for container in stats['containers'][:15]:  # Show first 15
                        print(f"  - {container}")
                    
                    print()
                    print("Supported Video Codecs:")
                    for codec in stats['codecs'][:20]:  # Show first 20
                        print(f"  - {codec}")
                
                else:
                    # Fallback list
                    basic_formats = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', 
                                   '.mpg', '.mpeg', '.ts', '.m2ts', '.3gp', '.3g2', '.mxf']
                    print("Basic Format Support:")
                    for i, ext in enumerate(basic_formats, 1):
                        print(f"  {i:2d}. {ext}")
                        
            except Exception as e:
                print(f"[ERROR] Error retrieving format information: {str(e)}")
        else:
            print("[ERROR] Video repair engine not available")
    
    def validate_video_file(self, file_path):
        """אימות קובץ וידיאו"""
        if not os.path.exists(file_path):
            print(f"[ERROR] Error: File '{file_path}' not found")
            return False
        
        print(f"[SCAN] Validating video file: {file_path}")
        print()
        
        try:
            if self.repair_engine:
                # Check if format is supported
                is_supported = self.repair_engine.is_format_supported(file_path)
                print(f"[INFO] Format Support: {'Supported' if is_supported else 'Not Supported'}")
                
                # Get detailed validation
                format_info = self.repair_engine.get_detailed_format_info(file_path)
                
                print("[INFO] Validation Results:")
                print(f"  File Format: {format_info.get('name', 'Unknown')}")
                print(f"  Container Type: {format_info.get('container', 'Unknown')}")
                print(f"  File Size: {format_info.get('file_size', 0) / (1024*1024):.2f} MB")
                print(f"  File Valid: {'Yes' if format_info.get('is_valid', False) else 'No'}")
                print(f"  Validation Status: {format_info.get('validation_message', 'No validation')}")
                print(f"  Repair Priority: {format_info.get('repair_priority', 0)}/10")
                print(f"  Professional Format: {'Yes' if format_info.get('is_professional', False) else 'No'}")
                
                if format_info.get('detected_codec'):
                    print(f"  Detected Codec: {format_info['detected_codec']}")
                
                print()
                
                # Recommendations
                if not format_info.get('is_valid', False):
                    print("[WARNING] File validation failed. Possible issues:")
                    print("  - File is corrupted or incomplete")
                    print("  - Invalid file signature/header")
                    print("  - Unsupported format variation")
                    print()
                    print("[RECOMMENDATION] Try using the repair function to fix issues")
                elif format_info.get('repair_priority', 0) >= 7:
                    print("[SUCCESS] High-quality format with excellent repair support")
                elif not is_supported:
                    print("[WARNING] Format not fully supported by PhoenixDRS")
                else:
                    print("[SUCCESS] File format is valid and supported")
                
                return format_info.get('is_valid', False)
                
            else:
                # Basic validation
                file_size = os.path.getsize(file_path)
                print(f"[INFO] File Size: {file_size / (1024*1024):.2f} MB")
                
                if file_size == 0:
                    print("[ERROR] File is empty")
                    return False
                elif file_size < 1024:
                    print("[WARNING] File is very small, might be corrupted")
                    return False
                else:
                    print("[SUCCESS] Basic validation passed")
                    return True
                    
        except Exception as e:
            print(f"[ERROR] Error during validation: {str(e)}")
            return False
    
    def run_interactive(self):
        """הרצת מצב אינטראקטיבי"""
        self.print_banner()
        print("Welcome to PhoenixDRS Interactive Mode")
        print("Type 'help' for available commands or 'exit' to quit")
        print()
        
        while True:
            try:
                command = input("PhoenixDRS> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'exit' or cmd == 'quit':
                    print("[BYE] Goodbye!")
                    break
                
                elif cmd == 'help':
                    self.print_help()
                
                elif cmd == 'repair':
                    if len(parts) < 2:
                        print("[ERROR] Usage: repair <input_file> [output_file]")
                    else:
                        input_file = parts[1]
                        output_file = parts[2] if len(parts) > 2 else None
                        self.repair_video(input_file, output_file)
                
                elif cmd == 'recover':
                    if len(parts) < 3:
                        print("[ERROR] Usage: recover <source_path> <output_dir>")
                    else:
                        source_path = parts[1]
                        output_dir = parts[2]
                        self.recover_videos(source_path, output_dir)
                
                elif cmd == 'analyze':
                    if len(parts) < 2:
                        print("[ERROR] Usage: analyze <file_path>")
                    else:
                        file_path = parts[1]
                        self.analyze_video(file_path)
                
                elif cmd == 'list-drives':
                    self.list_drives()
                
                elif cmd == 'formats':
                    self.list_supported_formats()
                
                elif cmd == 'validate':
                    if len(parts) < 2:
                        print("[ERROR] Usage: validate <file_path>")
                    else:
                        file_path = parts[1]
                        self.validate_video_file(file_path)
                
                else:
                    print(f"[ERROR] Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n[BYE] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {str(e)}")

def main():
    """פונקציה ראשית"""
    parser = argparse.ArgumentParser(description='PhoenixDRS Professional - Video Recovery & Repair')
    parser.add_argument('--repair', metavar=('INPUT', 'OUTPUT'), nargs='+', 
                       help='Repair video file')
    parser.add_argument('--recover', metavar=('SOURCE', 'OUTPUT'), nargs=2,
                       help='Recover deleted videos')
    parser.add_argument('--analyze', metavar='FILE',
                       help='Analyze video file')
    parser.add_argument('--validate', metavar='FILE',
                       help='Validate video file format')
    parser.add_argument('--formats', action='store_true',
                       help='List all supported video formats')
    parser.add_argument('--list-drives', action='store_true',
                       help='List available storage devices')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    cli = PhoenixDRS_CLI()
    
    # If no arguments, run interactive mode
    if len(sys.argv) == 1:
        cli.run_interactive()
        return
    
    cli.print_banner()
    
    if args.repair:
        input_file = args.repair[0]
        output_file = args.repair[1] if len(args.repair) > 1 else None
        cli.repair_video(input_file, output_file)
    
    elif args.recover:
        source_path, output_dir = args.recover
        cli.recover_videos(source_path, output_dir)
    
    elif args.analyze:
        cli.analyze_video(args.analyze)
    
    elif args.validate:
        cli.validate_video_file(args.validate)
    
    elif args.formats:
        cli.list_supported_formats()
    
    elif args.list_drives:
        cli.list_drives()
    
    elif args.interactive:
        cli.run_interactive()

if __name__ == "__main__":
    main()