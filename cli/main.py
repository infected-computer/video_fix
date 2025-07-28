#!/usr/bin/env python3
"""
PhoenixDRS - Phoenix Data Recovery Suite
ממשק שורת פקודה ראשי למערכת שחזור מידע מקצועית
"""

import argparse
import sys
import os
from pathlib import Path

# ייבוא המודולים שלנו
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from disk_imager import DiskImager
from file_carver import FileCarver
from video_rebuilder import VideoRebuilder
from validation_reporting import validate_and_report
from video_repair_engine import VideoRepairEngine


def create_image_command(args):
    """פקודת יצירת תמונת דיסק"""
    print(f"יוצר תמונת דיסק: {args.source} -> {args.dest}")
    
    # בדיקת קיום התקן המקור
    if not os.path.exists(args.source):
        print(f"שגיאה: התקן המקור {args.source} לא נמצא")
        return 1
    
    # בדיקת תיקיית היעד
    dest_dir = os.path.dirname(args.dest)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    try:
        imager = DiskImager(max_retries=args.retries, retry_delay=args.retry_delay)
        metadata = imager.create_image(args.source, args.dest, args.sector_size)
        
        print(f"\nהדמיה הושלמה בהצלחה!")
        print(f"סה\"כ סקטורים: {metadata.total_sectors:,}")
        print(f"סקטורים פגומים: {len(metadata.bad_sectors)}")
        print(f"MD5: {metadata.md5_hash}")
        print(f"SHA256: {metadata.sha256_hash}")
        
        return 0
        
    except Exception as e:
        print(f"שגיאה בהדמיה: {e}")
        return 1


def carve_command(args):
    """פקודת חיתוך קבצים"""
    print(f"מתחיל חיתוך קבצים: {args.image}")
    
    # בדיקת קיום קבצים
    if not os.path.exists(args.image):
        print(f"שגיאה: תמונת הדיסק {args.image} לא נמצאת")
        return 1
    
    if not os.path.exists(args.db):
        print(f"שגיאה: מסד נתוני החתימות {args.db} לא נמצא")
        return 1
    
    try:
        carver = FileCarver(
            chunk_size=args.chunk_size,
            enable_parallel=getattr(args, 'parallel', True),
            max_workers=getattr(args, 'workers', None)
        )
        
        # בחירת מתודת חיתוך
        if getattr(args, 'parallel', True):
            carved_files = carver.carve_parallel(args.image, args.db, args.output)
        else:
            carved_files = carver.carve(args.image, args.db, args.output)
        
        print(f"\nחיתוך הושלם בהצלחה!")
        print(f"נחתכו {len(carved_files)} קבצים")
        print(f"תוצאות נשמרו ב-{args.output}")
        
        return 0
        
    except Exception as e:
        print(f"שגיאה בחיתוך: {e}")
        return 1


def rebuild_video_command(args):
    """פקודת שחזור וידאו"""
    print(f"מתחיל שחזור וידאו: {args.source}")
    
    # בדיקת קיום תמונת הדיסק
    if not os.path.exists(args.source):
        print(f"שגיאה: תמונת הדיסק {args.source} לא נמצאת")
        return 1
    
    try:
        rebuilder = VideoRebuilder()
        
        if args.type == 'canon_mov':
            rebuilt_videos = rebuilder.rebuild_canon_mov(args.source, args.output)
        else:
            print(f"שגיאה: סוג וידאו לא נתמך: {args.type}")
            return 1
        
        print(f"\nשחזור וידאו הושלם בהצלחה!")
        print(f"נבנו {len(rebuilt_videos)} קבצי וידאו")
        print(f"תוצאות נשמרו ב-{args.output}")
        
        return 0
        
    except Exception as e:
        print(f"שגיאה בשחזור וידאו: {e}")
        return 1


def repair_video_command(args):
    """פקודת תיקון וידאו"""
    print(f"מתחיל תיקון וידאו: {args.input}")

    if not os.path.exists(args.input):
        print(f"שגיאה: הקובץ {args.input} לא נמצא")
        return 1

    engine = VideoRepairEngine()

    output = args.output
    if not output:
        name, ext = os.path.splitext(args.input)
        output = f"{name}_repaired{ext}"

    try:
        result = engine.repair_video(args.input, output, use_ai=args.use_ai)
        if result.success:
            print("\nתיקון הושלם בהצלחה!")
            print(f"הקובץ המתוקן נשמר ב-{output}")
            return 0
        print("\nתיקון נכשל")
        for err in result.errors_found:
            print(f"- {err}")
        return 1
    except Exception as e:
        print(f"שגיאה בתיקון וידאו: {e}")
        return 1


def analyze_command(args):
    """פקודת ניתוח תמונת דיסק"""
    print(f"מנתח תמונת דיסק: {args.image}")
    
    if not os.path.exists(args.image):
        print(f"שגיאה: תמונת הדיסק {args.image} לא נמצאת")
        return 1
    
    try:
        # ניתוח בסיסי של התמונה
        file_size = os.path.getsize(args.image)
        print(f"גודל תמונה: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
        
        # בדיקת קיום קבצי מטא-דטה
        metadata_file = f"{args.image}.metadata"
        if os.path.exists(metadata_file):
            print(f"נמצא קובץ מטא-דטה: {metadata_file}")
            with open(metadata_file, 'r') as f:
                print("מטא-דטה:")
                for line in f:
                    if not line.startswith('#'):
                        print(f"  {line.strip()}")
        
        bad_sectors_file = f"{args.image}.bad_sectors"
        if os.path.exists(bad_sectors_file):
            with open(bad_sectors_file, 'r') as f:
                bad_sectors = sum(1 for line in f if not line.startswith('#'))
                print(f"סקטורים פגומים: {bad_sectors}")
        
        return 0
        
    except Exception as e:
        print(f"שגיאה בניתוח: {e}")
        return 1


def validate_command(args):
    """פקודת אימות קבצים שוחזרו"""
    print(f"מאמת קבצים ב-{args.directory}")
    
    if not os.path.exists(args.directory):
        print(f"שגיאה: התיקייה {args.directory} לא נמצאת")
        return 1
    
    try:
        # מידע על המקרה (אופציונלי)
        case_info = {}
        if args.case_name:
            case_info['case_name'] = args.case_name
        if args.examiner:
            case_info['examiner'] = args.examiner
        if args.case_number:
            case_info['case_number'] = args.case_number
        
        # הרצת אימות ויצירת דוח
        validation_results, statistics = validate_and_report(
            args.directory, args.output, case_info if case_info else None
        )
        
        print(f"\nאימות הושלם בהצלחה!")
        print(f"קבצים תקינים: {statistics.valid_files}/{statistics.total_files_found}")
        print(f"אחוז הצלחה: {(statistics.valid_files/statistics.total_files_found*100):.1f}%")
        print(f"דוח נשמר ב-{args.output}")
        
        return 0
        
    except Exception as e:
        print(f"שגיאה באימות: {e}")
        return 1


def gui_command(args):
    """פקודת הפעלת ממשק גרפי"""
    print("מפעיל ממשק גרפי של PhoenixDRS...")
    print("Starting PhoenixDRS GUI...")
    
    try:
        from gui_main import main as gui_main
        return gui_main()
    except ImportError:
        print("שגיאה: PySide6 לא מותקן")
        print("Error: PySide6 is not installed")
        print("אנא התקן עם: pip install PySide6")
        print("Please install with: pip install PySide6")
        return 1
    except Exception as e:
        print(f"שגיאה בפתיחת ממשק גרפי: {e}")
        print(f"Error launching GUI: {e}")
        return 1


def main():
    """פונקציה ראשית"""
    parser = argparse.ArgumentParser(
        description='PhoenixDRS - מערכת שחזור מידע מקצועית',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות שימוש:
  phoenixdrs gui                                                                    # הפעלת ממשק גרפי
  phoenixdrs image create --source /dev/sdb --dest disk_image.dd
  phoenixdrs carve --image disk_image.dd --db signatures.json --output carved_files
  phoenixdrs rebuild-video --type canon_mov --source disk_image.dd --output rebuilt_videos
  phoenixdrs analyze --image disk_image.dd
  phoenixdrs validate --directory carved_files --output recovery_report.txt --case-name "Case001"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='פקודות זמינות')
    
    # פקודת image
    image_parser = subparsers.add_parser('image', help='פעולות הדמיה')
    image_subparsers = image_parser.add_subparsers(dest='image_command')
    
    # image create
    create_parser = image_subparsers.add_parser('create', help='יצירת תמונת דיסק')
    create_parser.add_argument('--source', required=True, help='התקן המקור')
    create_parser.add_argument('--dest', required=True, help='קובץ היעד')
    create_parser.add_argument('--sector-size', type=int, default=512, help='גודל סקטור (ברירת מחדל: 512)')
    create_parser.add_argument('--retries', type=int, default=3, help='מספר ניסיונות חוזרים (ברירת מחדל: 3)')
    create_parser.add_argument('--retry-delay', type=float, default=0.1, help='השהיה בין ניסיונות (ברירת מחדל: 0.1)')
    create_parser.set_defaults(func=create_image_command)
    
    # פקודת carve
    carve_parser = subparsers.add_parser('carve', help='חיתוך קבצים')
    carve_parser.add_argument('--image', required=True, help='תמונת דיסק')
    carve_parser.add_argument('--db', default='signatures.json', help='מסד נתוני חתימות')
    carve_parser.add_argument('--output', required=True, help='תיקיית פלט')
    carve_parser.add_argument('--chunk-size', type=int, default=1024*1024, help='גודל chunk לעיבוד (ברירת מחדל: 1MB)')
    carve_parser.add_argument('--parallel', action='store_true', default=True, help='עיבוד מקבילי (ברירת מחדל: מופעל)')
    carve_parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='ביטול עיבוד מקבילי')
    carve_parser.add_argument('--workers', type=int, help='מספר workers לעיבוד מקבילי (ברירת מחדל: אוטומטי)')
    carve_parser.set_defaults(func=carve_command)
    
    # פקודת rebuild-video
    video_parser = subparsers.add_parser('rebuild-video', help='שחזור וידאו')
    video_parser.add_argument('--type', choices=['canon_mov'], required=True, help='סוג וידאו')
    video_parser.add_argument('--source', required=True, help='תמונת דיסק')
    video_parser.add_argument('--output', required=True, help='תיקיית פלט')
    video_parser.set_defaults(func=rebuild_video_command)

    # פקודת repair-video
    repair_parser = subparsers.add_parser('repair-video', help='תיקון קובץ וידאו')
    repair_parser.add_argument('--input', required=True, help='קובץ וידאו פגום')
    repair_parser.add_argument('--output', help='קובץ פלט מתוקן')
    repair_parser.add_argument('--no-ai', dest='use_ai', action='store_false', help='ביטול שימוש ב-AI')
    repair_parser.set_defaults(func=repair_video_command, use_ai=True)
    
    # פקודת analyze
    analyze_parser = subparsers.add_parser('analyze', help='ניתוח תמונת דיסק')
    analyze_parser.add_argument('--image', required=True, help='תמונת דיסק')
    analyze_parser.set_defaults(func=analyze_command)
    
    # פקודת validate
    validate_parser = subparsers.add_parser('validate', help='אימות קבצים שוחזרו')
    validate_parser.add_argument('--directory', required=True, help='תיקיית קבצים לאימות')
    validate_parser.add_argument('--output', required=True, help='קובץ דוח פלט')
    validate_parser.add_argument('--case-name', help='שם המקרה')
    validate_parser.add_argument('--examiner', help='שם הבודק')
    validate_parser.add_argument('--case-number', help='מספר מקרה')
    validate_parser.set_defaults(func=validate_command)
    
    # פקודת GUI
    gui_parser = subparsers.add_parser('gui', help='הפעלת ממשק גרפי')
    gui_parser.set_defaults(func=gui_command)
    
    # ניתוח ארגומנטים
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # הרצת הפקודה המתאימה
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        if args.command == 'image' and not args.image_command:
            image_parser.print_help()
        else:
            parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())