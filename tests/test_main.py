"""
PhoenixDRS Main CLI Module Tests
בדיקות עבור מודול ממשק שורת הפקודה הראשי
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import argparse

import main
from main import (
    create_image_command, carve_command, rebuild_video_command,
    analyze_command, validate_command, main as main_func
)


class TestCreateImageCommand:
    """בדיקות עבור פקודת יצירת תמונת דיסק"""
    
    def test_create_image_success(self, temp_dir):
        """בדיקת יצירת תמונה מוצלחת"""
        # יצירת קובץ מקור מדומה
        source_file = temp_dir / "source_device"
        source_file.write_bytes(b"fake device data" * 1000)
        
        dest_file = temp_dir / "output.dd"
        
        # יצירת args מדומה
        args = Mock()
        args.source = str(source_file)
        args.dest = str(dest_file)
        args.sector_size = 512
        args.retries = 3
        args.retry_delay = 0.1
        
        # Mock של DiskImager
        mock_metadata = Mock()
        mock_metadata.total_sectors = 1000
        mock_metadata.bad_sectors = []
        mock_metadata.md5_hash = "fake_md5"
        mock_metadata.sha256_hash = "fake_sha256"
        
        with patch('main.DiskImager') as mock_imager_class:
            mock_imager = Mock()
            mock_imager.create_image.return_value = mock_metadata
            mock_imager_class.return_value = mock_imager
            
            with patch('builtins.print') as mock_print:
                result = create_image_command(args)
        
        # בדיקות
        assert result == 0
        mock_imager_class.assert_called_once_with(max_retries=3, retry_delay=0.1)
        mock_imager.create_image.assert_called_once_with(str(source_file), str(dest_file), 512)
        
        # בדיקת הדפסות
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("הדמיה הושלמה בהצלחה" in call for call in print_calls)
        assert any("1,000" in call for call in print_calls)
    
    def test_create_image_source_not_found(self, temp_dir):
        """בדיקת מקור לא נמצא"""
        non_existent_source = temp_dir / "non_existent"
        dest_file = temp_dir / "output.dd"
        
        args = Mock()
        args.source = str(non_existent_source)
        args.dest = str(dest_file)
        
        with patch('builtins.print') as mock_print:
            result = create_image_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: התקן המקור {non_existent_source} לא נמצא")
    
    def test_create_image_creates_dest_directory(self, temp_dir):
        """בדיקת יצירת תיקיית יעד"""
        source_file = temp_dir / "source"
        source_file.write_bytes(b"data")
        
        dest_dir = temp_dir / "new_dir" / "subdir"
        dest_file = dest_dir / "output.dd"
        
        args = Mock()
        args.source = str(source_file)
        args.dest = str(dest_file)
        args.sector_size = 512
        args.retries = 3
        args.retry_delay = 0.1
        
        mock_metadata = Mock()
        mock_metadata.total_sectors = 100
        mock_metadata.bad_sectors = []
        mock_metadata.md5_hash = "hash"
        mock_metadata.sha256_hash = "hash"
        
        with patch('main.DiskImager') as mock_imager_class:
            mock_imager = Mock()
            mock_imager.create_image.return_value = mock_metadata
            mock_imager_class.return_value = mock_imager
            
            result = create_image_command(args)
        
        assert result == 0
        assert dest_dir.exists()
    
    def test_create_image_exception_handling(self, temp_dir):
        """בדיקת טיפול בחריגות"""
        source_file = temp_dir / "source"
        source_file.write_bytes(b"data")
        dest_file = temp_dir / "output.dd"
        
        args = Mock()
        args.source = str(source_file)
        args.dest = str(dest_file)
        args.sector_size = 512
        args.retries = 3
        args.retry_delay = 0.1
        
        with patch('main.DiskImager') as mock_imager_class:
            mock_imager = Mock()
            mock_imager.create_image.side_effect = Exception("Test error")
            mock_imager_class.return_value = mock_imager
            
            with patch('builtins.print') as mock_print:
                result = create_image_command(args)
        
        assert result == 1
        mock_print.assert_called_with("שגיאה בהדמיה: Test error")


class TestCarveCommand:
    """בדיקות עבור פקודת חיתוך קבצים"""
    
    def test_carve_success(self, temp_dir):
        """בדיקת חיתוך מוצלח"""
        image_file = temp_dir / "disk.dd"
        db_file = temp_dir / "sigs.json"
        output_dir = temp_dir / "carved"
        
        # יצירת קבצים מדומים
        image_file.write_bytes(b"fake image data")
        db_file.write_text('{"signatures": []}')
        
        args = Mock()
        args.image = str(image_file)
        args.db = str(db_file)
        args.output = str(output_dir)
        args.chunk_size = 1024
        
        mock_carved_files = [Mock(), Mock(), Mock()]
        
        with patch('main.FileCarver') as mock_carver_class:
            mock_carver = Mock()
            mock_carver.carve.return_value = mock_carved_files
            mock_carver_class.return_value = mock_carver
            
            with patch('builtins.print') as mock_print:
                result = carve_command(args)
        
        assert result == 0
        mock_carver_class.assert_called_once_with(chunk_size=1024)
        mock_carver.carve.assert_called_once_with(str(image_file), str(db_file), str(output_dir))
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("חיתוך הושלם בהצלחה" in call for call in print_calls)
        assert any("3" in call for call in print_calls)
    
    def test_carve_image_not_found(self, temp_dir):
        """בדיקת תמונה לא נמצאת"""
        non_existent_image = temp_dir / "non_existent.dd"
        db_file = temp_dir / "sigs.json"
        output_dir = temp_dir / "carved"
        
        db_file.write_text('{"signatures": []}')
        
        args = Mock()
        args.image = str(non_existent_image)
        args.db = str(db_file)
        args.output = str(output_dir)
        
        with patch('builtins.print') as mock_print:
            result = carve_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: תמונת הדיסק {non_existent_image} לא נמצאת")
    
    def test_carve_db_not_found(self, temp_dir):
        """בדיקת DB לא נמצא"""
        image_file = temp_dir / "disk.dd"
        non_existent_db = temp_dir / "non_existent.json"
        output_dir = temp_dir / "carved"
        
        image_file.write_bytes(b"data")
        
        args = Mock()
        args.image = str(image_file)
        args.db = str(non_existent_db)
        args.output = str(output_dir)
        
        with patch('builtins.print') as mock_print:
            result = carve_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: מסד נתוני החתימות {non_existent_db} לא נמצא")


class TestRebuildVideoCommand:
    """בדיקות עבור פקודת שחזור וידאו"""
    
    def test_rebuild_video_canon_mov_success(self, temp_dir):
        """בדיקת שחזור Canon MOV מוצלח"""
        source_file = temp_dir / "source.dd"
        output_dir = temp_dir / "rebuilt"
        
        source_file.write_bytes(b"fake source data")
        
        args = Mock()
        args.source = str(source_file)
        args.output = str(output_dir)
        args.type = 'canon_mov'
        
        mock_rebuilt_videos = [Mock(), Mock()]
        
        with patch('main.VideoRebuilder') as mock_rebuilder_class:
            mock_rebuilder = Mock()
            mock_rebuilder.rebuild_canon_mov.return_value = mock_rebuilt_videos
            mock_rebuilder_class.return_value = mock_rebuilder
            
            with patch('builtins.print') as mock_print:
                result = rebuild_video_command(args)
        
        assert result == 0
        mock_rebuilder.rebuild_canon_mov.assert_called_once_with(str(source_file), str(output_dir))
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("שחזור וידאו הושלם בהצלחה" in call for call in print_calls)
        assert any("2" in call for call in print_calls)
    
    def test_rebuild_video_unsupported_type(self, temp_dir):
        """בדיקת סוג וידאו לא נתמך"""
        source_file = temp_dir / "source.dd"
        output_dir = temp_dir / "rebuilt"
        
        source_file.write_bytes(b"data")
        
        args = Mock()
        args.source = str(source_file)
        args.output = str(output_dir)
        args.type = 'unsupported_type'
        
        with patch('builtins.print') as mock_print:
            result = rebuild_video_command(args)
        
        assert result == 1
        mock_print.assert_called_with("שגיאה: סוג וידאו לא נתמך: unsupported_type")
    
    def test_rebuild_video_source_not_found(self, temp_dir):
        """בדיקת מקור לא נמצא"""
        non_existent_source = temp_dir / "non_existent.dd"
        output_dir = temp_dir / "rebuilt"
        
        args = Mock()
        args.source = str(non_existent_source)
        args.output = str(output_dir)
        args.type = 'canon_mov'
        
        with patch('builtins.print') as mock_print:
            result = rebuild_video_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: תמונת הדיסק {non_existent_source} לא נמצאת")


class TestAnalyzeCommand:
    """בדיקות עבור פקודת ניתוח"""
    
    def test_analyze_basic_info(self, temp_dir):
        """בדיקת ניתוח מידע בסיסי"""
        image_file = temp_dir / "test.dd"
        test_data = b"test data" * 1000
        image_file.write_bytes(test_data)
        
        args = Mock()
        args.image = str(image_file)
        
        with patch('builtins.print') as mock_print:
            result = analyze_command(args)
        
        assert result == 0
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any(f"גודל תמונה: {len(test_data):,} bytes" in call for call in print_calls)
    
    def test_analyze_with_metadata(self, temp_dir):
        """בדיקת ניתוח עם מטא-דטה"""
        image_file = temp_dir / "test.dd"
        metadata_file = temp_dir / "test.dd.metadata"
        
        image_file.write_bytes(b"data")
        metadata_file.write_text("# Metadata file\ntotal_sectors=1000\nmd5=abcd1234")
        
        args = Mock()
        args.image = str(image_file)
        
        with patch('builtins.print') as mock_print:
            result = analyze_command(args)
        
        assert result == 0
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("נמצא קובץ מטא-דטה" in call for call in print_calls)
        assert any("total_sectors=1000" in call for call in print_calls)
    
    def test_analyze_with_bad_sectors(self, temp_dir):
        """בדיקת ניתוח עם סקטורים פגומים"""
        image_file = temp_dir / "test.dd"
        bad_sectors_file = temp_dir / "test.dd.bad_sectors"
        
        image_file.write_bytes(b"data")
        bad_sectors_file.write_text("# Bad sectors\n1024\n2048\n4096\n")
        
        args = Mock()
        args.image = str(image_file)
        
        with patch('builtins.print') as mock_print:
            result = analyze_command(args)
        
        assert result == 0
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("סקטורים פגומים: 3" in call for call in print_calls)
    
    def test_analyze_image_not_found(self, temp_dir):
        """בדיקת תמונה לא נמצאת"""
        non_existent = temp_dir / "non_existent.dd"
        
        args = Mock()
        args.image = str(non_existent)
        
        with patch('builtins.print') as mock_print:
            result = analyze_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: תמונת הדיסק {non_existent} לא נמצאת")


class TestValidateCommand:
    """בדיקות עבור פקודת אימות"""
    
    def test_validate_success(self, temp_dir):
        """בדיקת אימות מוצלח"""
        directory = temp_dir / "carved_files"
        directory.mkdir()
        output_file = temp_dir / "report.txt"
        
        # יצירת קבצים מדומים
        (directory / "file1.jpg").write_bytes(b"fake jpeg")
        (directory / "file2.png").write_bytes(b"fake png")
        
        args = Mock()
        args.directory = str(directory)
        args.output = str(output_file)
        args.case_name = "Test Case"
        args.examiner = "Test Examiner"
        args.case_number = "12345"
        
        mock_results = Mock()
        mock_stats = Mock()
        mock_stats.valid_files = 2
        mock_stats.total_files_found = 2
        
        with patch('main.validate_and_report') as mock_validate:
            mock_validate.return_value = (mock_results, mock_stats)
            
            with patch('builtins.print') as mock_print:
                result = validate_command(args)
        
        assert result == 0
        
        expected_case_info = {
            'case_name': 'Test Case',
            'examiner': 'Test Examiner', 
            'case_number': '12345'
        }
        mock_validate.assert_called_once_with(str(directory), str(output_file), expected_case_info)
        
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("אימות הושלם בהצלחה" in call for call in print_calls)
        assert any("100.0%" in call for call in print_calls)
    
    def test_validate_directory_not_found(self, temp_dir):
        """בדיקת תיקייה לא נמצאת"""
        non_existent_dir = temp_dir / "non_existent"
        output_file = temp_dir / "report.txt"
        
        args = Mock()
        args.directory = str(non_existent_dir)
        args.output = str(output_file)
        args.case_name = None
        args.examiner = None
        args.case_number = None
        
        with patch('builtins.print') as mock_print:
            result = validate_command(args)
        
        assert result == 1
        mock_print.assert_called_with(f"שגיאה: התיקייה {non_existent_dir} לא נמצאת")
    
    def test_validate_no_case_info(self, temp_dir):
        """בדיקת אימות ללא מידע על המקרה"""
        directory = temp_dir / "carved_files"
        directory.mkdir()
        output_file = temp_dir / "report.txt"
        
        args = Mock()
        args.directory = str(directory)
        args.output = str(output_file)
        args.case_name = None
        args.examiner = None
        args.case_number = None
        
        mock_results = Mock()
        mock_stats = Mock()
        mock_stats.valid_files = 1
        mock_stats.total_files_found = 1
        
        with patch('main.validate_and_report') as mock_validate:
            mock_validate.return_value = (mock_results, mock_stats)
            
            result = validate_command(args)
        
        assert result == 0
        mock_validate.assert_called_once_with(str(directory), str(output_file), None)


class TestMainFunction:
    """בדיקות עבור הפונקציה הראשית"""
    
    def test_main_no_args(self):
        """בדיקת הרצה ללא ארגומנטים"""
        with patch.object(sys, 'argv', ['phoenixdrs']):
            with patch('main.argparse.ArgumentParser.print_help') as mock_help:
                result = main_func()
        
        assert result == 1
        mock_help.assert_called_once()
    
    def test_main_image_no_subcommand(self):
        """בדיקת image ללא תת-פקודה"""
        with patch.object(sys, 'argv', ['phoenixdrs', 'image']):
            with patch('main.argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.command = 'image'
                mock_args.image_command = None
                mock_parse.return_value = mock_args
                
                with patch('argparse.ArgumentParser.print_help') as mock_help:
                    result = main_func()
        
        assert result == 1
        mock_help.assert_called_once()
    
    def test_main_create_image_success(self):
        """בדיקת הרצה מוצלחת של create image"""
        test_args = [
            'phoenixdrs', 'image', 'create',
            '--source', '/dev/sdb',
            '--dest', 'output.dd'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.create_image_command') as mock_create:
                mock_create.return_value = 0
                
                result = main_func()
        
        assert result == 0
        mock_create.assert_called_once()
    
    def test_main_carve_success(self):
        """בדיקת הרצה מוצלחת של carve"""
        test_args = [
            'phoenixdrs', 'carve',
            '--image', 'disk.dd',
            '--output', 'carved_files'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.carve_command') as mock_carve:
                mock_carve.return_value = 0
                
                result = main_func()
        
        assert result == 0
        mock_carve.assert_called_once()
    
    def test_main_rebuild_video_success(self):
        """בדיקת הרצה מוצלחת של rebuild-video"""
        test_args = [
            'phoenixdrs', 'rebuild-video',
            '--type', 'canon_mov',
            '--source', 'disk.dd',
            '--output', 'rebuilt'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.rebuild_video_command') as mock_rebuild:
                mock_rebuild.return_value = 0
                
                result = main_func()
        
        assert result == 0
        mock_rebuild.assert_called_once()
    
    def test_main_analyze_success(self):
        """בדיקת הרצה מוצלחת של analyze"""
        test_args = ['phoenixdrs', 'analyze', '--image', 'disk.dd']
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.analyze_command') as mock_analyze:
                mock_analyze.return_value = 0
                
                result = main_func()
        
        assert result == 0
        mock_analyze.assert_called_once()
    
    def test_main_validate_success(self):
        """בדיקת הרצה מוצלחת של validate"""
        test_args = [
            'phoenixdrs', 'validate',
            '--directory', 'carved_files',
            '--output', 'report.txt'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.validate_command') as mock_validate:
                mock_validate.return_value = 0
                
                result = main_func()
        
        assert result == 0
        mock_validate.assert_called_once()


@pytest.mark.integration
class TestCLIIntegration:
    """בדיקות אינטגרציה עבור ממשק שורת הפקודה"""
    
    def test_full_workflow_simulation(self, temp_dir):
        """סימולציה של זרימת עבודה מלאה"""
        # הכנת קבצים
        source_device = temp_dir / "fake_device"
        source_device.write_bytes(b"fake device data" * 1000)
        
        disk_image = temp_dir / "disk.dd"
        signatures_db = temp_dir / "sigs.json"
        signatures_db.write_text('{"signatures": [{"name": "Test", "extension": ".test", "header": "1234", "footer": "", "max_size": 1000, "footer_search_offset": 0}]}')
        
        carved_dir = temp_dir / "carved"
        rebuilt_dir = temp_dir / "rebuilt"
        report_file = temp_dir / "report.txt"
        
        # Mock all the components
        with patch('main.DiskImager') as mock_imager_class, \
             patch('main.FileCarver') as mock_carver_class, \
             patch('main.VideoRebuilder') as mock_rebuilder_class, \
             patch('main.validate_and_report') as mock_validate:
            
            # Setup mocks
            mock_imager = Mock()
            mock_metadata = Mock()
            mock_metadata.total_sectors = 1000
            mock_metadata.bad_sectors = []
            mock_metadata.md5_hash = "hash"
            mock_metadata.sha256_hash = "hash"
            mock_imager.create_image.return_value = mock_metadata
            mock_imager_class.return_value = mock_imager
            
            mock_carver = Mock()
            mock_carver.carve.return_value = [Mock()]
            mock_carver_class.return_value = mock_carver
            
            mock_rebuilder = Mock()
            mock_rebuilder.rebuild_canon_mov.return_value = [Mock()]
            mock_rebuilder_class.return_value = mock_rebuilder
            
            mock_stats = Mock()
            mock_stats.valid_files = 1
            mock_stats.total_files_found = 1
            mock_validate.return_value = (Mock(), mock_stats)
            
            # Test each command
            # 1. Create image
            args_image = Mock()
            args_image.source = str(source_device)
            args_image.dest = str(disk_image)
            args_image.sector_size = 512
            args_image.retries = 3
            args_image.retry_delay = 0.1
            
            result = create_image_command(args_image)
            assert result == 0
            
            # 2. Carve files
            args_carve = Mock()
            args_carve.image = str(disk_image)
            args_carve.db = str(signatures_db)
            args_carve.output = str(carved_dir)
            args_carve.chunk_size = 1024
            
            result = carve_command(args_carve)
            assert result == 0
            
            # 3. Rebuild video
            args_rebuild = Mock()
            args_rebuild.source = str(disk_image)
            args_rebuild.output = str(rebuilt_dir)
            args_rebuild.type = 'canon_mov'
            
            result = rebuild_video_command(args_rebuild)
            assert result == 0
            
            # 4. Validate
            args_validate = Mock()
            args_validate.directory = str(carved_dir)
            args_validate.output = str(report_file)
            args_validate.case_name = "Integration Test"
            args_validate.examiner = "Test Suite"
            args_validate.case_number = "INT001"
            
            result = validate_command(args_validate)
            assert result == 0
            
            # Verify calls
            mock_imager.create_image.assert_called_once()
            mock_carver.carve.assert_called_once()
            mock_rebuilder.rebuild_canon_mov.assert_called_once()
            mock_validate.assert_called_once()