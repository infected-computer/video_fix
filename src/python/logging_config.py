"""
PhoenixDRS - Advanced Logging Configuration
תצורת רישום מתקדמת עבור PhoenixDRS

Professional-grade logging system for forensic data recovery operations.
Provides structured logging, audit trails, and performance monitoring.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    LOGURU_AVAILABLE = False
    print("Warning: loguru not available, falling back to standard logging")


class ForensicLogger:
    """
    מתקדם logger מיוחד לפעולות שחזור מידע פורנזיות
    
    Features:
    - Structured logging with JSON output
    - Chain of custody logging
    - Performance monitoring
    - Security event logging
    - Multi-level log management
    """
    
    def __init__(self, case_name: Optional[str] = None, examiner: Optional[str] = None):
        self.case_name = case_name or "UNKNOWN_CASE"
        self.examiner = examiner or "UNKNOWN_EXAMINER"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        if LOGURU_AVAILABLE:
            self._setup_loguru()
        else:
            self._setup_standard_logging()
    
    def _setup_loguru(self):
        """הגדרת loguru מתקדמת"""
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # Main log file - detailed logging
        main_log_file = self.logs_dir / f"phoenixdrs_{self.session_id}.log"
        logger.add(
            str(main_log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # Structured JSON log for analysis
        json_log_file = self.logs_dir / f"phoenixdrs_structured_{self.session_id}.jsonl"
        logger.add(
            str(json_log_file),
            format=self._json_formatter,
            level="INFO",
            rotation="50 MB",
            retention="90 days",
            compression="gz"
        )
        
        # Error-only log file
        error_log_file = self.logs_dir / f"phoenixdrs_errors_{self.session_id}.log"
        logger.add(
            str(error_log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="10 MB",
            retention="1 year",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # Audit trail for forensic operations
        audit_log_file = self.logs_dir / f"phoenixdrs_audit_{self.session_id}.log"
        logger.add(
            str(audit_log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AUDIT | {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("audit", False),
            rotation="20 MB",
            retention="5 years",  # Long retention for legal requirements
            compression="gz"
        )
        
        # Performance monitoring log
        perf_log_file = self.logs_dir / f"phoenixdrs_performance_{self.session_id}.log"
        logger.add(
            str(perf_log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("performance", False),
            rotation="30 MB",
            retention="30 days"
        )
        
        self.logger = logger
    
    def _setup_standard_logging(self):
        """הגדרת logging סטנדרטי כ-fallback"""
        import logging
        import logging.handlers
        
        # Create logger
        self.logger = logging.getLogger("PhoenixDRS")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"phoenixdrs_{self.session_id}.log",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def _json_formatter(self, record) -> str:
        """פורמטר JSON מובנה"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "case_name": self.case_name,
            "examiner": self.examiner,
            "session_id": self.session_id
        }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record["extra"])
        
        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def info(self, message: str, **kwargs):
        """רישום מידע כללי"""
        if LOGURU_AVAILABLE:
            self.logger.bind(**kwargs).info(message)
        else:
            self.logger.info(f"{message} | {kwargs}")
    
    def debug(self, message: str, **kwargs):
        """רישום מידע debug"""
        if LOGURU_AVAILABLE:
            self.logger.bind(**kwargs).debug(message)
        else:
            self.logger.debug(f"{message} | {kwargs}")
    
    def warning(self, message: str, **kwargs):
        """רישום אזהרה"""
        if LOGURU_AVAILABLE:
            self.logger.bind(**kwargs).warning(message)
        else:
            self.logger.warning(f"{message} | {kwargs}")
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """רישום שגיאה"""
        if LOGURU_AVAILABLE:
            if exception:
                self.logger.bind(**kwargs).exception(message)
            else:
                self.logger.bind(**kwargs).error(message)
        else:
            if exception:
                self.logger.error(f"{message} | {kwargs}", exc_info=True)
            else:
                self.logger.error(f"{message} | {kwargs}")
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """רישום שגיאה קריטית"""
        if LOGURU_AVAILABLE:
            if exception:
                self.logger.bind(**kwargs).critical(message)
            else:
                self.logger.bind(**kwargs).critical(message)
        else:
            if exception:
                self.logger.critical(f"{message} | {kwargs}", exc_info=True)
            else:
                self.logger.critical(f"{message} | {kwargs}")
    
    def audit(self, action: str, target: str, result: str, **kwargs):
        """רישום לaudit trail"""
        audit_data = {
            "audit": True,
            "action": action,
            "target": target,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        message = f"AUDIT: {action} on {target} - {result}"
        
        if LOGURU_AVAILABLE:
            self.logger.bind(**audit_data).info(message)
        else:
            self.logger.info(f"{message} | {audit_data}")
    
    def performance(self, operation: str, duration: float, **kwargs):
        """רישום מדדי ביצועים"""
        perf_data = {
            "performance": True,
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        message = f"PERFORMANCE: {operation} took {duration:.3f}s"
        
        if LOGURU_AVAILABLE:
            self.logger.bind(**perf_data).info(message)
        else:
            self.logger.info(f"{message} | {perf_data}")
    
    def chain_of_custody(self, evidence_id: str, action: str, location: str, **kwargs):
        """רישום chain of custody"""
        custody_data = {
            "audit": True,
            "chain_of_custody": True,
            "evidence_id": evidence_id,
            "action": action,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "case_name": self.case_name,
            "examiner": self.examiner,
            **kwargs
        }
        
        message = f"CUSTODY: {action} of evidence {evidence_id} at {location}"
        
        if LOGURU_AVAILABLE:
            self.logger.bind(**custody_data).info(message)
        else:
            self.logger.info(f"{message} | {custody_data}")
    
    def start_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """תחילת פעולה עם tracking"""
        start_time = datetime.now()
        operation_id = f"{operation}_{start_time.strftime('%H%M%S')}"
        
        context = {
            "operation_id": operation_id,
            "operation": operation,
            "start_time": start_time,
            **kwargs
        }
        
        self.info(f"Starting operation: {operation}", operation_id=operation_id, **kwargs)
        return context
    
    def end_operation(self, context: Dict[str, Any], result: str = "SUCCESS", **kwargs):
        """סיום פעולה עם tracking"""
        end_time = datetime.now()
        duration = (end_time - context["start_time"]).total_seconds()
        
        operation_id = context["operation_id"]
        operation = context["operation"]
        
        self.info(
            f"Completed operation: {operation} - {result}",
            operation_id=operation_id,
            result=result,
            duration_seconds=duration,
            **kwargs
        )
        
        self.performance(operation, duration, result=result, **kwargs)
        
        # Audit log for significant operations
        if context.get("audit_required", True):
            self.audit(operation, context.get("target", "unknown"), result, 
                      operation_id=operation_id, duration_seconds=duration)


class PerformanceMonitor:
    """מוניטור ביצועים למערכת"""
    
    def __init__(self, logger: ForensicLogger):
        self.logger = logger
        self.operations = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log any uncompleted operations
        for op_id, context in self.operations.items():
            if not context.get("completed", False):
                self.logger.warning(f"Operation {op_id} was not properly completed")
    
    def start_timing(self, operation: str, **kwargs) -> str:
        """התחלת מדידת זמן"""
        context = self.logger.start_operation(operation, **kwargs)
        op_id = context["operation_id"]
        self.operations[op_id] = context
        return op_id
    
    def end_timing(self, operation_id: str, result: str = "SUCCESS", **kwargs):
        """סיום מדידת זמן"""
        if operation_id in self.operations:
            context = self.operations[operation_id]
            context["completed"] = True
            self.logger.end_operation(context, result, **kwargs)
            del self.operations[operation_id]
        else:
            self.logger.warning(f"Unknown operation ID for timing: {operation_id}")


# Global logger instance
_global_logger: Optional[ForensicLogger] = None


def setup_logging(case_name: Optional[str] = None, examiner: Optional[str] = None) -> ForensicLogger:
    """הגדרת logging גלובלי"""
    global _global_logger
    _global_logger = ForensicLogger(case_name, examiner)
    return _global_logger


def get_logger() -> ForensicLogger:
    """קבלת logger גלובלי"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ForensicLogger()
    return _global_logger


# Convenience functions
def log_info(message: str, **kwargs):
    """רישום מידע"""
    get_logger().info(message, **kwargs)


def log_debug(message: str, **kwargs):
    """רישום debug"""
    get_logger().debug(message, **kwargs)


def log_warning(message: str, **kwargs):
    """רישום אזהרה"""
    get_logger().warning(message, **kwargs)


def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    """רישום שגיאה"""
    get_logger().error(message, exception, **kwargs)


def log_audit(action: str, target: str, result: str, **kwargs):
    """רישום audit"""
    get_logger().audit(action, target, result, **kwargs)


def log_performance(operation: str, duration: float, **kwargs):
    """רישום ביצועים"""
    get_logger().performance(operation, duration, **kwargs)


def log_chain_of_custody(evidence_id: str, action: str, location: str, **kwargs):
    """רישום chain of custody"""
    get_logger().chain_of_custody(evidence_id, action, location, **kwargs)


# Context manager for operations
class LoggedOperation:
    """Context manager לפעולות עם logging אוטומטי"""
    
    def __init__(self, operation: str, target: str = "", audit: bool = True, **kwargs):
        self.operation = operation
        self.target = target
        self.audit = audit
        self.kwargs = kwargs
        self.logger = get_logger()
        self.context = None
    
    def __enter__(self):
        self.context = self.logger.start_operation(
            self.operation, target=self.target, audit_required=self.audit, **self.kwargs
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            result = "SUCCESS"
        else:
            result = f"FAILED: {exc_type.__name__}: {exc_val}"
            self.logger.error(f"Operation {self.operation} failed", exception=exc_val)
        
        self.logger.end_operation(self.context, result)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for a test case
    logger = setup_logging("TEST_CASE_001", "Test Examiner")
    
    # Basic logging
    logger.info("PhoenixDRS system started", version="1.0.0", system="Windows")
    logger.debug("Debug information", details="System initialization")
    logger.warning("This is a warning", component="TestModule")
    
    # Performance monitoring
    import time
    
    with PerformanceMonitor(logger) as perf:
        op_id = perf.start_timing("test_operation", target="test_file.dd")
        time.sleep(0.1)  # Simulate work
        perf.end_timing(op_id, "SUCCESS", files_processed=100)
    
    # Audit logging
    logger.audit("DISK_IMAGE_CREATED", "evidence_disk_001.dd", "SUCCESS", 
                md5="abcd1234", size_bytes=1000000)
    
    # Chain of custody
    logger.chain_of_custody("EVID_001", "ACQUIRED", "Lab Station 1", 
                           method="DD clone", hash_verified=True)
    
    # Using context manager
    with LoggedOperation("file_carving", "disk_image.dd") as op:
        time.sleep(0.05)  # Simulate carving
        logger.info("Carved 50 files", operation_id=op.context["operation_id"])
    
    # Error handling
    try:
        raise ValueError("Test error for logging")
    except ValueError as e:
        logger.error("Test error occurred", exception=e, component="TestModule")
    
    logger.info("PhoenixDRS test logging completed")