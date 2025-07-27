"""
PhoenixDRS - Advanced Parallel Processing Engine
מנוע עיבוד מקבילי מתקדם עבור PhoenixDRS

High-performance parallel processing system for data recovery operations.
Supports multi-threading, multi-processing, and distributed computing.
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Iterator, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import queue
import psutil

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available, using basic parallel processing")

from logging_config import get_logger, LoggedOperation, PerformanceMonitor


class ProcessingMode(Enum):
    """מצבי עיבוד מקבילי"""
    THREADS = "threads"           # Threading לI/O intensive
    PROCESSES = "processes"       # Multiprocessing לCPU intensive
    JOBLIB = "joblib"            # Joblib להתאמה אוטומטית
    DISTRIBUTED = "distributed"  # עיבוד מבוזר (עתידי)
    AUTO = "auto"                # בחירה אוטומטית


class TaskPriority(Enum):
    """עדיפות משימות"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """משימת עיבוד"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 0.0
    dependencies: List[str] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ProcessingResult:
    """תוצאת עיבוד"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    worker_id: Optional[str] = None
    memory_used: int = 0
    
    @property
    def processing_time(self) -> float:
        """זמן עיבוד במילישניות"""
        return self.duration * 1000


@dataclass  
class ProcessingStats:
    """סטטיסטיקות עיבוד"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    memory_peak: int = 0
    cpu_usage_avg: float = 0.0
    workers_active: int = 0


class SystemMonitor:
    """מוניטור משאבי מערכת"""
    
    def __init__(self):
        self.logger = get_logger()
        self._monitoring = False
        self._stats = {}
    
    def start_monitoring(self, interval: float = 1.0):
        """התחלת מעקב אחר משאבים"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.debug("System monitoring started")
    
    def stop_monitoring(self):
        """עצירת מעקב"""
        self._monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=2.0)
        self.logger.debug("System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """לולאת מעקב"""
        while self._monitoring:
            try:
                self._stats = {
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available': psutil.virtual_memory().available,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                    'timestamp': time.time()
                }
                time.sleep(interval)
            except Exception as e:
                self.logger.warning("Error in system monitoring", exception=e)
                time.sleep(interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """קבלת סטטיסטיקות נוכחיות"""
        return self._stats.copy()
    
    def get_optimal_worker_count(self, mode: ProcessingMode) -> int:
        """חישוב מספר workers אופטימלי"""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if mode == ProcessingMode.THREADS:
            # לI/O intensive - יותר threads
            return min(cpu_count * 4, 32)
        elif mode == ProcessingMode.PROCESSES:
            # לCPU intensive - לפי מספר ליבות
            return cpu_count
        elif mode == ProcessingMode.JOBLIB:
            # Joblib יחליט
            return -1  # Use all available
        else:
            # Auto-detection
            current_stats = self.get_current_stats()
            cpu_usage = current_stats.get('cpu_percent', 0)
            memory_usage = current_stats.get('memory_percent', 0)
            
            if cpu_usage > 80:
                return max(1, cpu_count // 2)  # Reduce load
            elif memory_usage > 80:
                return max(1, min(cpu_count, int(memory_gb)))
            else:
                return cpu_count


class TaskScheduler:
    """מתזמן משימות מתקדם"""
    
    def __init__(self, max_concurrent_tasks: int = None):
        self.logger = get_logger()
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.max_concurrent = max_concurrent_tasks or mp.cpu_count()
        self._lock = threading.Lock()
    
    def add_task(self, task: ProcessingTask):
        """הוספת משימה לתור"""
        with self._lock:
            # Priority queue עם priority score (נמוך יותר = עדיפות גבוהה יותר)
            priority_score = -task.priority.value
            self.pending_tasks.put((priority_score, time.time(), task))
            
            # מעקב אחר dependencies
            if task.dependencies:
                self.task_dependencies[task.task_id] = task.dependencies.copy()
            
            self.logger.debug(f"Task {task.task_id} added to queue", 
                            priority=task.priority.value,
                            dependencies=len(task.dependencies))
    
    def get_ready_tasks(self, max_count: int = None) -> List[ProcessingTask]:
        """קבלת משימות מוכנות לביצוע"""
        if max_count is None:
            max_count = self.max_concurrent - len(self.running_tasks)
        
        ready_tasks = []
        temp_tasks = []
        
        with self._lock:
            # בדיקת משימות בתור
            while not self.pending_tasks.empty() and len(ready_tasks) < max_count:
                try:
                    priority_score, timestamp, task = self.pending_tasks.get_nowait()
                    
                    # בדיקת dependencies
                    if self._are_dependencies_satisfied(task.task_id):
                        ready_tasks.append(task)
                        self.running_tasks[task.task_id] = task
                    else:
                        temp_tasks.append((priority_score, timestamp, task))
                        
                except queue.Empty:
                    break
            
            # החזרת משימות שלא מוכנות לתור
            for temp_task in temp_tasks:
                self.pending_tasks.put(temp_task)
        
        return ready_tasks
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """בדיקת מילוי dependencies"""
        if task_id not in self.task_dependencies:
            return True
        
        dependencies = self.task_dependencies[task_id]
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep_id].success:
                return False
        
        return True
    
    def mark_task_completed(self, task_id: str, result: ProcessingResult):
        """סימון משימה כהושלמה"""
        with self._lock:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self.completed_tasks[task_id] = result
            
            self.logger.debug(f"Task {task_id} completed", 
                            success=result.success,
                            duration=result.duration)
    
    def get_pending_count(self) -> int:
        """מספר משימות ממתינות"""
        return self.pending_tasks.qsize()
    
    def get_running_count(self) -> int:
        """מספר משימות פועלות"""
        return len(self.running_tasks)
    
    def get_completed_count(self) -> int:
        """מספר משימות שהושלמו"""
        return len(self.completed_tasks)


class ParallelProcessingEngine:
    """מנוע עיבוד מקבילי ראשי"""
    
    def __init__(self, 
                 mode: ProcessingMode = ProcessingMode.AUTO,
                 max_workers: Optional[int] = None,
                 enable_monitoring: bool = True):
        self.logger = get_logger()
        self.mode = mode
        self.max_workers = max_workers
        self.system_monitor = SystemMonitor() if enable_monitoring else None
        self.scheduler = TaskScheduler()
        self.stats = ProcessingStats()
        self._shutdown_event = threading.Event()
        
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        
        # בחירת מצב עיבוד אוטומטי
        if self.mode == ProcessingMode.AUTO:
            self.mode = self._detect_optimal_mode()
        
        # קביעת מספר workers
        if self.max_workers is None:
            if self.system_monitor:
                self.max_workers = self.system_monitor.get_optimal_worker_count(self.mode)
            else:
                self.max_workers = mp.cpu_count()
        
        self.logger.info(f"Parallel processing engine initialized", 
                        mode=self.mode.value,
                        max_workers=self.max_workers)
    
    def _detect_optimal_mode(self) -> ProcessingMode:
        """זיהוי מצב עיבוד אופטימלי"""
        if JOBLIB_AVAILABLE:
            return ProcessingMode.JOBLIB
        
        # בדיקת סוג העבודה הצפויה
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if cpu_count >= 8 and memory_gb >= 16:
            return ProcessingMode.PROCESSES
        else:
            return ProcessingMode.THREADS
    
    def submit_task(self, 
                   task_id: str,
                   function: Callable,
                   *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   dependencies: List[str] = None,
                   timeout: Optional[float] = None,
                   **kwargs) -> str:
        """הגשת משימה לעיבוד"""
        
        task = ProcessingTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout
        )
        
        self.scheduler.add_task(task)
        self.stats.total_tasks += 1
        
        self.logger.debug(f"Task submitted: {task_id}", 
                         priority=priority.value,
                         has_dependencies=bool(dependencies))
        
        return task_id
    
    def submit_batch(self, 
                    tasks: List[Tuple[str, Callable, tuple, dict]],
                    priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """הגשת אצווה של משימות"""
        
        task_ids = []
        for task_id, function, args, kwargs in tasks:
            submitted_id = self.submit_task(
                task_id, function, *args, 
                priority=priority, **kwargs
            )
            task_ids.append(submitted_id)
        
        self.logger.info(f"Batch of {len(tasks)} tasks submitted")
        return task_ids
    
    def process_all(self, timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """עיבוד כל המשימות"""
        
        with LoggedOperation(f"parallel_processing_{self.mode.value}") as op:
            start_time = time.time()
            
            if self.mode == ProcessingMode.JOBLIB and JOBLIB_AVAILABLE:
                results = self._process_with_joblib(timeout)
            elif self.mode == ProcessingMode.PROCESSES:
                results = self._process_with_multiprocessing(timeout)
            elif self.mode == ProcessingMode.THREADS:
                results = self._process_with_threading(timeout)
            else:
                results = self._process_with_threading(timeout)  # Fallback
            
            end_time = time.time()
            self._update_stats(start_time, end_time, results)
            
            self.logger.info("Parallel processing completed",
                           total_tasks=len(results),
                           successful_tasks=sum(1 for r in results.values() if r.success),
                           failed_tasks=sum(1 for r in results.values() if not r.success),
                           total_time=end_time - start_time)
            
            return results
    
    def _process_with_joblib(self, timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """עיבוד עם joblib"""
        self.logger.debug("Processing with joblib")
        
        results = {}
        ready_tasks = self.scheduler.get_ready_tasks()
        
        if not ready_tasks:
            return results
        
        # הכנת משימות ל-joblib
        joblib_tasks = []
        for task in ready_tasks:
            joblib_tasks.append(delayed(self._execute_task_safe)(task))
        
        # ביצוע מקבילי
        try:
            with Parallel(n_jobs=self.max_workers, backend='threading') as parallel:
                task_results = parallel(joblib_tasks)
            
            # עיבוד תוצאות
            for task, result in zip(ready_tasks, task_results):
                results[task.task_id] = result
                self.scheduler.mark_task_completed(task.task_id, result)
                
        except Exception as e:
            self.logger.error("Error in joblib processing", exception=e)
            
            # Fallback לtask בודדים
            for task in ready_tasks:
                result = self._execute_task_safe(task)
                results[task.task_id] = result
                self.scheduler.mark_task_completed(task.task_id, result)
        
        return results
    
    def _process_with_multiprocessing(self, timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """עיבוד עם multiprocessing"""
        self.logger.debug("Processing with multiprocessing")
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # הגשת משימות
            future_to_task = {}
            ready_tasks = self.scheduler.get_ready_tasks()
            
            for task in ready_tasks:
                future = executor.submit(self._execute_task_safe, task)
                future_to_task[future] = task
            
            # איסוף תוצאות
            try:
                for future in as_completed(future_to_task, timeout=timeout):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results[task.task_id] = result
                        self.scheduler.mark_task_completed(task.task_id, result)
                    except Exception as e:
                        error_result = ProcessingResult(
                            task_id=task.task_id,
                            success=False,
                            error=e,
                            start_time=time.time(),
                            end_time=time.time()
                        )
                        results[task.task_id] = error_result
                        self.scheduler.mark_task_completed(task.task_id, error_result)
                        
            except TimeoutError:
                self.logger.warning("Multiprocessing timeout reached")
        
        return results
    
    def _process_with_threading(self, timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """עיבוד עם threading"""
        self.logger.debug("Processing with threading")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # הגשת משימות
            future_to_task = {}
            ready_tasks = self.scheduler.get_ready_tasks()
            
            for task in ready_tasks:
                future = executor.submit(self._execute_task_safe, task)
                future_to_task[future] = task
            
            # איסוף תוצאות
            try:
                for future in as_completed(future_to_task, timeout=timeout):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results[task.task_id] = result
                        self.scheduler.mark_task_completed(task.task_id, result)
                    except Exception as e:
                        error_result = ProcessingResult(
                            task_id=task.task_id,
                            success=False,
                            error=e,
                            start_time=time.time(),
                            end_time=time.time()
                        )
                        results[task.task_id] = error_result
                        self.scheduler.mark_task_completed(task.task_id, error_result)
                        
            except TimeoutError:
                self.logger.warning("Threading timeout reached")
        
        return results
    
    def _execute_task_safe(self, task: ProcessingTask) -> ProcessingResult:
        """ביצוע משימה בטוח עם error handling"""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # ביצוע המשימה
            result = task.function(*task.args, **task.kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                worker_id=worker_id
            )
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"Task {task.task_id} failed", exception=e)
            
            return ProcessingResult(
                task_id=task.task_id,
                success=False,
                error=e,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                worker_id=worker_id
            )
    
    def _update_stats(self, start_time: float, end_time: float, results: Dict[str, ProcessingResult]):
        """עדכון סטטיסטיקות"""
        total_time = end_time - start_time
        
        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)
        
        if results:
            avg_time = sum(r.duration for r in results.values()) / len(results)
        else:
            avg_time = 0.0
        
        self.stats.completed_tasks += successful
        self.stats.failed_tasks += failed
        self.stats.avg_processing_time = avg_time
        self.stats.total_processing_time += total_time
        self.stats.throughput_per_second = len(results) / total_time if total_time > 0 else 0.0
        
        if self.system_monitor:
            current_stats = self.system_monitor.get_current_stats()
            self.stats.memory_peak = max(self.stats.memory_peak, 
                                       current_stats.get('memory_percent', 0))
            self.stats.cpu_usage_avg = current_stats.get('cpu_percent', 0)
    
    def get_stats(self) -> ProcessingStats:
        """קבלת סטטיסטיקות"""
        return self.stats
    
    def shutdown(self):
        """כיבוי מנוע העיבוד"""
        self._shutdown_event.set()
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        
        self.logger.info("Parallel processing engine shutdown")


# Utility functions for common parallel operations
def parallel_file_carving(image_chunks: List[Tuple[str, int, int]], 
                         signature_db_path: str,
                         output_dir: str,
                         max_workers: int = None) -> List[Any]:
    """חיתוך קבצים מקבילי"""
    
    engine = ParallelProcessingEngine(
        mode=ProcessingMode.THREADS,  # I/O intensive
        max_workers=max_workers
    )
    
    # הגשת משימות חיתוך
    for i, (chunk_path, start_offset, size) in enumerate(image_chunks):
        engine.submit_task(
            task_id=f"carve_chunk_{i}",
            function=_carve_chunk,
            chunk_path, start_offset, size, signature_db_path, output_dir,
            priority=TaskPriority.NORMAL
        )
    
    # עיבוד כל המשימות
    results = engine.process_all()
    engine.shutdown()
    
    # החזרת תוצאות מוצלחות
    successful_results = []
    for result in results.values():
        if result.success:
            successful_results.extend(result.result)
    
    return successful_results


def parallel_filesystem_analysis(disk_paths: List[str], 
                                max_workers: int = None) -> Dict[str, Any]:
    """ניתוח מערכות קבצים מקבילי"""
    
    engine = ParallelProcessingEngine(
        mode=ProcessingMode.PROCESSES,  # CPU intensive
        max_workers=max_workers
    )
    
    # הגשת משימות ניתוח
    for i, disk_path in enumerate(disk_paths):
        engine.submit_task(
            task_id=f"analyze_fs_{i}",
            function=_analyze_filesystem,
            disk_path,
            priority=TaskPriority.HIGH
        )
    
    # עיבוד כל המשימות
    results = engine.process_all()
    engine.shutdown()
    
    # עיבוד תוצאות
    analysis_results = {}
    for task_id, result in results.items():
        if result.success:
            disk_path = disk_paths[int(task_id.split('_')[-1])]
            analysis_results[disk_path] = result.result
    
    return analysis_results


def _carve_chunk(chunk_path: str, start_offset: int, size: int, 
                signature_db_path: str, output_dir: str) -> List[Any]:
    """פונקציית עזר לחיתוך chunk"""
    from file_carver import FileCarver
    
    carver = FileCarver()
    # מימוש מפושט - בפועל צריך לטפל ב-chunk ספציפי
    return carver.carve(chunk_path, signature_db_path, output_dir)


def _analyze_filesystem(disk_path: str) -> Dict[str, Any]:
    """פונקציית עזר לניתוח מערכת קבצים"""
    from filesystem_parser import FilesystemDetector
    
    fs_type = FilesystemDetector.detect_filesystem_type(disk_path)
    parser = FilesystemDetector.get_parser(disk_path)
    
    if parser:
        with parser as p:
            if p.detect_filesystem():
                fs_info = p.parse_filesystem_info()
                return {
                    'filesystem_type': fs_info.filesystem_type.value,
                    'total_size': fs_info.total_size,
                    'cluster_size': fs_info.cluster_size,
                    'volume_label': fs_info.volume_label
                }
    
    return {'filesystem_type': 'unknown', 'error': 'Could not analyze filesystem'}


# Example usage and testing
if __name__ == "__main__":
    # דוגמה לשימוש במנוע עיבוד מקבילי
    
    def sample_task(task_name: str, duration: float = 1.0) -> str:
        """משימה מדומה לבדיקה"""
        time.sleep(duration)
        return f"Completed {task_name}"
    
    # יצירת מנוע עיבוד
    engine = ParallelProcessingEngine(
        mode=ProcessingMode.AUTO,
        max_workers=4
    )
    
    # הגשת משימות
    for i in range(10):
        engine.submit_task(
            task_id=f"task_{i}",
            function=sample_task,
            f"Sample Task {i}", 0.5,
            priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH
        )
    
    # עיבוד כל המשימות
    results = engine.process_all(timeout=30.0)
    
    # הדפסת תוצאות
    print(f"Processed {len(results)} tasks:")
    for task_id, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"  {status} {task_id}: {result.result if result.success else result.error}")
    
    # הדפסת סטטיסטיקות
    stats = engine.get_stats()
    print(f"\nStatistics:")
    print(f"  Total tasks: {stats.total_tasks}")
    print(f"  Completed: {stats.completed_tasks}")
    print(f"  Failed: {stats.failed_tasks}")
    print(f"  Average time: {stats.avg_processing_time:.3f}s")
    print(f"  Throughput: {stats.throughput_per_second:.2f} tasks/sec")
    
    # כיבוי המנוע
    engine.shutdown()