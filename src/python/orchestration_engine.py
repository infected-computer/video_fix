"""
Orchestration Engine for PhoenixDRS
Manages workflow orchestration, task scheduling, and AI coordination
"""

import asyncio
import logging
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import PriorityQueue, Empty
import uuid

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torchvision
import tensorflow as tf
from transformers import pipeline, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class WorkflowType(Enum):
    """Supported workflow types"""
    DATA_RECOVERY = "data_recovery"
    FORENSIC_ANALYSIS = "forensic_analysis"
    MEDIA_RECONSTRUCTION = "media_reconstruction"
    AI_CLASSIFICATION = "ai_classification"
    MALWARE_DETECTION = "malware_detection"
    METADATA_EXTRACTION = "metadata_extraction"
    CUSTOM = "custom"


@dataclass
class TaskResult:
    """Container for task execution results"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    id: str
    name: str
    operation: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    error_handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    type: WorkflowType
    tasks: List[WorkflowTask]
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class ResourceManager:
    """Manages system resources for task execution"""
    
    def __init__(self, max_cpu_threads: int = 8, max_gpu_memory_gb: float = 8.0):
        self.max_cpu_threads = max_cpu_threads
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.allocated_cpu_threads = 0
        self.allocated_gpu_memory = 0.0
        self.resource_lock = threading.Lock()
        self.allocations: Dict[str, Dict[str, Any]] = {}
        
    def allocate_resources(self, task_id: str, cpu_threads: int = 1, 
                          gpu_memory_gb: float = 0.0) -> bool:
        """Allocate resources for a task"""
        with self.resource_lock:
            if (self.allocated_cpu_threads + cpu_threads <= self.max_cpu_threads and
                self.allocated_gpu_memory + gpu_memory_gb <= self.max_gpu_memory_gb):
                
                self.allocated_cpu_threads += cpu_threads
                self.allocated_gpu_memory += gpu_memory_gb
                self.allocations[task_id] = {
                    'cpu_threads': cpu_threads,
                    'gpu_memory_gb': gpu_memory_gb,
                    'allocated_at': datetime.utcnow()
                }
                return True
            return False
    
    def release_resources(self, task_id: str) -> None:
        """Release resources allocated to a task"""
        with self.resource_lock:
            if task_id in self.allocations:
                allocation = self.allocations.pop(task_id)
                self.allocated_cpu_threads -= allocation['cpu_threads']
                self.allocated_gpu_memory -= allocation['gpu_memory_gb']
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get current resource availability"""
        with self.resource_lock:
            return {
                'available_cpu_threads': self.max_cpu_threads - self.allocated_cpu_threads,
                'available_gpu_memory_gb': self.max_gpu_memory_gb - self.allocated_gpu_memory,
                'total_cpu_threads': self.max_cpu_threads,
                'total_gpu_memory_gb': self.max_gpu_memory_gb,
                'active_allocations': len(self.allocations)
            }


class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    async def execute(self, task: WorkflowTask) -> TaskResult:
        """Execute a task and return the result"""
        pass
    
    @abstractmethod
    def can_handle(self, task: WorkflowTask) -> bool:
        """Check if this executor can handle the given task"""
        pass
    
    @abstractmethod
    def get_resource_requirements(self, task: WorkflowTask) -> Dict[str, Any]:
        """Get resource requirements for a task"""
        pass


class CPUTaskExecutor(TaskExecutor):
    """CPU-based task executor"""
    
    def __init__(self, thread_pool_size: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=thread_pool_size // 2)
    
    async def execute(self, task: WorkflowTask) -> TaskResult:
        """Execute task on CPU"""
        start_time = time.time()
        
        try:
            # Determine execution method based on task type
            if task.parameters.get('use_multiprocessing', False):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_pool, self._execute_task, task
                )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, self._execute_task, task
                )
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={'executor': 'CPU', 'threads_used': task.parameters.get('threads', 1)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.id} failed: {str(e)}")
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={'executor': 'CPU'}
            )
    
    def _execute_task(self, task: WorkflowTask) -> Any:
        """Internal task execution logic"""
        operation = task.operation
        parameters = task.parameters
        
        if operation == "file_analysis":
            return self._analyze_file(parameters)
        elif operation == "data_extraction":
            return self._extract_data(parameters)
        elif operation == "pattern_matching":
            return self._pattern_matching(parameters)
        elif operation == "text_processing":
            return self._process_text(parameters)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _analyze_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file structure and content"""
        file_path = Path(parameters['file_path'])
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Basic file analysis
        stat = file_path.stat()
        
        analysis = {
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime),
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'file_extension': file_path.suffix.lower(),
            'is_binary': self._is_binary_file(file_path)
        }
        
        # Content analysis based on file type
        if analysis['file_extension'] in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            analysis.update(self._analyze_image(file_path))
        elif analysis['file_extension'] in ['.mp4', '.avi', '.mov', '.mkv']:
            analysis.update(self._analyze_video(file_path))
        elif analysis['file_extension'] in ['.txt', '.log', '.csv']:
            analysis.update(self._analyze_text_file(file_path))
        
        return analysis
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except Exception:
            return True
    
    def _analyze_image(self, file_path: Path) -> Dict[str, Any]:
        """Analyze image file"""
        try:
            import PIL.Image
            import PIL.ExifTags
            
            with PIL.Image.open(file_path) as img:
                analysis = {
                    'image_width': img.width,
                    'image_height': img.height,
                    'image_mode': img.mode,
                    'image_format': img.format
                }
                
                # Extract EXIF data if available
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                
                analysis['exif_data'] = exif_data
                return analysis
                
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return {'image_analysis_error': str(e)}
    
    def _analyze_video(self, file_path: Path) -> Dict[str, Any]:
        """Analyze video file"""
        try:
            # Basic video analysis - would integrate with FFmpeg or similar
            return {
                'video_analysis': 'basic',
                'analysis_note': 'Full video analysis requires FFmpeg integration'
            }
        except Exception as e:
            logger.warning(f"Video analysis failed: {e}")
            return {'video_analysis_error': str(e)}
    
    def _analyze_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                'character_count': len(content),
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split()),
                'encoding': 'utf-8'  # Simplified encoding detection
            }
        except Exception as e:
            logger.warning(f"Text analysis failed: {e}")
            return {'text_analysis_error': str(e)}
    
    def _extract_data(self, parameters: Dict[str, Any]) -> Any:
        """Extract data based on patterns or rules"""
        # Implementation for data extraction
        return {"extracted_data": "placeholder"}
    
    def _pattern_matching(self, parameters: Dict[str, Any]) -> Any:
        """Perform pattern matching operations"""
        # Implementation for pattern matching
        return {"patterns_found": []}
    
    def _process_text(self, parameters: Dict[str, Any]) -> Any:
        """Process text content"""
        # Implementation for text processing
        return {"processed_text": "placeholder"}
    
    def can_handle(self, task: WorkflowTask) -> bool:
        """Check if this executor can handle the task"""
        cpu_operations = {
            "file_analysis", "data_extraction", "pattern_matching", 
            "text_processing", "encryption", "compression"
        }
        return task.operation in cpu_operations
    
    def get_resource_requirements(self, task: WorkflowTask) -> Dict[str, Any]:
        """Get resource requirements for the task"""
        return {
            'cpu_threads': task.parameters.get('threads', 1),
            'gpu_memory_gb': 0.0,
            'memory_mb': task.parameters.get('memory_mb', 512)
        }


class AITaskExecutor(TaskExecutor):
    """AI/ML task executor with model management"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize text classification pipeline
            self.models['text_classifier'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize image classification model
            if torchvision.models.resnet50:
                self.models['image_classifier'] = torchvision.models.resnet50(pretrained=True)
                self.models['image_classifier'].eval()
                self.models['image_classifier'].to(self.device)
            
            # Initialize file type classifier
            self.models['file_classifier'] = RandomForestClassifier(n_estimators=100)
            
            # Initialize malware detection model
            self.models['malware_detector'] = self._load_malware_model()
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    def _load_malware_model(self) -> Optional[BaseEstimator]:
        """Load or create malware detection model"""
        try:
            # In a real implementation, this would load a trained model
            # For now, create a simple placeholder model
            model = RandomForestClassifier(n_estimators=50)
            return model
        except Exception as e:
            logger.error(f"Failed to load malware model: {e}")
            return None
    
    async def execute(self, task: WorkflowTask) -> TaskResult:
        """Execute AI/ML task"""
        start_time = time.time()
        
        try:
            operation = task.operation
            parameters = task.parameters
            
            if operation == "classify_text":
                result = self._classify_text(parameters)
            elif operation == "classify_image":
                result = await self._classify_image(parameters)
            elif operation == "detect_malware":
                result = self._detect_malware(parameters)
            elif operation == "classify_file_type":
                result = self._classify_file_type(parameters)
            elif operation == "extract_features":
                result = self._extract_features(parameters)
            elif operation == "anomaly_detection":
                result = self._detect_anomalies(parameters)
            else:
                raise ValueError(f"Unknown AI operation: {operation}")
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={'executor': 'AI', 'device': str(self.device)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"AI task {task.id} failed: {str(e)}")
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={'executor': 'AI'}
            )
    
    def _classify_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text content"""
        text = parameters.get('text', '')
        
        if not text:
            raise ValueError("No text provided for classification")
        
        if 'text_classifier' not in self.models:
            raise ValueError("Text classifier not available")
        
        result = self.models['text_classifier'](text)
        
        return {
            'classification': result[0]['label'],
            'confidence': result[0]['score'],
            'text_length': len(text)
        }
    
    async def _classify_image(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image content"""
        image_path = parameters.get('image_path')
        
        if not image_path or not Path(image_path).exists():
            raise ValueError("Invalid image path provided")
        
        if 'image_classifier' not in self.models:
            raise ValueError("Image classifier not available")
        
        try:
            import torchvision.transforms as transforms
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.models['image_classifier'](input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # Load ImageNet class labels (simplified)
            # In practice, you'd load the actual ImageNet labels
            classes = [f"class_{i}" for i in range(1000)]
            
            results = []
            for i in range(5):
                results.append({
                    'class': classes[top5_catid[i]],
                    'probability': float(top5_prob[i])
                })
            
            return {
                'top_predictions': results,
                'image_size': image.size
            }
            
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            raise
    
    def _detect_malware(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect malware in file"""
        file_path = parameters.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            raise ValueError("Invalid file path provided")
        
        # Extract features for malware detection
        features = self._extract_malware_features(file_path)
        
        # Placeholder malware detection logic
        # In practice, this would use trained models
        risk_score = np.random.random()  # Placeholder
        
        return {
            'is_malware': risk_score > 0.7,
            'risk_score': float(risk_score),
            'features': features,
            'detection_method': 'ml_classifier'
        }
    
    def _extract_malware_features(self, file_path: str) -> Dict[str, Any]:
        """Extract features for malware detection"""
        path = Path(file_path)
        
        features = {
            'file_size': path.stat().st_size,
            'file_extension': path.suffix.lower(),
            'entropy': self._calculate_entropy(file_path),
            'pe_features': self._extract_pe_features(file_path) if path.suffix.lower() == '.exe' else None
        }
        
        return features
    
    def _calculate_entropy(self, file_path: str) -> float:
        """Calculate file entropy"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if not data:
                return 0.0
            
            # Calculate Shannon entropy
            byte_counts = np.bincount(data, minlength=256)
            probabilities = byte_counts / len(data)
            probabilities = probabilities[probabilities > 0]
            
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def _extract_pe_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract PE file features"""
        try:
            # Placeholder for PE analysis
            # In practice, would use pefile library
            return {
                'has_pe_header': True,
                'section_count': 5,  # Placeholder
                'imports_count': 10,  # Placeholder
                'exports_count': 0    # Placeholder
            }
        except Exception:
            return None
    
    def _classify_file_type(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify file type based on content"""
        file_path = parameters.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            raise ValueError("Invalid file path provided")
        
        # Extract features and classify
        features = self._extract_file_features(file_path)
        
        # Placeholder classification
        file_type = self._determine_file_type(file_path)
        
        return {
            'predicted_type': file_type,
            'confidence': 0.95,  # Placeholder
            'features': features
        }
    
    def _extract_file_features(self, file_path: str) -> Dict[str, Any]:
        """Extract features for file type classification"""
        path = Path(file_path)
        
        features = {
            'file_size': path.stat().st_size,
            'extension': path.suffix.lower(),
            'header_bytes': self._get_file_header(file_path)
        }
        
        return features
    
    def _get_file_header(self, file_path: str, size: int = 32) -> List[int]:
        """Get file header bytes"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(size)
            return list(header)
        except Exception:
            return []
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type from header analysis"""
        header = self._get_file_header(file_path, 16)
        
        if not header:
            return "unknown"
        
        # Magic number detection
        if header[:2] == [0xFF, 0xD8]:
            return "jpeg"
        elif header[:8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]:
            return "png"
        elif header[:4] == [0x25, 0x50, 0x44, 0x46]:
            return "pdf"
        elif header[:2] == [0x4D, 0x5A]:
            return "executable"
        else:
            return "unknown"
    
    def _extract_features(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from data"""
        data_type = parameters.get('data_type', 'file')
        
        if data_type == 'file':
            file_path = parameters.get('file_path')
            return self._extract_file_features(file_path)
        elif data_type == 'text':
            text = parameters.get('text', '')
            return self._extract_text_features(text)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text"""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'unique_words': len(set(text.lower().split())),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
    
    def _detect_anomalies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in data"""
        data = parameters.get('data')
        
        if not data:
            raise ValueError("No data provided for anomaly detection")
        
        # Placeholder anomaly detection
        anomaly_score = np.random.random()
        
        return {
            'is_anomaly': anomaly_score > 0.8,
            'anomaly_score': float(anomaly_score),
            'method': 'statistical'
        }
    
    def can_handle(self, task: WorkflowTask) -> bool:
        """Check if this executor can handle the task"""
        ai_operations = {
            "classify_text", "classify_image", "detect_malware",
            "classify_file_type", "extract_features", "anomaly_detection"
        }
        return task.operation in ai_operations
    
    def get_resource_requirements(self, task: WorkflowTask) -> Dict[str, Any]:
        """Get resource requirements for AI tasks"""
        base_requirements = {
            'cpu_threads': 2,
            'gpu_memory_gb': 0.5,
            'memory_mb': 1024
        }
        
        # Adjust based on operation
        if task.operation in ["classify_image", "detect_malware"]:
            base_requirements['gpu_memory_gb'] = 2.0
            base_requirements['memory_mb'] = 2048
        
        return base_requirements


class WorkflowOrchestrator:
    """Main orchestrator for managing workflows and task execution"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_workflows: Dict[str, Workflow] = {}
        self.task_queue = PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Initialize resource manager and executors
        self.resource_manager = ResourceManager()
        self.executors: List[TaskExecutor] = [
            CPUTaskExecutor(),
            AITaskExecutor()
        ]
        
        # State management
        self._running = False
        self._orchestrator_task = None
        
        logger.info("Workflow Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator"""
        if self._running:
            logger.warning("Orchestrator is already running")
            return
        
        self._running = True
        self._orchestrator_task = asyncio.create_task(self._orchestration_loop())
        logger.info("Workflow Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Wait for orchestrator task to complete
        if self._orchestrator_task:
            await self._orchestrator_task
        
        logger.info("Workflow Orchestrator stopped")
    
    async def submit_workflow(self, workflow: Workflow) -> str:
        """Submit a workflow for execution"""
        self.active_workflows[workflow.id] = workflow
        
        # Add all tasks to the queue
        for task in workflow.tasks:
            priority = -task.priority.value  # Negative for max heap behavior
            self.task_queue.put((priority, task))
        
        logger.info(f"Workflow {workflow.id} submitted with {len(workflow.tasks)} tasks")
        return workflow.id
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow and its tasks"""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        # Cancel running tasks for this workflow
        tasks_to_cancel = [
            task_id for task_id, task in self.running_tasks.items()
            if any(wf_task.id == task_id for wf_task in workflow.tasks)
        ]
        
        for task_id in tasks_to_cancel:
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                self.resource_manager.release_resources(task_id)
        
        # Remove workflow
        del self.active_workflows[workflow_id]
        
        logger.info(f"Workflow {workflow_id} cancelled")
        return True
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        task_statuses = {}
        
        for task in workflow.tasks:
            if task.id in self.running_tasks:
                task_statuses[task.id] = TaskStatus.RUNNING
            elif task.id in self.completed_tasks:
                task_statuses[task.id] = self.completed_tasks[task.id].status
            else:
                task_statuses[task.id] = TaskStatus.PENDING
        
        completed_count = sum(1 for status in task_statuses.values() 
                            if status == TaskStatus.COMPLETED)
        failed_count = sum(1 for status in task_statuses.values() 
                         if status == TaskStatus.FAILED)
        
        return {
            "workflow_id": workflow_id,
            "total_tasks": len(workflow.tasks),
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "running_tasks": len([s for s in task_statuses.values() 
                                if s == TaskStatus.RUNNING]),
            "progress": completed_count / len(workflow.tasks) if workflow.tasks else 0,
            "task_statuses": task_statuses
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "active_workflows": len(self.active_workflows),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "resource_usage": self.resource_manager.get_available_resources(),
            "orchestrator_running": self._running
        }
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self._running:
            try:
                # Process task queue
                await self._process_task_queue()
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_task_queue(self):
        """Process tasks from the queue"""
        while (len(self.running_tasks) < self.max_concurrent_tasks and 
               not self.task_queue.empty()):
            
            try:
                priority, task = self.task_queue.get_nowait()
                
                # Check dependencies
                if not self._are_dependencies_satisfied(task):
                    # Put task back in queue
                    self.task_queue.put((priority, task))
                    break
                
                # Find suitable executor
                executor = self._find_executor(task)
                if not executor:
                    logger.warning(f"No suitable executor found for task {task.id}")
                    continue
                
                # Check resource availability
                requirements = executor.get_resource_requirements(task)
                if not self.resource_manager.allocate_resources(
                    task.id, 
                    requirements.get('cpu_threads', 1),
                    requirements.get('gpu_memory_gb', 0.0)
                ):
                    # Put task back in queue
                    self.task_queue.put((priority, task))
                    break
                
                # Execute task
                task_coroutine = self._execute_task(task, executor)
                self.running_tasks[task.id] = asyncio.create_task(task_coroutine)
                
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
    
    def _are_dependencies_satisfied(self, task: WorkflowTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if (dep_id not in self.completed_tasks or 
                self.completed_tasks[dep_id].status != TaskStatus.COMPLETED):
                return False
        return True
    
    def _find_executor(self, task: WorkflowTask) -> Optional[TaskExecutor]:
        """Find suitable executor for a task"""
        for executor in self.executors:
            if executor.can_handle(task):
                return executor
        return None
    
    async def _execute_task(self, task: WorkflowTask, executor: TaskExecutor):
        """Execute a single task"""
        try:
            result = await executor.execute(task)
            self.completed_tasks[task.id] = result
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
            self.completed_tasks[task.id] = result
            
            # Call error handler if provided
            if task.error_handler:
                try:
                    task.error_handler(result)
                except Exception as eh_error:
                    logger.error(f"Error in task error handler: {eh_error}")
            
            logger.error(f"Task {task.id} failed: {e}")
        
        finally:
            # Release resources
            self.resource_manager.release_resources(task.id)
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed task references"""
        completed_task_ids = []
        
        for task_id, task in self.running_tasks.items():
            if task.done():
                completed_task_ids.append(task_id)
        
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]


# Example workflow definitions
def create_forensic_analysis_workflow(evidence_path: str) -> Workflow:
    """Create a forensic analysis workflow"""
    tasks = [
        WorkflowTask(
            id="file_analysis_001",
            name="Analyze Evidence File",
            operation="file_analysis",
            parameters={"file_path": evidence_path},
            priority=TaskPriority.HIGH
        ),
        WorkflowTask(
            id="malware_detection_001",
            name="Malware Detection",
            operation="detect_malware",
            parameters={"file_path": evidence_path},
            dependencies=["file_analysis_001"],
            priority=TaskPriority.HIGH
        ),
        WorkflowTask(
            id="feature_extraction_001",
            name="Extract Features",
            operation="extract_features",
            parameters={"data_type": "file", "file_path": evidence_path},
            dependencies=["file_analysis_001"],
            priority=TaskPriority.NORMAL
        )
    ]
    
    return Workflow(
        id="forensic_analysis_" + str(uuid.uuid4())[:8],
        name="Forensic Analysis Workflow",
        type=WorkflowType.FORENSIC_ANALYSIS,
        tasks=tasks
    )


def create_media_recovery_workflow(damaged_file_path: str) -> Workflow:
    """Create a media recovery workflow"""
    tasks = [
        WorkflowTask(
            id="header_analysis_001",
            name="Analyze File Header",
            operation="file_analysis",
            parameters={"file_path": damaged_file_path},
            priority=TaskPriority.CRITICAL
        ),
        WorkflowTask(
            id="type_classification_001",
            name="Classify File Type",
            operation="classify_file_type",
            parameters={"file_path": damaged_file_path},
            dependencies=["header_analysis_001"],
            priority=TaskPriority.HIGH
        ),
        WorkflowTask(
            id="content_extraction_001",
            name="Extract Content",
            operation="data_extraction",
            parameters={"file_path": damaged_file_path, "method": "carving"},
            dependencies=["type_classification_001"],
            priority=TaskPriority.HIGH
        )
    ]
    
    return Workflow(
        id="media_recovery_" + str(uuid.uuid4())[:8],
        name="Media Recovery Workflow",
        type=WorkflowType.MEDIA_RECONSTRUCTION,
        tasks=tasks
    )


# Example usage and testing
async def main():
    """Example usage of the orchestration engine"""
    orchestrator = WorkflowOrchestrator()
    
    try:
        await orchestrator.start()
        
        # Create and submit a test workflow
        workflow = create_forensic_analysis_workflow("/path/to/evidence.bin")
        workflow_id = await orchestrator.submit_workflow(workflow)
        
        # Monitor workflow progress
        while True:
            status = orchestrator.get_workflow_status(workflow_id)
            print(f"Workflow progress: {status['progress']:.2%}")
            
            if status['progress'] >= 1.0:
                break
            
            await asyncio.sleep(1.0)
        
        print("Workflow completed!")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())