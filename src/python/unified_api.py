"""
Unified API Interface for PhoenixDRS
Provides comprehensive REST API, GraphQL API, and WebSocket interfaces
"""

import asyncio
import json
import logging
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

# FastAPI and async frameworks
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.websockets import WebSocketState
import uvicorn

# GraphQL
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

# Database and caching
import redis
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Validation and serialization
from pydantic import BaseModel, Field, validator
from marshmallow import Schema, fields, ValidationError

# Background tasks and queues
from celery import Celery
from rq import Queue
import dramatiq

# Monitoring and observability
import prometheus_client
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class OperationStatus(str, Enum):
    """Operation status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class APIUser:
    """API user information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    rate_limit: int = 1000
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    is_active: bool = True


@dataclass
class APIOperation:
    """API operation tracking"""
    operation_id: str
    user_id: str
    operation_type: str
    status: OperationStatus
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    progress: int = 0


# Pydantic models for request/response validation
class FileAnalysisRequest(BaseModel):
    """File analysis request model"""
    file_path: str = Field(..., description="Path to the file to analyze")
    analysis_types: List[str] = Field(default=["all"], description="Types of analysis to perform")
    deep_scan: bool = Field(default=False, description="Perform deep analysis")
    include_metadata: bool = Field(default=True, description="Include file metadata")
    
    @validator('analysis_types')
    def validate_analysis_types(cls, v):
        valid_types = ["file_type", "malware", "content", "structure", "all"]
        for analysis_type in v:
            if analysis_type not in valid_types:
                raise ValueError(f"Invalid analysis type: {analysis_type}")
        return v


class FileAnalysisResponse(BaseModel):
    """File analysis response model"""
    operation_id: str
    file_path: str
    status: OperationStatus
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowCreateRequest(BaseModel):
    """Workflow creation request model"""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    workflow_type: str = Field(..., description="Type of workflow")
    tasks: List[Dict[str, Any]] = Field(..., description="Workflow tasks")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")
    
    @validator('workflow_type')
    def validate_workflow_type(cls, v):
        valid_types = ["data_recovery", "forensic_analysis", "media_reconstruction", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid workflow type: {v}")
        return v


class WorkflowResponse(BaseModel):
    """Workflow response model"""
    workflow_id: str
    name: str
    status: OperationStatus
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PluginInfo(BaseModel):
    """Plugin information model"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    is_active: bool
    capabilities: Dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: str
    version: str
    uptime: float
    memory_usage: Dict[str, Any]
    cpu_usage: float
    active_operations: int
    plugin_count: int
    api_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Authentication and authorization
class AuthenticationManager:
    """Handle API authentication and authorization"""
    
    def __init__(self, secret_key: str, redis_client: Optional[redis.Redis] = None):
        self.secret_key = secret_key
        self.redis_client = redis_client
        self.users: Dict[str, APIUser] = {}
        self.api_keys: Dict[str, str] = {}  # API key -> user_id mapping
        self.rate_limits: Dict[str, List[datetime]] = {}
        
    def create_user(self, username: str, email: str, role: UserRole, 
                   permissions: List[str] = None) -> APIUser:
        """Create a new API user"""
        user_id = str(uuid.uuid4())
        api_key = self.generate_api_key()
        
        user = APIUser(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions or [],
            api_key=api_key
        )
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        logger.info(f"Created API user: {username} ({role})")
        return user
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def create_jwt_token(self, user: APIUser, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token for user"""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "phoenixdrs-api"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[APIUser]:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            user.last_accessed = datetime.utcnow()
            
            return user
        except jwt.InvalidTokenError:
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[APIUser]:
        """Verify API key and return user"""
        if api_key not in self.api_keys:
            return None
        
        user_id = self.api_keys[api_key]
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        user.last_accessed = datetime.utcnow()
        
        return user
    
    def check_rate_limit(self, user: APIUser) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.utcnow()
        user_requests = self.rate_limits.get(user.user_id, [])
        
        # Remove requests older than 1 hour
        user_requests = [req_time for req_time in user_requests 
                        if now - req_time < timedelta(hours=1)]
        
        if len(user_requests) >= user.rate_limit:
            return False
        
        user_requests.append(now)
        self.rate_limits[user.user_id] = user_requests
        
        return True
    
    def has_permission(self, user: APIUser, permission: str) -> bool:
        """Check if user has specific permission"""
        if user.role == UserRole.ADMIN:
            return True
        
        return permission in user.permissions


# WebSocket connection manager
class WebSocketManager:
    """Manage WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}
        self.connection_subscriptions: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id] = [
                conn_id for conn_id in self.user_connections[user_id] 
                if conn_id != connection_id
            ]
            
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        if connection_id in self.connection_subscriptions:
            del self.connection_subscriptions[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                # Connection might be closed, remove it
                if connection_id in self.active_connections:
                    del self.active_connections[connection_id]
    
    async def send_user_message(self, message: Dict[str, Any], user_id: str):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_message(self, message: Dict[str, Any], subscription: Optional[str] = None):
        """Broadcast message to all or subscribed connections"""
        target_connections = []
        
        if subscription:
            # Send to subscribed connections only
            for connection_id, subscriptions in self.connection_subscriptions.items():
                if subscription in subscriptions:
                    target_connections.append(connection_id)
        else:
            # Send to all connections
            target_connections = list(self.active_connections.keys())
        
        for connection_id in target_connections:
            await self.send_personal_message(message, connection_id)
    
    def subscribe(self, connection_id: str, subscription: str):
        """Subscribe connection to specific events"""
        if connection_id not in self.connection_subscriptions:
            self.connection_subscriptions[connection_id] = []
        
        if subscription not in self.connection_subscriptions[connection_id]:
            self.connection_subscriptions[connection_id].append(subscription)
    
    def unsubscribe(self, connection_id: str, subscription: str):
        """Unsubscribe connection from specific events"""
        if connection_id in self.connection_subscriptions:
            subscriptions = self.connection_subscriptions[connection_id]
            if subscription in subscriptions:
                subscriptions.remove(subscription)


# GraphQL schema definitions
@strawberry.type
class FileAnalysisResultType:
    """GraphQL type for file analysis results"""
    operation_id: str
    file_path: str
    status: str
    results: strawberry.scalars.JSON
    confidence_scores: strawberry.scalars.JSON
    execution_time: float
    timestamp: datetime


@strawberry.type
class WorkflowType:
    """GraphQL type for workflows"""
    workflow_id: str
    name: str
    status: str
    progress: float
    created_at: datetime
    results: strawberry.scalars.JSON


@strawberry.type
class PluginType:
    """GraphQL type for plugins"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    is_active: bool
    capabilities: strawberry.scalars.JSON


@strawberry.type
class Query:
    """GraphQL queries"""
    
    @strawberry.field
    async def file_analysis_result(self, operation_id: str, info: Info) -> Optional[FileAnalysisResultType]:
        """Get file analysis result by operation ID"""
        # Implementation would query the actual system
        return None
    
    @strawberry.field
    async def workflows(self, info: Info) -> List[WorkflowType]:
        """Get all workflows"""
        # Implementation would query the actual system
        return []
    
    @strawberry.field
    async def plugins(self, info: Info) -> List[PluginType]:
        """Get all plugins"""
        # Implementation would query the actual system
        return []


@strawberry.type
class Mutation:
    """GraphQL mutations"""
    
    @strawberry.mutation
    async def analyze_file(self, file_path: str, analysis_types: List[str], info: Info) -> str:
        """Start file analysis"""
        # Implementation would start actual analysis
        return str(uuid.uuid4())
    
    @strawberry.mutation
    async def create_workflow(self, name: str, workflow_type: str, 
                            tasks: strawberry.scalars.JSON, info: Info) -> str:
        """Create new workflow"""
        # Implementation would create actual workflow
        return str(uuid.uuid4())


# Main API application
class UnifiedAPI:
    """Unified API interface for PhoenixDRS"""
    
    def __init__(self, secret_key: str = None, redis_url: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Initialize components
        self.auth_manager = AuthenticationManager(self.secret_key, self.redis_client)
        self.websocket_manager = WebSocketManager()
        self.operations: Dict[str, APIOperation] = {}
        
        # FastAPI app
        self.app = FastAPI(
            title="PhoenixDRS API",
            description="Professional Digital Forensics and Data Recovery API",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Setup GraphQL
        self.setup_graphql()
        
        # Setup WebSocket
        self.setup_websocket()
        
        # Create default admin user
        self.create_default_users()
        
        logger.info("Unified API initialized")
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for request logging and metrics
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = datetime.utcnow()
            response = await call_next(request)
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
            return response
    
    def setup_routes(self):
        """Setup REST API routes"""
        
        # Authentication dependency
        security = HTTPBearer()
        
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIUser:
            """Get current authenticated user"""
            token = credentials.credentials
            
            # Try JWT token first
            user = self.auth_manager.verify_jwt_token(token)
            if user:
                if not self.auth_manager.check_rate_limit(user):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                return user
            
            # Try API key
            user = self.auth_manager.verify_api_key(token)
            if user:
                if not self.auth_manager.check_rate_limit(user):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                return user
            
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        # Public endpoints
        @self.app.get("/")
        async def root():
            """API root endpoint"""
            return {
                "name": "PhoenixDRS API",
                "version": "2.0.0",
                "status": "operational",
                "documentation": "/docs"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        @self.app.post("/auth/login")
        async def login(username: str, password: str):
            """Login and get JWT token"""
            # In a real implementation, verify password against database
            user = None
            for u in self.auth_manager.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            token = self.auth_manager.create_jwt_token(user)
            return {"access_token": token, "token_type": "bearer", "user": user}
        
        @self.app.post("/auth/refresh")
        async def refresh_token(current_user: APIUser = Depends(get_current_user)):
            """Refresh JWT token"""
            token = self.auth_manager.create_jwt_token(current_user)
            return {"access_token": token, "token_type": "bearer"}
        
        # API v1 routes
        v1_router = self.app.include_router
        
        @self.app.get("/api/v1/status", response_model=SystemStatusResponse)
        async def get_system_status_v1(current_user: APIUser = Depends(get_current_user)):
            """Get system status"""
            return SystemStatusResponse(
                status="operational",
                version="2.0.0",
                uptime=0.0,  # Would calculate actual uptime
                memory_usage={"used": 0, "total": 0},
                cpu_usage=0.0,
                active_operations=len(self.operations),
                plugin_count=0,  # Would get from plugin manager
                api_version="v1"
            )
        
        @self.app.post("/api/v1/analyze/file", response_model=FileAnalysisResponse)
        async def analyze_file_v1(
            request: FileAnalysisRequest,
            current_user: APIUser = Depends(get_current_user)
        ):
            """Analyze a file"""
            if not self.auth_manager.has_permission(current_user, "file_analysis"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            operation_id = str(uuid.uuid4())
            
            # Create operation record
            operation = APIOperation(
                operation_id=operation_id,
                user_id=current_user.user_id,
                operation_type="file_analysis",
                status=OperationStatus.PENDING,
                parameters=request.dict()
            )
            
            self.operations[operation_id] = operation
            
            # In a real implementation, this would trigger actual analysis
            # For now, return a pending response
            return FileAnalysisResponse(
                operation_id=operation_id,
                file_path=request.file_path,
                status=OperationStatus.PENDING
            )
        
        @self.app.get("/api/v1/analyze/{operation_id}", response_model=FileAnalysisResponse)
        async def get_analysis_result_v1(
            operation_id: str,
            current_user: APIUser = Depends(get_current_user)
        ):
            """Get analysis result"""
            if operation_id not in self.operations:
                raise HTTPException(status_code=404, detail="Operation not found")
            
            operation = self.operations[operation_id]
            
            # Check if user owns this operation or has admin role
            if operation.user_id != current_user.user_id and current_user.role != UserRole.ADMIN:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return FileAnalysisResponse(
                operation_id=operation.operation_id,
                file_path=operation.parameters.get("file_path", ""),
                status=operation.status,
                results=operation.result or {},
                execution_time=0.0  # Would calculate actual time
            )
        
        @self.app.post("/api/v1/workflow", response_model=WorkflowResponse)
        async def create_workflow_v1(
            request: WorkflowCreateRequest,
            current_user: APIUser = Depends(get_current_user)
        ):
            """Create a new workflow"""
            if not self.auth_manager.has_permission(current_user, "workflow_management"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            workflow_id = str(uuid.uuid4())
            
            # In a real implementation, this would create actual workflow
            return WorkflowResponse(
                workflow_id=workflow_id,
                name=request.name,
                status=OperationStatus.PENDING,
                created_at=datetime.utcnow()
            )
        
        @self.app.get("/api/v1/workflow/{workflow_id}", response_model=WorkflowResponse)
        async def get_workflow_v1(
            workflow_id: str,
            current_user: APIUser = Depends(get_current_user)
        ):
            """Get workflow status"""
            # In a real implementation, this would query actual workflow
            return WorkflowResponse(
                workflow_id=workflow_id,
                name="Sample Workflow",
                status=OperationStatus.COMPLETED,
                progress=100.0,
                created_at=datetime.utcnow()
            )
        
        @self.app.get("/api/v1/plugins", response_model=List[PluginInfo])
        async def list_plugins_v1(current_user: APIUser = Depends(get_current_user)):
            """List available plugins"""
            # In a real implementation, this would query plugin manager
            return []
        
        @self.app.post("/api/v1/upload")
        async def upload_file_v1(
            file: UploadFile = File(...),
            current_user: APIUser = Depends(get_current_user)
        ):
            """Upload file for analysis"""
            if not self.auth_manager.has_permission(current_user, "file_upload"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # In a real implementation, this would save and process the file
            file_id = str(uuid.uuid4())
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type,
                "uploaded_at": datetime.utcnow()
            }
        
        # Stream endpoint for large file downloads
        @self.app.get("/api/v1/download/{file_id}")
        async def download_file_v1(
            file_id: str,
            current_user: APIUser = Depends(get_current_user)
        ):
            """Download file by ID"""
            # In a real implementation, this would stream the actual file
            def generate_dummy_content():
                for i in range(100):
                    yield f"Chunk {i}\n".encode()
            
            return StreamingResponse(
                generate_dummy_content(),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=file_{file_id}.bin"}
            )
    
    def setup_graphql(self):
        """Setup GraphQL endpoint"""
        schema = strawberry.Schema(query=Query, mutation=Mutation)
        graphql_app = GraphQLRouter(schema)
        
        self.app.include_router(graphql_app, prefix="/graphql")
    
    def setup_websocket(self):
        """Setup WebSocket endpoints"""
        
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            """WebSocket endpoint for real-time updates"""
            connection_id = str(uuid.uuid4())
            
            try:
                await self.websocket_manager.connect(websocket, connection_id, user_id)
                
                while True:
                    # Receive messages from client
                    data = await websocket.receive_json()
                    message_type = data.get("type")
                    
                    if message_type == "subscribe":
                        subscription = data.get("subscription")
                        if subscription:
                            self.websocket_manager.subscribe(connection_id, subscription)
                            await websocket.send_json({
                                "type": "subscription_success",
                                "subscription": subscription
                            })
                    
                    elif message_type == "unsubscribe":
                        subscription = data.get("subscription")
                        if subscription:
                            self.websocket_manager.unsubscribe(connection_id, subscription)
                            await websocket.send_json({
                                "type": "unsubscription_success",
                                "subscription": subscription
                            })
                    
                    elif message_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(connection_id, user_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(connection_id, user_id)
        
        @self.app.websocket("/ws/system")
        async def system_websocket(websocket: WebSocket):
            """System-wide WebSocket for monitoring"""
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            
            try:
                while True:
                    # Send periodic system updates
                    system_status = {
                        "type": "system_status",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_operations": len(self.operations),
                        "active_connections": len(self.websocket_manager.active_connections)
                    }
                    
                    await websocket.send_json(system_status)
                    await asyncio.sleep(5)  # Send updates every 5 seconds
            
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"System WebSocket error: {e}")
    
    def create_default_users(self):
        """Create default users for testing"""
        # Create admin user
        admin_user = self.auth_manager.create_user(
            username="admin",
            email="admin@phoenixdrs.com",
            role=UserRole.ADMIN,
            permissions=["*"]  # All permissions
        )
        
        # Create analyst user
        analyst_user = self.auth_manager.create_user(
            username="analyst",
            email="analyst@phoenixdrs.com",
            role=UserRole.ANALYST,
            permissions=["file_analysis", "workflow_management", "evidence_access"]
        )
        
        # Create API client
        api_client = self.auth_manager.create_user(
            username="api_client",
            email="api@phoenixdrs.com",
            role=UserRole.API_CLIENT,
            permissions=["file_analysis", "workflow_read"]
        )
        
        logger.info("Default users created:")
        logger.info(f"Admin API Key: {admin_user.api_key}")
        logger.info(f"Analyst API Key: {analyst_user.api_key}")
        logger.info(f"API Client Key: {api_client.api_key}")
    
    async def broadcast_operation_update(self, operation: APIOperation):
        """Broadcast operation status update via WebSocket"""
        message = {
            "type": "operation_update",
            "operation_id": operation.operation_id,
            "status": operation.status.value,
            "progress": operation.progress,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to specific user
        await self.websocket_manager.send_user_message(message, operation.user_id)
        
        # Broadcast to subscribers
        await self.websocket_manager.broadcast_message(message, "operation_updates")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        logger.info(f"Starting PhoenixDRS API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# CLI interface for API management
class APICLIManager:
    """CLI interface for API management"""
    
    def __init__(self, api: UnifiedAPI):
        self.api = api
    
    def create_user(self, username: str, email: str, role: str, permissions: List[str] = None):
        """Create a new API user via CLI"""
        try:
            user_role = UserRole(role)
            user = self.api.auth_manager.create_user(username, email, user_role, permissions)
            
            print(f"User created successfully:")
            print(f"  Username: {user.username}")
            print(f"  User ID: {user.user_id}")
            print(f"  Role: {user.role.value}")
            print(f"  API Key: {user.api_key}")
            
            return user
        except ValueError as e:
            print(f"Error: {e}")
            return None
    
    def list_users(self):
        """List all API users"""
        print("API Users:")
        print("-" * 80)
        print(f"{'Username':<20} {'Role':<15} {'Active':<8} {'Last Accessed':<20}")
        print("-" * 80)
        
        for user in self.api.auth_manager.users.values():
            last_accessed = user.last_accessed.strftime("%Y-%m-%d %H:%M:%S") if user.last_accessed else "Never"
            print(f"{user.username:<20} {user.role.value:<15} {'Yes' if user.is_active else 'No':<8} {last_accessed:<20}")
    
    def show_api_keys(self):
        """Show API keys for all users"""
        print("API Keys:")
        print("-" * 60)
        
        for user in self.api.auth_manager.users.values():
            print(f"{user.username}: {user.api_key}")
    
    def get_system_stats(self):
        """Get system statistics"""
        print("System Statistics:")
        print("-" * 40)
        print(f"Active Operations: {len(self.api.operations)}")
        print(f"Active WebSocket Connections: {len(self.api.websocket_manager.active_connections)}")
        print(f"Total API Users: {len(self.api.auth_manager.users)}")
        print(f"Rate Limit Entries: {len(self.api.auth_manager.rate_limits)}")


# Example usage and testing
async def example_api_usage():
    """Example API usage"""
    # Create API instance
    api = UnifiedAPI()
    
    # Example client usage would go here
    # This would typically be done by external clients
    
    print("API server ready at http://localhost:8000")
    print("Documentation available at http://localhost:8000/docs")
    
    # Run the server
    api.run(port=8000)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PhoenixDRS Unified API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--create-user", nargs=3, metavar=("USERNAME", "EMAIL", "ROLE"), 
                       help="Create a new user")
    parser.add_argument("--list-users", action="store_true", help="List all users")
    parser.add_argument("--show-keys", action="store_true", help="Show API keys")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    
    args = parser.parse_args()
    
    # Create API instance
    api = UnifiedAPI()
    cli = APICLIManager(api)
    
    # Handle CLI commands
    if args.create_user:
        username, email, role = args.create_user
        cli.create_user(username, email, role)
    elif args.list_users:
        cli.list_users()
    elif args.show_keys:
        cli.show_api_keys()
    elif args.stats:
        cli.get_system_stats()
    else:
        # Run API server
        api.run(host=args.host, port=args.port)