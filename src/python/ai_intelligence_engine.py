"""
AI Intelligence Engine for PhoenixDRS
Advanced AI/ML capabilities for forensic analysis and data recovery
"""

import logging
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import joblib
from collections import defaultdict

# Machine Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep learning frameworks
import tensorflow as tf
from transformers import (
    AutoModel, AutoTokenizer, pipeline,
    DistilBertForSequenceClassification, BertTokenizer
)

# Specialized libraries
import cv2
import librosa
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for AI models"""
    name: str
    version: str
    model_type: str
    accuracy: float
    training_date: datetime
    input_shape: Optional[Tuple] = None
    output_classes: Optional[List[str]] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result container for AI analysis"""
    analysis_type: str
    confidence: float
    primary_result: Any
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseAIModel(ABC):
    """Abstract base class for AI models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.metadata = None
        self.scaler = None
        self.label_encoder = None
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, file_path: str) -> bool:
        """Save the model to disk"""
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> bool:
        """Load the model from disk"""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None


class FileTypeClassifier(BaseAIModel):
    """AI model for classifying file types based on content"""
    
    def __init__(self):
        super().__init__("FileTypeClassifier")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_extractors = {
            'header_bytes': self._extract_header_features,
            'entropy': self._calculate_entropy,
            'byte_distribution': self._extract_byte_distribution,
            'magic_numbers': self._extract_magic_numbers
        }
    
    def train(self, file_paths: List[str], labels: List[str], **kwargs) -> Dict[str, float]:
        """Train the file type classifier"""
        logger.info(f"Training {self.model_name} with {len(file_paths)} samples")
        
        # Extract features from files
        features = []
        valid_labels = []
        
        for file_path, label in zip(file_paths, labels):
            try:
                file_features = self._extract_file_features(file_path)
                features.append(file_features)
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract features from {file_path}: {e}")
        
        if not features:
            raise ValueError("No valid features extracted")
        
        X = np.array(features)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(valid_labels)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        self.is_trained = True
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"Training completed. Test accuracy: {test_score:.3f}")
        return metrics
    
    def predict(self, file_path: str) -> AnalysisResult:
        """Predict file type"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        features = self._extract_file_features(file_path)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'class': self.label_encoder.inverse_transform([idx])[0],
                'probability': probabilities[idx]
            }
            for idx in top_indices
        ]
        
        return AnalysisResult(
            analysis_type="file_type_classification",
            confidence=confidence,
            primary_result=predicted_class,
            detailed_results={
                'top_predictions': top_predictions,
                'feature_values': dict(zip(
                    ['header', 'entropy', 'byte_dist', 'magic'],
                    features
                ))
            }
        )
    
    def _extract_file_features(self, file_path: str) -> List[float]:
        """Extract features from a file"""
        features = []
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                feature_value = extractor_func(file_path)
                features.extend(feature_value if isinstance(feature_value, list) else [feature_value])
            except Exception as e:
                logger.warning(f"Feature extraction {extractor_name} failed for {file_path}: {e}")
                # Add default values
                if extractor_name == 'header_bytes':
                    features.extend([0] * 32)
                elif extractor_name == 'byte_distribution':
                    features.extend([0] * 256)
                else:
                    features.append(0)
        
        return features
    
    def _extract_header_features(self, file_path: str) -> List[int]:
        """Extract header byte features"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
            return list(header) + [0] * (32 - len(header))
        except Exception:
            return [0] * 32
    
    def _calculate_entropy(self, file_path: str) -> float:
        """Calculate file entropy"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if not data:
                return 0.0
            
            byte_counts = np.bincount(data, minlength=256)
            probabilities = byte_counts / len(data)
            probabilities = probabilities[probabilities > 0]
            
            return float(-np.sum(probabilities * np.log2(probabilities)))
        except Exception:
            return 0.0
    
    def _extract_byte_distribution(self, file_path: str) -> List[float]:
        """Extract byte frequency distribution"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(1024)  # Sample first 1KB
            
            byte_counts = np.bincount(data, minlength=256)
            return (byte_counts / len(data)).tolist()
        except Exception:
            return [0.0] * 256
    
    def _extract_magic_numbers(self, file_path: str) -> float:
        """Extract magic number features"""
        magic_signatures = {
            b'\xFF\xD8\xFF': 1.0,  # JPEG
            b'\x89PNG\r\n\x1a\n': 2.0,  # PNG
            b'%PDF': 3.0,  # PDF
            b'PK\x03\x04': 4.0,  # ZIP
            b'MZ': 5.0,  # PE executable
        }
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            for signature, value in magic_signatures.items():
                if header.startswith(signature):
                    return value
            
            return 0.0
        except Exception:
            return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        feature_names = ['header_bytes'] * 32 + ['entropy'] + ['byte_dist'] * 256 + ['magic_numbers']
        importance_dict = dict(zip(feature_names, self.model.feature_importances_))
        
        # Group by feature type
        grouped = defaultdict(float)
        for name, importance in importance_dict.items():
            if name.startswith('header_bytes'):
                grouped['header_bytes'] += importance
            elif name.startswith('byte_dist'):
                grouped['byte_distribution'] += importance
            else:
                grouped[name] = importance
        
        return dict(grouped)
    
    def save_model(self, file_path: str) -> bool:
        """Save the model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'is_trained': self.is_trained,
                'metadata': self.metadata
            }
            joblib.dump(model_data, file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """Load the model"""
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            self.metadata = model_data.get('metadata')
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class MalwareDetector(BaseAIModel):
    """AI model for malware detection"""
    
    def __init__(self):
        super().__init__("MalwareDetector")
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.use_anomaly_detection = True
    
    def train(self, file_paths: List[str], labels: List[str], **kwargs) -> Dict[str, float]:
        """Train malware detection model"""
        logger.info(f"Training {self.model_name} with {len(file_paths)} samples")
        
        features = []
        valid_labels = []
        
        for file_path, label in zip(file_paths, labels):
            try:
                malware_features = self._extract_malware_features(file_path)
                features.append(malware_features)
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract features from {file_path}: {e}")
        
        if not features:
            raise ValueError("No valid features extracted")
        
        X = np.array(features)
        
        # For anomaly detection, train on benign samples only
        benign_mask = np.array(valid_labels) == 'benign'
        X_benign = X[benign_mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_benign_scaled = X_scaled[benign_mask]
        
        # Train anomaly detector on benign samples
        self.model.fit(X_benign_scaled)
        
        # Also train a classifier for comparison
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(valid_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate both models
        anomaly_predictions = self.model.predict(X_test)
        # Convert -1 (outlier) to 1 (malware), 1 (inlier) to 0 (benign)
        anomaly_predictions = (anomaly_predictions == -1).astype(int)
        
        classifier_predictions = self.classifier.predict(X_test)
        
        self.is_trained = True
        
        metrics = {
            'classifier_accuracy': self.classifier.score(X_test, y_test),
            'anomaly_detector_accuracy': np.mean(anomaly_predictions == y_test)
        }
        
        logger.info(f"Training completed. Classifier accuracy: {metrics['classifier_accuracy']:.3f}")
        return metrics
    
    def predict(self, file_path: str) -> AnalysisResult:
        """Predict if file is malware"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        features = self._extract_malware_features(file_path)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        anomaly_score = self.model.decision_function(X_scaled)[0]
        anomaly_prediction = self.model.predict(X_scaled)[0]
        
        classifier_prediction = self.classifier.predict(X_scaled)[0]
        classifier_probabilities = self.classifier.predict_proba(X_scaled)[0]
        
        # Combine results
        is_malware_anomaly = anomaly_prediction == -1
        is_malware_classifier = classifier_prediction == 1
        
        # Use classifier confidence as primary confidence
        confidence = classifier_probabilities.max()
        
        # Final decision (can be customized)
        if self.use_anomaly_detection:
            is_malware = is_malware_anomaly or is_malware_classifier
        else:
            is_malware = is_malware_classifier
        
        risk_score = 1.0 - anomaly_score if is_malware_anomaly else classifier_probabilities[1]
        
        return AnalysisResult(
            analysis_type="malware_detection",
            confidence=confidence,
            primary_result=is_malware,
            detailed_results={
                'risk_score': float(risk_score),
                'anomaly_score': float(anomaly_score),
                'classifier_probabilities': classifier_probabilities.tolist(),
                'detection_methods': {
                    'anomaly_detection': is_malware_anomaly,
                    'classification': is_malware_classifier
                }
            }
        )
    
    def _extract_malware_features(self, file_path: str) -> List[float]:
        """Extract malware-specific features"""
        features = []
        
        # File size features
        try:
            file_size = Path(file_path).stat().st_size
            features.extend([
                file_size,
                np.log10(file_size + 1),  # Log file size
                file_size % 1024,  # File size modulo
            ])
        except Exception:
            features.extend([0, 0, 0])
        
        # Entropy features
        entropy = self._calculate_entropy_sections(file_path)
        features.extend(entropy)
        
        # Byte frequency features
        byte_freq = self._calculate_byte_frequencies(file_path)
        features.extend(byte_freq)
        
        # String features
        string_features = self._extract_string_features(file_path)
        features.extend(string_features)
        
        # PE features (if applicable)
        pe_features = self._extract_pe_features(file_path)
        features.extend(pe_features)
        
        return features
    
    def _calculate_entropy_sections(self, file_path: str, sections: int = 10) -> List[float]:
        """Calculate entropy for different sections of the file"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if not data:
                return [0.0] * sections
            
            section_size = len(data) // sections
            entropies = []
            
            for i in range(sections):
                start = i * section_size
                end = start + section_size if i < sections - 1 else len(data)
                section_data = data[start:end]
                
                if section_data:
                    byte_counts = np.bincount(section_data, minlength=256)
                    probabilities = byte_counts / len(section_data)
                    probabilities = probabilities[probabilities > 0]
                    entropy = -np.sum(probabilities * np.log2(probabilities))
                    entropies.append(entropy)
                else:
                    entropies.append(0.0)
            
            return entropies
        except Exception:
            return [0.0] * sections
    
    def _calculate_byte_frequencies(self, file_path: str) -> List[float]:
        """Calculate byte frequency statistics"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(8192)  # Sample first 8KB
            
            if not data:
                return [0] * 10
            
            byte_counts = np.bincount(data, minlength=256)
            
            # Statistical features
            features = [
                np.mean(byte_counts),
                np.std(byte_counts),
                np.max(byte_counts),
                np.min(byte_counts),
                np.median(byte_counts),
                len(np.nonzero(byte_counts)[0]),  # Number of unique bytes
                np.sum(byte_counts > 0),  # Non-zero byte types
                np.sum(byte_counts == 1),  # Singleton bytes
                np.sum(byte_counts > np.mean(byte_counts)),  # Above average
                np.var(byte_counts)  # Variance
            ]
            
            return features
        except Exception:
            return [0] * 10
    
    def _extract_string_features(self, file_path: str) -> List[float]:
        """Extract string-based features"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(16384)  # Sample first 16KB
            
            # Count printable strings
            printable_chars = sum(1 for b in data if 32 <= b <= 126)
            printable_ratio = printable_chars / len(data) if data else 0
            
            # Count specific patterns
            null_bytes = data.count(0)
            high_bytes = sum(1 for b in data if b > 127)
            
            features = [
                printable_ratio,
                null_bytes / len(data) if data else 0,
                high_bytes / len(data) if data else 0,
                len(set(data)) / 256,  # Byte diversity
                data.count(b'\\x00\\x00') / (len(data) - 1) if len(data) > 1 else 0  # Null pairs
            ]
            
            return features
        except Exception:
            return [0] * 5
    
    def _extract_pe_features(self, file_path: str) -> List[float]:
        """Extract PE-specific features"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for PE signature
            if len(header) < 64 or header[:2] != b'MZ':
                return [0] * 5
            
            # Basic PE features (simplified)
            pe_offset = int.from_bytes(header[60:64], 'little')
            
            features = [
                1.0,  # Is PE file
                pe_offset,  # PE header offset
                len(header),  # Header size sampled
                header.count(0),  # Null bytes in header
                len(set(header))  # Unique bytes in header
            ]
            
            return features
        except Exception:
            return [0] * 5
    
    def save_model(self, file_path: str) -> bool:
        """Save the malware detection model"""
        try:
            model_data = {
                'anomaly_model': self.model,
                'classifier': self.classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'is_trained': self.is_trained,
                'use_anomaly_detection': self.use_anomaly_detection
            }
            joblib.dump(model_data, file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save malware model: {e}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """Load the malware detection model"""
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['anomaly_model']
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            self.use_anomaly_detection = model_data.get('use_anomaly_detection', True)
            return True
        except Exception as e:
            logger.error(f"Failed to load malware model: {e}")
            return False


class ContentAnalyzer(BaseAIModel):
    """AI model for analyzing file content and extracting insights"""
    
    def __init__(self):
        super().__init__("ContentAnalyzer")
        self.text_models = {}
        self.image_models = {}
        self.audio_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize various content analysis models"""
        try:
            # Text analysis models
            self.text_models['sentiment'] = SentimentIntensityAnalyzer()
            self.text_models['language_detector'] = None  # Would use langdetect
            
            # Image analysis models
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            self.image_models['classifier'] = resnet50(pretrained=True)
            self.image_models['classifier'].eval()
            self.image_models['device'] = device
            
            logger.info("Content analyzer models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize content models: {e}")
    
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, float]:
        """Train content analysis models"""
        # Content analyzer typically uses pre-trained models
        # This method would be used for fine-tuning
        return {'status': 'pre-trained models loaded'}
    
    def predict(self, file_path: str, analysis_type: str = 'auto') -> AnalysisResult:
        """Analyze file content"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine analysis type based on file extension
        if analysis_type == 'auto':
            extension = file_path.suffix.lower()
            if extension in ['.txt', '.log', '.csv', '.json']:
                analysis_type = 'text'
            elif extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                analysis_type = 'image'
            elif extension in ['.mp3', '.wav', '.flac']:
                analysis_type = 'audio'
            else:
                analysis_type = 'binary'
        
        if analysis_type == 'text':
            return self._analyze_text_file(file_path)
        elif analysis_type == 'image':
            return self._analyze_image_file(file_path)
        elif analysis_type == 'audio':
            return self._analyze_audio_file(file_path)
        else:
            return self._analyze_binary_file(file_path)
    
    def _analyze_text_file(self, file_path: Path) -> AnalysisResult:
        """Analyze text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return AnalysisResult(
                analysis_type="text_analysis",
                confidence=0.0,
                primary_result="error",
                detailed_results={'error': str(e)}
            )
        
        analysis = {}
        
        # Basic text statistics
        analysis['basic_stats'] = {
            'character_count': len(content),
            'word_count': len(content.split()),
            'line_count': content.count('\n') + 1,
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        }
        
        # Sentiment analysis
        if 'sentiment' in self.text_models:
            try:
                sentiment_scores = self.text_models['sentiment'].polarity_scores(content)
                analysis['sentiment'] = sentiment_scores
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Language detection (placeholder)
        analysis['language'] = {'detected': 'en', 'confidence': 0.9}
        
        # Content classification
        content_type = self._classify_text_content(content)
        analysis['content_type'] = content_type
        
        # Extract key phrases and entities
        analysis['entities'] = self._extract_entities(content)
        
        return AnalysisResult(
            analysis_type="text_analysis",
            confidence=0.85,
            primary_result=content_type,
            detailed_results=analysis
        )
    
    def _classify_text_content(self, content: str) -> str:
        """Classify text content type"""
        content_lower = content.lower()
        
        # Simple heuristic classification
        if any(keyword in content_lower for keyword in ['error', 'exception', 'stack trace', 'warning']):
            return 'log_file'
        elif any(keyword in content_lower for keyword in ['select', 'insert', 'update', 'delete', 'create table']):
            return 'sql_file'
        elif content.count(',') > content.count(' ') and '\n' in content:
            return 'csv_data'
        elif content.strip().startswith('{') or content.strip().startswith('['):
            return 'json_data'
        elif any(keyword in content_lower for keyword in ['<html', '<head', '<body', '<div']):
            return 'html_file'
        else:
            return 'plain_text'
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        # Simplified entity extraction
        import re
        
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content),
            'ip_addresses': re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', content),
            'phone_numbers': re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b', content)
        }
        
        return entities
    
    def _analyze_image_file(self, file_path: Path) -> AnalysisResult:
        """Analyze image file content"""
        try:
            import PIL.Image
            
            with PIL.Image.open(file_path) as img:
                # Basic image properties
                basic_info = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'has_transparency': img.mode in ('RGBA', 'LA'),
                }
                
                # Image classification using pre-trained model
                classification = self._classify_image_content(img)
                
                # Color analysis
                color_analysis = self._analyze_image_colors(img)
                
                # EXIF data extraction
                exif_data = self._extract_exif_data(img)
                
                analysis = {
                    'basic_info': basic_info,
                    'classification': classification,
                    'color_analysis': color_analysis,
                    'exif_data': exif_data
                }
                
                return AnalysisResult(
                    analysis_type="image_analysis",
                    confidence=0.9,
                    primary_result=classification['top_class'],
                    detailed_results=analysis
                )
                
        except Exception as e:
            return AnalysisResult(
                analysis_type="image_analysis",
                confidence=0.0,
                primary_result="error",
                detailed_results={'error': str(e)}
            )
    
    def _classify_image_content(self, img) -> Dict[str, Any]:
        """Classify image content using deep learning"""
        try:
            # Preprocess image
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            input_tensor = preprocess(img).unsqueeze(0)
            if 'device' in self.image_models:
                input_tensor = input_tensor.to(self.image_models['device'])
            
            # Classify
            with torch.no_grad():
                outputs = self.image_models['classifier'](input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # Simplified class names (in practice, would load ImageNet classes)
            classes = [f"class_{i}" for i in range(1000)]
            
            top_predictions = []
            for i in range(5):
                top_predictions.append({
                    'class': classes[top5_catid[i]],
                    'probability': float(top5_prob[i])
                })
            
            return {
                'top_class': top_predictions[0]['class'],
                'confidence': float(top5_prob[0]),
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            return {'top_class': 'unknown', 'confidence': 0.0}
    
    def _analyze_image_colors(self, img) -> Dict[str, Any]:
        """Analyze image color distribution"""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get color histogram
            img_array = np.array(img)
            
            # Calculate color statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            dominant_colors = self._get_dominant_colors(img_array)
            
            return {
                'mean_rgb': mean_color.tolist(),
                'dominant_colors': dominant_colors,
                'brightness': float(np.mean(img_array)),
                'contrast': float(np.std(img_array))
            }
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {}
    
    def _get_dominant_colors(self, img_array: np.ndarray, n_colors: int = 5) -> List[Dict]:
        """Get dominant colors using K-means clustering"""
        try:
            # Reshape image to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels for performance
            if len(pixels) > 10000:
                pixels = pixels[::len(pixels)//10000]
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = []
            for i, color in enumerate(kmeans.cluster_centers_):
                colors.append({
                    'rgb': color.astype(int).tolist(),
                    'percentage': float(np.sum(kmeans.labels_ == i) / len(kmeans.labels_))
                })
            
            return sorted(colors, key=lambda x: x['percentage'], reverse=True)
        except Exception:
            return []
    
    def _extract_exif_data(self, img) -> Dict[str, Any]:
        """Extract EXIF metadata from image"""
        try:
            import PIL.ExifTags
            
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)  # Convert to string for JSON serialization
            
            return exif_data
        except Exception:
            return {}
    
    def _analyze_audio_file(self, file_path: Path) -> AnalysisResult:
        """Analyze audio file content"""
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=None)
            
            # Extract audio features
            features = self._extract_audio_features(y, sr)
            
            # Audio classification (simplified)
            audio_type = self._classify_audio_content(features)
            
            analysis = {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'channels': 1,  # librosa loads as mono by default
                'features': features,
                'audio_type': audio_type
            }
            
            return AnalysisResult(
                analysis_type="audio_analysis",
                confidence=0.8,
                primary_result=audio_type,
                detailed_results=analysis
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type="audio_analysis",
                confidence=0.0,
                primary_result="error",
                detailed_results={'error': str(e)}
            )
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features using librosa"""
        features = {}
        
        try:
            # Spectral features
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # RMS energy
            features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
        
        return features
    
    def _classify_audio_content(self, features: Dict[str, float]) -> str:
        """Classify audio content type based on features"""
        # Simplified audio classification based on heuristics
        tempo = features.get('tempo', 0)
        spectral_centroid = features.get('spectral_centroid', 0)
        
        if tempo > 120 and spectral_centroid > 2000:
            return 'music_upbeat'
        elif tempo < 80 and spectral_centroid < 1500:
            return 'speech_or_slow_music'
        elif spectral_centroid < 1000:
            return 'speech'
        else:
            return 'music_general'
    
    def _analyze_binary_file(self, file_path: Path) -> AnalysisResult:
        """Analyze binary file content"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(8192)  # Sample first 8KB
            
            analysis = {
                'file_size': file_path.stat().st_size,
                'entropy': self._calculate_binary_entropy(data),
                'byte_distribution': self._analyze_byte_distribution(data),
                'patterns': self._find_binary_patterns(data),
                'structure': self._analyze_binary_structure(data)
            }
            
            # Determine binary type
            binary_type = self._classify_binary_type(analysis)
            
            return AnalysisResult(
                analysis_type="binary_analysis",
                confidence=0.7,
                primary_result=binary_type,
                detailed_results=analysis
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type="binary_analysis",
                confidence=0.0,
                primary_result="error",
                detailed_results={'error': str(e)}
            )
    
    def _calculate_binary_entropy(self, data: bytes) -> float:
        """Calculate entropy of binary data"""
        if not data:
            return 0.0
        
        byte_counts = np.bincount(data, minlength=256)
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]
        
        return float(-np.sum(probabilities * np.log2(probabilities)))
    
    def _analyze_byte_distribution(self, data: bytes) -> Dict[str, float]:
        """Analyze byte value distribution"""
        byte_counts = np.bincount(data, minlength=256)
        
        return {
            'null_bytes_ratio': byte_counts[0] / len(data),
            'printable_ratio': sum(byte_counts[32:127]) / len(data),
            'high_bytes_ratio': sum(byte_counts[128:]) / len(data),
            'unique_bytes': len(np.nonzero(byte_counts)[0]),
            'most_common_byte': int(np.argmax(byte_counts)),
            'most_common_frequency': float(np.max(byte_counts) / len(data))
        }
    
    def _find_binary_patterns(self, data: bytes) -> Dict[str, int]:
        """Find common binary patterns"""
        patterns = {
            'repeated_nulls': data.count(b'\\x00\\x00\\x00\\x00'),
            'repeated_ff': data.count(b'\\xff\\xff\\xff\\xff'),
            'ascending_sequence': 0,  # Would implement sequence detection
            'magic_numbers': 0  # Would check for known magic numbers
        }
        
        return patterns
    
    def _analyze_binary_structure(self, data: bytes) -> Dict[str, Any]:
        """Analyze binary file structure"""
        return {
            'has_header_structure': len(data) > 64 and len(set(data[:64])) > 10,
            'appears_compressed': self._calculate_binary_entropy(data) > 7.0,
            'has_repeating_blocks': False,  # Would implement block detection
            'alignment_patterns': self._check_alignment_patterns(data)
        }
    
    def _check_alignment_patterns(self, data: bytes) -> Dict[str, bool]:
        """Check for common alignment patterns"""
        return {
            'word_aligned': len([i for i in range(0, len(data), 4) if data[i:i+4] == b'\\x00\\x00\\x00\\x00']) > 10,
            'sector_aligned': len(data) % 512 == 0,
            'page_aligned': len(data) % 4096 == 0
        }
    
    def _classify_binary_type(self, analysis: Dict[str, Any]) -> str:
        """Classify binary file type based on analysis"""
        entropy = analysis['entropy']
        byte_dist = analysis['byte_distribution']
        structure = analysis['structure']
        
        if entropy > 7.5:
            return 'compressed_or_encrypted'
        elif byte_dist['null_bytes_ratio'] > 0.5:
            return 'sparse_data'
        elif byte_dist['printable_ratio'] > 0.7:
            return 'text_like_binary'
        elif structure['has_header_structure']:
            return 'structured_binary'
        else:
            return 'unknown_binary'
    
    def save_model(self, file_path: str) -> bool:
        """Save content analyzer models"""
        # Content analyzer primarily uses pre-trained models
        # Save configuration and any fine-tuned models
        try:
            config = {
                'model_name': self.model_name,
                'text_models_available': list(self.text_models.keys()),
                'image_models_available': list(self.image_models.keys()),
                'audio_models_available': list(self.audio_models.keys())
            }
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save content analyzer config: {e}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """Load content analyzer configuration"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Re-initialize models based on config
            self._initialize_models()
            return True
        except Exception as e:
            logger.error(f"Failed to load content analyzer config: {e}")
            return False


class AIIntelligenceEngine:
    """Main AI Intelligence Engine coordinating all AI models"""
    
    def __init__(self, models_directory: str = "models"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        # Initialize AI models
        self.models = {
            'file_type_classifier': FileTypeClassifier(),
            'malware_detector': MalwareDetector(),
            'content_analyzer': ContentAnalyzer()
        }
        
        # Model performance tracking
        self.performance_metrics = {}
        self.usage_statistics = defaultdict(int)
        
        logger.info("AI Intelligence Engine initialized")
    
    def get_model(self, model_name: str) -> Optional[BaseAIModel]:
        """Get a specific AI model"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List available AI models"""
        return list(self.models.keys())
    
    def analyze_file(self, file_path: str, analysis_types: List[str] = None) -> Dict[str, AnalysisResult]:
        """Perform comprehensive file analysis using multiple AI models"""
        if analysis_types is None:
            analysis_types = ['file_type', 'malware', 'content']
        
        results = {}
        
        # File type classification
        if 'file_type' in analysis_types and 'file_type_classifier' in self.models:
            try:
                if self.models['file_type_classifier'].is_trained:
                    results['file_type'] = self.models['file_type_classifier'].predict(file_path)
                    self.usage_statistics['file_type_classifier'] += 1
            except Exception as e:
                logger.error(f"File type classification failed: {e}")
        
        # Malware detection
        if 'malware' in analysis_types and 'malware_detector' in self.models:
            try:
                if self.models['malware_detector'].is_trained:
                    results['malware'] = self.models['malware_detector'].predict(file_path)
                    self.usage_statistics['malware_detector'] += 1
            except Exception as e:
                logger.error(f"Malware detection failed: {e}")
        
        # Content analysis
        if 'content' in analysis_types and 'content_analyzer' in self.models:
            try:
                results['content'] = self.models['content_analyzer'].predict(file_path)
                self.usage_statistics['content_analyzer'] += 1
            except Exception as e:
                logger.error(f"Content analysis failed: {e}")
        
        return results
    
    def train_model(self, model_name: str, training_data: Any, labels: Any = None, **kwargs) -> Dict[str, float]:
        """Train a specific AI model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        try:
            if labels is not None:
                metrics = model.train(training_data, labels, **kwargs)
            else:
                metrics = model.train(training_data, **kwargs)
            
            # Save performance metrics
            self.performance_metrics[model_name] = {
                'last_training': datetime.utcnow(),
                'metrics': metrics
            }
            
            # Auto-save trained model
            model_path = self.models_directory / f"{model_name}.joblib"
            model.save_model(str(model_path))
            
            logger.info(f"Model {model_name} trained successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for model {model_name}: {e}")
            raise
    
    def load_trained_models(self):
        """Load all trained models from disk"""
        for model_name, model in self.models.items():
            model_path = self.models_directory / f"{model_name}.joblib"
            
            if model_path.exists():
                try:
                    if model.load_model(str(model_path)):
                        logger.info(f"Loaded trained model: {model_name}")
                    else:
                        logger.warning(f"Failed to load model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
    
    def save_all_models(self):
        """Save all trained models to disk"""
        for model_name, model in self.models.items():
            if model.is_trained:
                model_path = self.models_directory / f"{model_name}.joblib"
                try:
                    if model.save_model(str(model_path)):
                        logger.info(f"Saved model: {model_name}")
                    else:
                        logger.warning(f"Failed to save model: {model_name}")
                except Exception as e:
                    logger.error(f"Error saving model {model_name}: {e}")
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        
        for model_name, model in self.models.items():
            status[model_name] = {
                'is_trained': model.is_trained,
                'usage_count': self.usage_statistics[model_name],
                'last_used': 'N/A',  # Would track this
                'performance_metrics': self.performance_metrics.get(model_name, {}),
                'model_file_exists': (self.models_directory / f"{model_name}.joblib").exists()
            }
        
        return status
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        return {
            'total_models': len(self.models),
            'trained_models': sum(1 for model in self.models.values() if model.is_trained),
            'total_predictions': sum(self.usage_statistics.values()),
            'models_directory': str(self.models_directory),
            'usage_by_model': dict(self.usage_statistics),
            'performance_summary': self.performance_metrics
        }
    
    def reset_statistics(self):
        """Reset usage statistics"""
        self.usage_statistics.clear()
        logger.info("AI engine statistics reset")


# Example usage and testing
def example_usage():
    """Example usage of the AI Intelligence Engine"""
    # Initialize the engine
    ai_engine = AIIntelligenceEngine()
    
    # Load any pre-trained models
    ai_engine.load_trained_models()
    
    # Example: Analyze a file
    try:
        results = ai_engine.analyze_file("/path/to/sample/file.exe")
        
        for analysis_type, result in results.items():
            print(f"{analysis_type.upper()} Analysis:")
            print(f"  Result: {result.primary_result}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Details: {result.detailed_results}")
            print()
    
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Get system status
    status = ai_engine.get_model_status()
    print("Model Status:")
    for model_name, model_status in status.items():
        print(f"  {model_name}: {'Trained' if model_status['is_trained'] else 'Not trained'}")
    
    # Get statistics
    stats = ai_engine.get_system_statistics()
    print(f"\\nSystem Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Trained models: {stats['trained_models']}")
    print(f"  Total predictions: {stats['total_predictions']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()