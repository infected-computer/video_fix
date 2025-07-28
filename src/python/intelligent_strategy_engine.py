"""
Intelligent Strategy Engine for Video Repair
מנוע אסטרטגיות חכם לתיקון וידאו

This module implements intelligent decision-making for video repair strategies,
combining machine learning, heuristics, and expert knowledge to select the
optimal repair approach for each specific corruption scenario.
"""

import logging
import json
import pickle
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import statistics

# Machine learning imports (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class DecisionFactors(Enum):
    """Factors that influence repair strategy decisions"""
    FILE_SIZE = "file_size"
    CORRUPTION_SEVERITY = "corruption_severity"
    CORRUPTION_PATTERN = "corruption_pattern"
    CONTAINER_TYPE = "container_type"
    CODEC_TYPE = "codec_type"
    DURATION = "duration"
    BITRATE = "bitrate"
    RESOLUTION = "resolution"
    AUDIO_PRESENCE = "audio_presence"
    METADATA_INTEGRITY = "metadata_integrity"
    TOOL_AVAILABILITY = "tool_availability"
    PREVIOUS_SUCCESS_RATE = "previous_success_rate"
    TIME_CONSTRAINTS = "time_constraints"
    QUALITY_REQUIREMENTS = "quality_requirements"


@dataclass
class RepairScenario:
    """Complete description of a repair scenario"""
    # File characteristics
    file_path: str = ""
    file_size_mb: float = 0.0
    container_format: str = ""
    video_codec: str = ""
    audio_codec: str = ""
    duration_seconds: float = 0.0
    bitrate_mbps: float = 0.0
    width: int = 0
    height: int = 0
    
    # Corruption characteristics  
    corruption_severity: float = 0.0  # 0.0-1.0
    corruption_patterns: List[str] = field(default_factory=list)
    header_corruption: bool = False
    metadata_corruption: bool = False
    stream_corruption: bool = False
    fragment_corruption: bool = False
    
    # Technical constraints
    available_tools: List[str] = field(default_factory=list)
    reference_file_available: bool = False
    max_processing_time: int = 3600
    quality_requirements: str = "balanced"  # fast, balanced, high_quality
    
    # Context
    scenario_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert scenario to numerical feature vector for ML"""
        features = [
            self.file_size_mb,
            self.duration_seconds,
            self.bitrate_mbps,
            self.width,
            self.height,
            self.corruption_severity,
            1.0 if self.header_corruption else 0.0,
            1.0 if self.metadata_corruption else 0.0,
            1.0 if self.stream_corruption else 0.0,
            1.0 if self.fragment_corruption else 0.0,
            len(self.available_tools),
            1.0 if self.reference_file_available else 0.0,
            self.max_processing_time / 3600.0,  # Normalize to hours
            
            # Container format encoding (one-hot style)
            1.0 if self.container_format.lower() == "mp4" else 0.0,
            1.0 if self.container_format.lower() == "mov" else 0.0,
            1.0 if self.container_format.lower() == "avi" else 0.0,
            1.0 if self.container_format.lower() == "mkv" else 0.0,
            
            # Codec encoding
            1.0 if "h264" in self.video_codec.lower() else 0.0,
            1.0 if "h265" in self.video_codec.lower() else 0.0,
            1.0 if "prores" in self.video_codec.lower() else 0.0,
            
            # Quality requirements
            1.0 if self.quality_requirements == "fast" else 0.0,
            1.0 if self.quality_requirements == "balanced" else 0.0,
            1.0 if self.quality_requirements == "high_quality" else 0.0,
        ]
        
        return np.array(features, dtype=np.float32)


@dataclass
class StrategyRecommendation:
    """Recommendation for repair strategy"""
    primary_tool: str
    fallback_tools: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence and timing
    confidence_score: float = 0.0
    estimated_success_probability: float = 0.0
    estimated_processing_time: int = 0
    estimated_quality_score: float = 0.0
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    recommendation_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RepairOutcome:
    """Record of actual repair outcome for learning"""
    scenario_id: str
    recommendation_id: str
    
    # Results
    success: bool = False
    actual_processing_time: int = 0
    actual_quality_score: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    techniques_used: List[str] = field(default_factory=list)
    
    # Performance metrics
    file_size_change: float = 0.0
    metadata_preserved: bool = False
    audio_quality: float = 0.0
    video_quality: float = 0.0
    
    # User feedback
    user_satisfaction: float = 0.0  # 0.0-1.0
    user_comments: str = ""
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)


class HistoricalDataManager:
    """Manages historical repair data for learning and optimization"""
    
    def __init__(self, data_dir: str = "repair_history"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.scenarios_file = self.data_dir / "scenarios.json"
        self.outcomes_file = self.data_dir / "outcomes.json"
        self.statistics_file = self.data_dir / "statistics.json"
        
        self.scenarios: Dict[str, RepairScenario] = {}
        self.outcomes: Dict[str, RepairOutcome] = {}
        self.statistics: Dict[str, Any] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load historical data from disk"""
        try:
            if self.scenarios_file.exists():
                with open(self.scenarios_file, 'r') as f:
                    data = json.load(f)
                    self.scenarios = {
                        k: RepairScenario(**v) for k, v in data.items()
                    }
            
            if self.outcomes_file.exists():
                with open(self.outcomes_file, 'r') as f:
                    data = json.load(f)
                    self.outcomes = {
                        k: RepairOutcome(**v) for k, v in data.items()
                    }
            
            if self.statistics_file.exists():
                with open(self.statistics_file, 'r') as f:
                    self.statistics = json.load(f)
            
            logger.info(f"Loaded {len(self.scenarios)} scenarios and {len(self.outcomes)} outcomes")
            
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
    
    def save_scenario(self, scenario: RepairScenario):
        """Save a repair scenario"""
        self.scenarios[scenario.scenario_id] = scenario
        self._save_scenarios()
    
    def save_outcome(self, outcome: RepairOutcome):
        """Save a repair outcome"""
        self.outcomes[outcome.scenario_id] = outcome
        self._save_outcomes()
        self._update_statistics()
    
    def _save_scenarios(self):
        """Save scenarios to disk"""
        try:
            data = {k: asdict(v) for k, v in self.scenarios.items()}
            with open(self.scenarios_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save scenarios: {e}")
    
    def _save_outcomes(self):
        """Save outcomes to disk"""
        try:
            data = {k: asdict(v) for k, v in self.outcomes.items()}
            with open(self.outcomes_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save outcomes: {e}")
    
    def _update_statistics(self):
        """Update repair statistics"""
        if not self.outcomes:
            return
        
        outcomes = list(self.outcomes.values())
        
        # Overall statistics
        total_repairs = len(outcomes)
        successful_repairs = sum(1 for o in outcomes if o.success)
        success_rate = successful_repairs / total_repairs if total_repairs > 0 else 0.0
        
        # Tool performance
        tool_stats = {}
        for outcome in outcomes:
            for tool in outcome.tools_used:
                if tool not in tool_stats:
                    tool_stats[tool] = {"total": 0, "successful": 0}
                tool_stats[tool]["total"] += 1
                if outcome.success:
                    tool_stats[tool]["successful"] += 1
        
        # Add success rates
        for tool, stats in tool_stats.items():
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Quality statistics
        quality_scores = [o.actual_quality_score for o in outcomes if o.actual_quality_score > 0]
        
        self.statistics = {
            "total_repairs": total_repairs,
            "success_rate": success_rate,
            "tool_performance": tool_stats,
            "average_quality": statistics.mean(quality_scores) if quality_scores else 0.0,
            "median_quality": statistics.median(quality_scores) if quality_scores else 0.0,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Save statistics
        try:
            with open(self.statistics_file, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def get_similar_scenarios(self, scenario: RepairScenario, limit: int = 10) -> List[Tuple[RepairScenario, RepairOutcome, float]]:
        """Find similar scenarios from history"""
        if not self.scenarios:
            return []
        
        target_vector = scenario.to_feature_vector()
        similarities = []
        
        for scenario_id, historical_scenario in self.scenarios.items():
            if scenario_id not in self.outcomes:
                continue
            
            historical_vector = historical_scenario.to_feature_vector()
            
            # Calculate cosine similarity
            dot_product = np.dot(target_vector, historical_vector)
            norms = np.linalg.norm(target_vector) * np.linalg.norm(historical_vector)
            similarity = dot_product / norms if norms > 0 else 0.0
            
            outcome = self.outcomes[scenario_id]
            similarities.append((historical_scenario, outcome, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:limit]
    
    def get_tool_success_rate(self, tool: str, scenario: RepairScenario = None) -> float:
        """Get success rate for a specific tool, optionally filtered by scenario similarity"""
        if tool not in self.statistics.get("tool_performance", {}):
            return 0.5  # Default unknown success rate
        
        if scenario is None:
            return self.statistics["tool_performance"][tool]["success_rate"]
        
        # Get success rate for similar scenarios
        similar_scenarios = self.get_similar_scenarios(scenario, limit=20)
        
        relevant_outcomes = []
        for _, outcome, similarity in similar_scenarios:
            if tool in outcome.tools_used and similarity > 0.7:  # High similarity threshold
                relevant_outcomes.append(outcome.success)
        
        if relevant_outcomes:
            return sum(relevant_outcomes) / len(relevant_outcomes)
        else:
            # Fall back to overall success rate
            return self.statistics["tool_performance"][tool]["success_rate"]


class RuleBasedExpert:
    """Expert system with hand-crafted rules for video repair"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize expert rules"""
        return [
            {
                "name": "small_mp4_header_corruption",
                "conditions": {
                    "container_format": "mp4",
                    "file_size_mb": {"max": 100},
                    "header_corruption": True,
                    "corruption_severity": {"max": 0.3}
                },
                "recommendation": {
                    "primary_tool": "untrunc",
                    "fallback_tools": ["ffmpeg"],
                    "confidence": 0.9,
                    "reasoning": "Small MP4 with header corruption responds well to untrunc"
                }
            },
            {
                "name": "large_file_time_constraint",
                "conditions": {
                    "file_size_mb": {"min": 1000},
                    "max_processing_time": {"max": 1800}  # 30 minutes
                },
                "recommendation": {
                    "primary_tool": "ffmpeg",
                    "techniques": ["container_remux"],
                    "parameters": {"fast_mode": True},
                    "confidence": 0.8,
                    "reasoning": "Large files with time constraints need fast processing"
                }
            },
            {
                "name": "professional_format_preservation",
                "conditions": {
                    "video_codec": {"contains": ["prores", "raw"]},
                    "quality_requirements": "high_quality"
                },
                "recommendation": {
                    "primary_tool": "untrunc",
                    "fallback_tools": ["mp4recover"],
                    "parameters": {"preserve_quality": True},
                    "confidence": 0.85,
                    "reasoning": "Professional formats need quality-preserving tools"
                }
            },
            {
                "name": "severe_corruption_multiple_tools",
                "conditions": {
                    "corruption_severity": {"min": 0.7}
                },
                "recommendation": {
                    "primary_tool": "mp4recover",
                    "fallback_tools": ["untrunc", "ffmpeg"],
                    "techniques": ["fragment_recovery", "deep_scan"],
                    "confidence": 0.6,
                    "reasoning": "Severe corruption requires multiple recovery approaches"
                }
            },
            {
                "name": "fragmented_corruption",
                "conditions": {
                    "fragment_corruption": True,
                    "corruption_patterns": {"contains": "fragmented"}
                },
                "recommendation": {
                    "primary_tool": "mp4recover",
                    "fallback_tools": ["ffmpeg"],
                    "techniques": ["fragment_recovery"],
                    "confidence": 0.8,
                    "reasoning": "Fragmented corruption needs specialized recovery"
                }
            },
            {
                "name": "reference_file_available",
                "conditions": {
                    "reference_file_available": True,
                    "container_format": {"in": ["mp4", "mov"]}
                },
                "recommendation": {
                    "primary_tool": "untrunc",
                    "parameters": {"use_reference": True},
                    "confidence": 0.95,
                    "reasoning": "Reference file greatly improves untrunc success"
                }
            }
        ]
    
    def evaluate_scenario(self, scenario: RepairScenario) -> List[StrategyRecommendation]:
        """Evaluate scenario against expert rules"""
        recommendations = []
        
        for rule in self.rules:
            if self._rule_matches(rule["conditions"], scenario):
                recommendation = self._create_recommendation_from_rule(rule, scenario)
                recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence_score, reverse=True)
        return recommendations
    
    def _rule_matches(self, conditions: Dict[str, Any], scenario: RepairScenario) -> bool:
        """Check if scenario matches rule conditions"""
        for field, condition in conditions.items():
            scenario_value = getattr(scenario, field, None)
            
            if scenario_value is None:
                continue
            
            if isinstance(condition, dict):
                # Range or list conditions
                if "min" in condition and scenario_value < condition["min"]:
                    return False
                if "max" in condition and scenario_value > condition["max"]:
                    return False
                if "in" in condition and scenario_value not in condition["in"]:
                    return False
                if "contains" in condition:
                    if isinstance(scenario_value, list):
                        if not any(item in scenario_value for item in condition["contains"]):
                            return False
                    else:
                        if not any(item in str(scenario_value).lower() for item in condition["contains"]):
                            return False
            elif isinstance(condition, bool):
                if scenario_value != condition:
                    return False
            elif isinstance(condition, str):
                if str(scenario_value).lower() != condition.lower():
                    return False
        
        return True
    
    def _create_recommendation_from_rule(self, rule: Dict[str, Any], scenario: RepairScenario) -> StrategyRecommendation:
        """Create recommendation from matched rule"""
        rec_data = rule["recommendation"]
        
        return StrategyRecommendation(
            primary_tool=rec_data.get("primary_tool", "ffmpeg"),
            fallback_tools=rec_data.get("fallback_tools", []),
            techniques=rec_data.get("techniques", []),
            parameters=rec_data.get("parameters", {}),
            confidence_score=rec_data.get("confidence", 0.5),
            reasoning=[rec_data.get("reasoning", "Expert rule match")],
            recommendation_id=f"rule_{rule['name']}_{scenario.scenario_id}"
        )


class MachineLearningPredictor:
    """Machine learning component for strategy prediction"""
    
    def __init__(self, model_path: str = "repair_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder = {}
        self.feature_names = []
        
        if SKLEARN_AVAILABLE:
            self._load_or_create_model()
        else:
            logger.warning("Scikit-learn not available, ML prediction disabled")
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.label_encoder = model_data['label_encoder']
                    self.feature_names = model_data['feature_names']
                logger.info("Loaded trained ML model")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create new untrained model"""
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            logger.info("Created new ML model")
    
    def train(self, scenarios: List[RepairScenario], outcomes: List[RepairOutcome]) -> bool:
        """Train model on historical data"""
        if not SKLEARN_AVAILABLE or not scenarios or not outcomes:
            return False
        
        try:
            # Prepare features and labels
            X = np.array([scenario.to_feature_vector() for scenario in scenarios])
            
            # Create labels based on primary successful tool
            y = []
            for outcome in outcomes:
                if outcome.success and outcome.tools_used:
                    y.append(outcome.tools_used[0])  # Primary tool
                else:
                    y.append("failed")
            
            # Encode labels
            unique_labels = list(set(y))
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
            y_encoded = [self.label_encoder[label] for label in y]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            
            # Save model
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def predict(self, scenario: RepairScenario) -> Optional[StrategyRecommendation]:
        """Predict best strategy for scenario"""
        if not SKLEARN_AVAILABLE or self.model is None:
            return None
        
        try:
            X = scenario.to_feature_vector().reshape(1, -1)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Decode prediction
            reverse_encoder = {v: k for k, v in self.label_encoder.items()}
            predicted_tool = reverse_encoder.get(prediction, "unknown")
            
            if predicted_tool == "failed":
                return None
            
            # Get confidence from probability
            confidence = max(probabilities)
            
            # Get feature importance for reasoning
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_features = np.argsort(importances)[-3:]  # Top 3 features
                reasoning = [f"ML model based on feature importance: {i}" for i in top_features]
            else:
                reasoning = ["ML model prediction"]
            
            return StrategyRecommendation(
                primary_tool=predicted_tool,
                confidence_score=confidence,
                reasoning=reasoning,
                recommendation_id=f"ml_{scenario.scenario_id}"
            )
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None
    
    def _save_model(self):
        """Save trained model"""
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class IntelligentStrategyEngine:
    """Main intelligent strategy engine combining multiple approaches"""
    
    def __init__(self, data_dir: str = "strategy_data"):
        self.data_manager = HistoricalDataManager(data_dir)
        self.rule_expert = RuleBasedExpert()
        self.ml_predictor = MachineLearningPredictor(
            str(Path(data_dir) / "repair_model.pkl")
        )
        
        # Configuration
        self.confidence_threshold = 0.7
        self.max_recommendations = 5
        
        # Initialize ML model if enough data available
        self._initialize_ml_model()
        
        logger.info("Intelligent Strategy Engine initialized")
    
    def _initialize_ml_model(self):
        """Initialize ML model with historical data"""
        scenarios = list(self.data_manager.scenarios.values())
        outcomes = [self.data_manager.outcomes.get(s.scenario_id) 
                   for s in scenarios]
        outcomes = [o for o in outcomes if o is not None]
        
        if len(scenarios) >= 10 and len(outcomes) >= 10:  # Minimum data threshold
            self.ml_predictor.train(scenarios, outcomes)
            logger.info("ML model initialized with historical data")
        else:
            logger.info("Insufficient data for ML model training")
    
    def recommend_strategy(self, scenario: RepairScenario) -> StrategyRecommendation:
        """Generate intelligent strategy recommendation"""
        # Generate unique scenario ID
        scenario.scenario_id = self._generate_scenario_id(scenario)
        
        # Save scenario for learning
        self.data_manager.save_scenario(scenario)
        
        # Get recommendations from different sources
        recommendations = []
        
        # 1. Rule-based expert system
        expert_recommendations = self.rule_expert.evaluate_scenario(scenario)
        recommendations.extend(expert_recommendations)
        
        # 2. Machine learning predictor
        ml_recommendation = self.ml_predictor.predict(scenario)
        if ml_recommendation:
            recommendations.append(ml_recommendation)
        
        # 3. Historical similarity analysis
        similarity_recommendation = self._get_similarity_recommendation(scenario)
        if similarity_recommendation:
            recommendations.append(similarity_recommendation)
        
        # 4. Ensemble combination
        final_recommendation = self._combine_recommendations(recommendations, scenario)
        
        # Add estimates
        final_recommendation = self._add_estimates(final_recommendation, scenario)
        
        logger.info(f"Generated strategy recommendation: {final_recommendation.primary_tool} "
                   f"(confidence: {final_recommendation.confidence_score:.2f})")
        
        return final_recommendation
    
    def _generate_scenario_id(self, scenario: RepairScenario) -> str:
        """Generate unique ID for scenario"""
        # Create hash based on key characteristics
        key_data = f"{scenario.file_path}_{scenario.file_size_mb}_{scenario.corruption_severity}_{scenario.timestamp}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]
    
    def _get_similarity_recommendation(self, scenario: RepairScenario) -> Optional[StrategyRecommendation]:
        """Get recommendation based on similar historical scenarios"""
        similar_scenarios = self.data_manager.get_similar_scenarios(scenario, limit=5)
        
        if not similar_scenarios:
            return None
        
        # Analyze successful outcomes from similar scenarios
        successful_tools = []
        confidences = []
        
        for historical_scenario, outcome, similarity in similar_scenarios:
            if outcome.success and similarity > 0.6:  # Reasonable similarity threshold
                if outcome.tools_used:
                    successful_tools.append(outcome.tools_used[0])
                    confidences.append(similarity * 0.8)  # Similarity-based confidence
        
        if not successful_tools:
            return None
        
        # Find most common successful tool
        tool_counts = {}
        for tool in successful_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        most_common_tool = max(tool_counts, key=tool_counts.get)
        confidence = statistics.mean(confidences) if confidences else 0.5
        
        return StrategyRecommendation(
            primary_tool=most_common_tool,
            confidence_score=confidence,
            reasoning=[f"Based on {len(similar_scenarios)} similar scenarios"],
            recommendation_id=f"similarity_{scenario.scenario_id}"
        )
    
    def _combine_recommendations(self, recommendations: List[StrategyRecommendation], 
                               scenario: RepairScenario) -> StrategyRecommendation:
        """Combine multiple recommendations into final strategy"""
        if not recommendations:
            # Default fallback recommendation
            return self._get_default_recommendation(scenario)
        
        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence_score, reverse=True)
        
        # Take highest confidence as base
        best_recommendation = recommendations[0]
        
        # Combine insights from other recommendations
        all_tools = [r.primary_tool for r in recommendations]
        tool_votes = {}
        confidence_weights = {}
        
        for rec in recommendations:
            tool = rec.primary_tool
            tool_votes[tool] = tool_votes.get(tool, 0) + 1
            confidence_weights[tool] = max(
                confidence_weights.get(tool, 0), 
                rec.confidence_score
            )
        
        # Select primary tool (highest weighted vote)
        weighted_scores = {
            tool: votes * confidence_weights[tool] 
            for tool, votes in tool_votes.items()
        }
        
        primary_tool = max(weighted_scores, key=weighted_scores.get)
        
        # Collect fallback tools
        fallback_tools = [tool for tool in all_tools 
                         if tool != primary_tool and tool not in [primary_tool]]
        fallback_tools = list(dict.fromkeys(fallback_tools))  # Remove duplicates, preserve order
        
        # Combine reasoning
        all_reasoning = []
        for rec in recommendations[:3]:  # Top 3 recommendations
            all_reasoning.extend(rec.reasoning)
        
        # Calculate ensemble confidence
        ensemble_confidence = weighted_scores[primary_tool] / len(recommendations)
        ensemble_confidence = min(ensemble_confidence, 1.0)
        
        return StrategyRecommendation(
            primary_tool=primary_tool,
            fallback_tools=fallback_tools[:3],  # Limit fallbacks
            confidence_score=ensemble_confidence,
            reasoning=all_reasoning[:5],  # Limit reasoning items
            recommendation_id=f"ensemble_{scenario.scenario_id}"
        )
    
    def _get_default_recommendation(self, scenario: RepairScenario) -> StrategyRecommendation:
        """Get default recommendation when no other options available"""
        # Simple heuristic-based default
        if scenario.corruption_severity < 0.3:
            primary_tool = "ffmpeg"
        elif scenario.container_format.lower() in ["mp4", "mov"]:
            primary_tool = "untrunc"
        else:
            primary_tool = "ffmpeg"
        
        return StrategyRecommendation(
            primary_tool=primary_tool,
            fallback_tools=["ffmpeg", "mp4recover"],
            confidence_score=0.5,
            reasoning=["Default heuristic recommendation"],
            recommendation_id=f"default_{scenario.scenario_id}"
        )
    
    def _add_estimates(self, recommendation: StrategyRecommendation, 
                      scenario: RepairScenario) -> StrategyRecommendation:
        """Add time and quality estimates to recommendation"""
        # Get historical performance for this tool
        success_rate = self.data_manager.get_tool_success_rate(
            recommendation.primary_tool, scenario
        )
        recommendation.estimated_success_probability = success_rate
        
        # Estimate processing time based on file size and tool
        base_time = scenario.file_size_mb * 2  # 2 seconds per MB base
        
        tool_multipliers = {
            "ffmpeg": 1.0,      # Fast
            "untrunc": 1.5,     # Medium
            "mp4recover": 2.0,  # Slower but thorough
        }
        
        multiplier = tool_multipliers.get(recommendation.primary_tool, 1.5)
        estimated_time = int(base_time * multiplier)
        
        # Adjust for corruption severity
        severity_multiplier = 1.0 + (scenario.corruption_severity * 2.0)
        estimated_time = int(estimated_time * severity_multiplier)
        
        recommendation.estimated_processing_time = min(estimated_time, scenario.max_processing_time)
        
        # Estimate quality score
        base_quality = 0.8 - (scenario.corruption_severity * 0.3)
        tool_quality_bonus = {
            "untrunc": 0.1,     # Good quality preservation
            "mp4recover": 0.05, # Decent quality
            "ffmpeg": 0.0       # Standard quality
        }
        
        quality_bonus = tool_quality_bonus.get(recommendation.primary_tool, 0.0)
        recommendation.estimated_quality_score = min(base_quality + quality_bonus, 1.0)
        
        return recommendation
    
    def record_outcome(self, scenario_id: str, recommendation_id: str, 
                      outcome: RepairOutcome):
        """Record repair outcome for learning"""
        outcome.scenario_id = scenario_id
        outcome.recommendation_id = recommendation_id
        
        self.data_manager.save_outcome(outcome)
        
        # Retrain ML model periodically
        total_outcomes = len(self.data_manager.outcomes)
        if total_outcomes % 50 == 0 and total_outcomes >= 10:  # Every 50 new outcomes
            logger.info("Retraining ML model with new data")
            self._initialize_ml_model()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy engine statistics"""
        stats = self.data_manager.statistics.copy()
        
        # Add engine-specific stats
        stats.update({
            "total_scenarios": len(self.data_manager.scenarios),
            "ml_model_available": self.ml_predictor.model is not None,
            "confidence_threshold": self.confidence_threshold,
            "rules_count": len(self.rule_expert.rules)
        })
        
        return stats
    
    def optimize_thresholds(self):
        """Optimize engine parameters based on historical performance"""
        # Analyze historical accuracy vs confidence
        outcomes = list(self.data_manager.outcomes.values())
        
        if len(outcomes) < 20:
            return
        
        # Find optimal confidence threshold
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        best_threshold = self.confidence_threshold
        best_accuracy = 0.0
        
        for threshold in thresholds:
            high_confidence_outcomes = [o for o in outcomes 
                                      if getattr(o, 'confidence_score', 0.7) >= threshold]
            
            if len(high_confidence_outcomes) >= 10:
                accuracy = sum(o.success for o in high_confidence_outcomes) / len(high_confidence_outcomes)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
        
        self.confidence_threshold = best_threshold
        logger.info(f"Optimized confidence threshold to {best_threshold:.2f} "
                   f"(accuracy: {best_accuracy:.2f})")


# Usage example and testing
async def example_intelligent_strategy():
    """Example usage of the intelligent strategy engine"""
    
    # Initialize engine
    engine = IntelligentStrategyEngine("example_strategy_data")
    
    # Create example scenario
    scenario = RepairScenario(
        file_path="corrupted_video.mp4",
        file_size_mb=150.0,
        container_format="mp4",
        video_codec="h264",
        audio_codec="aac",
        duration_seconds=300.0,
        bitrate_mbps=5.0,
        width=1920,
        height=1080,
        corruption_severity=0.4,
        corruption_patterns=["header_damage"],
        header_corruption=True,
        available_tools=["ffmpeg", "untrunc", "mp4recover"],
        max_processing_time=1800,
        quality_requirements="balanced"
    )
    
    # Get recommendation
    recommendation = engine.recommend_strategy(scenario)
    
    print("Strategy Recommendation:")
    print(f"  Primary tool: {recommendation.primary_tool}")
    print(f"  Fallback tools: {recommendation.fallback_tools}")
    print(f"  Confidence: {recommendation.confidence_score:.2f}")
    print(f"  Success probability: {recommendation.estimated_success_probability:.2f}")
    print(f"  Estimated time: {recommendation.estimated_processing_time}s")
    print(f"  Estimated quality: {recommendation.estimated_quality_score:.2f}")
    print(f"  Reasoning: {recommendation.reasoning}")
    
    # Simulate repair outcome
    outcome = RepairOutcome(
        scenario_id=scenario.scenario_id,
        recommendation_id=recommendation.recommendation_id,
        success=True,
        actual_processing_time=1200,
        actual_quality_score=0.85,
        tools_used=[recommendation.primary_tool],
        user_satisfaction=0.9
    )
    
    # Record outcome for learning
    engine.record_outcome(scenario.scenario_id, recommendation.recommendation_id, outcome)
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\nEngine Statistics:")
    print(f"  Total scenarios: {stats['total_scenarios']}")
    print(f"  Success rate: {stats.get('success_rate', 0):.2f}")
    print(f"  ML model available: {stats['ml_model_available']}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_intelligent_strategy())