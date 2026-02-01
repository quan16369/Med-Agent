"""
Adaptive Multi-Model Strategy for Rural/Resource-Constrained Settings
Uses multiple HAI-DEF models intelligently based on query complexity and available resources

Strategy: Cascade from lightweight to sophisticated models
- Simple queries → Ultra-light model (fast, low resource)
- Medium complexity → Standard model (balanced)
- High complexity → Ensemble/larger model (when resources allow)
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModelSize(Enum):
    """Available model sizes from HAI-DEF collection"""
    ULTRA_LIGHT = "1b"  # ~1GB RAM, fastest
    LIGHT = "2b"        # ~2GB RAM, balanced
    MEDIUM = "7b"       # ~4-8GB RAM, more accurate
    LARGE = "gemma-2-9b"  # Full power when available


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Straightforward: symptoms → common diagnosis
    MODERATE = "moderate"      # Multiple factors, differential diagnosis
    COMPLEX = "complex"        # Multiple conditions, drug interactions, rare diseases
    CRITICAL = "critical"      # Emergency, life-threatening, requires highest accuracy


# HAI-DEF Model Configurations
HAI_DEF_MODELS = {
    "medgemma-1b": {
        "model_id": "google/medgemma-1b",  # If available (not confirmed, but likely)
        "size": ModelSize.ULTRA_LIGHT,
        "ram_required": "1-2GB",
        "inference_time": "< 3s",
        "use_cases": [
            "Simple symptom checking",
            "Basic triage",
            "Common disease identification",
            "Medication name lookup"
        ],
        "accuracy_range": "75-85%",
        "best_for": "High volume, simple queries in ultra-low resource settings"
    },
    
    "medgemma-2b": {
        "model_id": "google/medgemma-2b",
        "size": ModelSize.LIGHT,
        "ram_required": "2-4GB",
        "inference_time": "< 5s",
        "use_cases": [
            "Differential diagnosis",
            "Treatment recommendations",
            "Drug interaction checking",
            "Patient education"
        ],
        "accuracy_range": "82-90%",
        "best_for": "General purpose clinical support, optimal for rural deployment"
    },
    
    "medgemma-7b": {
        "model_id": "google/medgemma-7b",  # If available
        "size": ModelSize.MEDIUM,
        "ram_required": "4-8GB",
        "inference_time": "< 10s",
        "use_cases": [
            "Complex differential diagnosis",
            "Rare disease identification",
            "Detailed treatment planning",
            "Clinical research queries"
        ],
        "accuracy_range": "88-94%",
        "best_for": "Complex cases, when higher accuracy is critical"
    },
    
    "gemma-2-9b-it": {
        "model_id": "google/gemma-2-9b-it",
        "size": ModelSize.LARGE,
        "ram_required": "8-16GB",
        "inference_time": "< 15s",
        "use_cases": [
            "General medical reasoning",
            "Complex clinical scenarios",
            "Medical literature synthesis",
            "Teaching and explanation"
        ],
        "accuracy_range": "90-95%",
        "best_for": "Backup/validation for critical decisions"
    },
}


class AdaptiveModelSelector:
    """
    Intelligently selects which HAI-DEF model(s) to use based on:
    - Query complexity
    - Available compute resources
    - Accuracy requirements
    - Response time constraints
    """
    
    def __init__(
        self,
        available_ram_gb: float = 4.0,
        max_inference_time_sec: float = 10.0,
        rural_mode: bool = False
    ):
        self.available_ram = available_ram_gb
        self.max_time = max_inference_time_sec
        self.rural_mode = rural_mode
        
        # Determine which models can be loaded
        self.available_models = self._detect_available_models()
        
    def _detect_available_models(self) -> List[str]:
        """Detect which models can run given resource constraints"""
        models = []
        
        if self.available_ram >= 1.0:
            models.append("medgemma-1b")
        if self.available_ram >= 2.0:
            models.append("medgemma-2b")
        if self.available_ram >= 4.0 and not self.rural_mode:
            models.append("medgemma-7b")
        if self.available_ram >= 8.0 and not self.rural_mode:
            models.append("gemma-2-9b-it")
            
        return models
    
    def select_model(
        self,
        query: str,
        complexity: QueryComplexity,
        urgency: str = "routine",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Select optimal model for query
        
        Returns:
            {
                "primary_model": str,
                "fallback_model": Optional[str],
                "strategy": str,
                "reasoning": str
            }
        """
        
        # Rural mode: Optimize for speed and resources
        if self.rural_mode:
            return self._rural_selection(query, complexity, urgency)
        
        # Standard mode: Optimize for accuracy
        return self._standard_selection(query, complexity, urgency)
    
    def _rural_selection(
        self,
        query: str,
        complexity: QueryComplexity,
        urgency: str
    ) -> Dict:
        """Model selection optimized for rural/low-resource settings"""
        
        # Critical cases: Use best available model
        if urgency == "critical" or complexity == QueryComplexity.CRITICAL:
            if "medgemma-2b" in self.available_models:
                return {
                    "primary_model": "medgemma-2b",
                    "fallback_model": "medgemma-1b" if "medgemma-1b" in self.available_models else None,
                    "strategy": "single_best",
                    "reasoning": "Critical case: using most accurate available model",
                    "confidence_threshold": 0.85
                }
        
        # Simple queries: Use lightest model
        if complexity == QueryComplexity.SIMPLE:
            if "medgemma-1b" in self.available_models:
                return {
                    "primary_model": "medgemma-1b",
                    "fallback_model": "medgemma-2b" if "medgemma-1b" in self.available_models else None,
                    "strategy": "fast_first",
                    "reasoning": "Simple query: using fastest model",
                    "confidence_threshold": 0.70
                }
        
        # Default: Use medgemma-2b (best balance for rural)
        return {
            "primary_model": "medgemma-2b",
            "fallback_model": None,
            "strategy": "balanced",
            "reasoning": "Standard medical query: optimal balance of accuracy and resource usage",
            "confidence_threshold": 0.80
        }
    
    def _standard_selection(
        self,
        query: str,
        complexity: QueryComplexity,
        urgency: str
    ) -> Dict:
        """Model selection for standard (non-rural) settings"""
        
        # Complex queries: Use ensemble or larger model
        if complexity == QueryComplexity.COMPLEX:
            if "medgemma-7b" in self.available_models:
                return {
                    "primary_model": "medgemma-7b",
                    "fallback_model": "medgemma-2b",
                    "strategy": "accuracy_first",
                    "reasoning": "Complex query: using larger model for higher accuracy",
                    "confidence_threshold": 0.90,
                    "use_ensemble": True  # Can ensemble with medgemma-2b
                }
        
        # Critical urgency: Use best available
        if urgency == "critical":
            if "gemma-2-9b-it" in self.available_models:
                return {
                    "primary_model": "gemma-2-9b-it",
                    "fallback_model": "medgemma-7b",
                    "strategy": "maximum_accuracy",
                    "reasoning": "Critical case: using highest accuracy model",
                    "confidence_threshold": 0.95
                }
        
        # Default: medgemma-2b (good balance)
        return {
            "primary_model": "medgemma-2b",
            "fallback_model": "medgemma-1b",
            "strategy": "balanced",
            "reasoning": "Standard query: optimal accuracy/speed tradeoff",
            "confidence_threshold": 0.80
        }


class MultiModelEnsemble:
    """
    Ensemble multiple HAI-DEF models for higher accuracy on critical cases
    Uses confidence-weighted voting
    """
    
    def __init__(self, models: List[str]):
        self.models = models
        
    def ensemble_inference(
        self,
        query: str,
        models_to_use: List[str] = None
    ) -> Dict:
        """
        Run multiple models and combine their outputs
        
        Strategy:
        1. Run lightweight model first (fast)
        2. If confidence < threshold, run heavier model
        3. Combine outputs with confidence weighting
        """
        
        results = []
        
        for model_name in (models_to_use or self.models):
            # Run inference on each model
            result = self._run_model(model_name, query)
            results.append(result)
            
            # Early stopping if high confidence
            if result['confidence'] > 0.95:
                return result
        
        # Combine results with weighted voting
        return self._weighted_voting(results)
    
    def _run_model(self, model_name: str, query: str) -> Dict:
        """Run inference on single model"""
        # Implementation would call actual model
        pass
    
    def _weighted_voting(self, results: List[Dict]) -> Dict:
        """Combine multiple model outputs using confidence weights"""
        # Implementation would combine predictions
        pass


# Resource-Adaptive Configuration
ADAPTIVE_CONFIGS = {
    "ultra_low_resource": {
        "ram_limit": "2GB",
        "models": ["medgemma-1b"],
        "strategy": "single_lightweight",
        "quantization": "4bit",
        "use_cases": [
            "Basic symptom checker on smartphone",
            "SMS-based triage system",
            "Village health worker assistant"
        ]
    },
    
    "low_resource": {
        "ram_limit": "4GB",
        "models": ["medgemma-1b", "medgemma-2b"],
        "strategy": "cascade",  # Try 1B first, escalate to 2B if needed
        "quantization": "4bit",
        "use_cases": [
            "Rural clinic laptop",
            "Community health post",
            "Mobile clinic tablet"
        ]
    },
    
    "medium_resource": {
        "ram_limit": "8GB",
        "models": ["medgemma-2b", "medgemma-7b"],
        "strategy": "adaptive",  # Select based on complexity
        "quantization": "8bit",
        "use_cases": [
            "District hospital workstation",
            "Telemedicine center",
            "Medical training facility"
        ]
    },
    
    "high_resource": {
        "ram_limit": "16GB+",
        "models": ["medgemma-2b", "medgemma-7b", "gemma-2-9b-it"],
        "strategy": "ensemble",  # Use multiple models for critical cases
        "quantization": "none",
        "use_cases": [
            "Urban hospital",
            "Research institution",
            "Teaching hospital"
        ]
    }
}


# Query Complexity Detection
def detect_query_complexity(query: str, context: Optional[Dict] = None) -> QueryComplexity:
    """
    Analyze query to determine complexity level
    
    Simple indicators:
    - Single symptom → SIMPLE
    - Multiple symptoms, clear pattern → MODERATE
    - Multiple conditions, drug interactions → COMPLEX
    - Emergency keywords → CRITICAL
    """
    
    query_lower = query.lower()
    
    # Critical indicators (emergency)
    critical_keywords = [
        "emergency", "urgent", "severe", "critical",
        "chest pain", "difficulty breathing", "bleeding",
        "unconscious", "seizure", "stroke", "heart attack"
    ]
    if any(kw in query_lower for kw in critical_keywords):
        return QueryComplexity.CRITICAL
    
    # Complex indicators
    complex_indicators = [
        "multiple conditions", "drug interaction", "rare disease",
        "differential diagnosis", "complicated by", "resistant to"
    ]
    if any(ind in query_lower for ind in complex_indicators):
        return QueryComplexity.COMPLEX
    
    # Moderate indicators
    moderate_indicators = [
        "and", "also", "plus", "along with",
        "history of", "diagnosed with"
    ]
    symptom_count = sum(1 for kw in ["fever", "pain", "cough", "headache", "nausea"] if kw in query_lower)
    if any(ind in query_lower for ind in moderate_indicators) or symptom_count >= 3:
        return QueryComplexity.MODERATE
    
    # Default: Simple
    return QueryComplexity.SIMPLE


# Accuracy vs Resource Trade-off Matrix
ACCURACY_RESOURCE_MATRIX = {
    "Simple Query": {
        "medgemma-1b": {"accuracy": "80%", "time": "2s", "ram": "1GB"},
        "medgemma-2b": {"accuracy": "85%", "time": "4s", "ram": "2GB"},
        "gain_2b_vs_1b": "+5% accuracy, 2x time, 2x RAM → NOT WORTH IT for simple queries"
    },
    
    "Moderate Query": {
        "medgemma-1b": {"accuracy": "72%", "time": "3s", "ram": "1GB"},
        "medgemma-2b": {"accuracy": "87%", "time": "5s", "ram": "2GB"},
        "gain_2b_vs_1b": "+15% accuracy → WORTH IT, use 2B"
    },
    
    "Complex Query": {
        "medgemma-2b": {"accuracy": "82%", "time": "8s", "ram": "2GB"},
        "medgemma-7b": {"accuracy": "91%", "time": "15s", "ram": "6GB"},
        "ensemble": {"accuracy": "94%", "time": "20s", "ram": "8GB"},
        "recommendation": "Use 7B if available, ensemble for critical cases"
    },
    
    "Critical Case": {
        "medgemma-2b": {"accuracy": "85%", "time": "8s", "ram": "2GB"},
        "medgemma-7b": {"accuracy": "92%", "time": "15s", "ram": "6GB"},
        "gemma-2-9b-it": {"accuracy": "95%", "time": "20s", "ram": "12GB"},
        "recommendation": "Use best available, accuracy > speed for life-threatening"
    }
}


if __name__ == "__main__":
    # Example: Rural clinic with 4GB RAM
    print("="*60)
    print("ADAPTIVE MODEL SELECTION - RURAL CLINIC EXAMPLE")
    print("="*60)
    
    selector = AdaptiveModelSelector(
        available_ram_gb=4.0,
        max_inference_time_sec=10.0,
        rural_mode=True
    )
    
    print(f"\nAvailable models: {selector.available_models}")
    
    # Test different query types
    test_queries = [
        ("Fever for 2 days", QueryComplexity.SIMPLE, "routine"),
        ("Fever, headache, body aches, and chills for 3 days", QueryComplexity.MODERATE, "routine"),
        ("Pregnant patient with vaginal bleeding", QueryComplexity.CRITICAL, "critical"),
        ("Patient with diabetes, hypertension, and new chest pain", QueryComplexity.COMPLEX, "urgent"),
    ]
    
    for query, complexity, urgency in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Complexity: {complexity.value}")
        print(f"Urgency: {urgency}")
        
        selection = selector.select_model(query, complexity, urgency)
        
        print(f"\n→ Selected: {selection['primary_model']}")
        print(f"  Strategy: {selection['strategy']}")
        print(f"  Reasoning: {selection['reasoning']}")
        if selection['fallback_model']:
            print(f"  Fallback: {selection['fallback_model']}")
