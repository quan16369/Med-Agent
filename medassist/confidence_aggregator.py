"""
Confidence Aggregation Module
Sophisticated methods for aggregating confidence across multiple agents
"""

from typing import List, Tuple, Dict
import numpy as np


class ConfidenceAggregator:
    """
    Sophisticated confidence aggregation across agents
    Uses weighted averaging and Bayesian methods
    """
    
    def __init__(self):
        # Default weights for different agent types
        self.default_weights = {
            'diagnostic': 0.35,
            'treatment': 0.30,
            'history': 0.20,
            'knowledge': 0.15
        }
    
    def aggregate(
        self,
        agent_confidences: List[Tuple[str, float]],
        method: str = 'weighted_average'
    ) -> Dict:
        """
        Aggregate confidence scores
        
        Args:
            agent_confidences: List of (agent_name, confidence) tuples
            method: Aggregation method ('weighted_average', 'harmonic_mean', 'bayesian')
            
        Returns:
            Dictionary with aggregate confidence and metadata
        """
        if not agent_confidences:
            return {
                'aggregate': 0.5,
                'method': 'default',
                'confidence_interval': (0.4, 0.6)
            }
        
        if method == 'weighted_average':
            return self._weighted_average(agent_confidences)
        elif method == 'harmonic_mean':
            return self._harmonic_mean(agent_confidences)
        elif method == 'bayesian':
            return self._bayesian_aggregation(agent_confidences)
        else:
            return self._weighted_average(agent_confidences)
    
    def _weighted_average(self, agent_confidences: List[Tuple[str, float]]) -> Dict:
        """Weighted average aggregation"""
        weighted_sum = 0
        total_weight = 0
        
        for agent_name, confidence in agent_confidences:
            weight = self.default_weights.get(agent_name, 0.25)
            weighted_sum += confidence * weight
            total_weight += weight
        
        aggregate = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate confidence interval (simple approach)
        confidences = [c for _, c in agent_confidences]
        std_dev = np.std(confidences) if len(confidences) > 1 else 0.1
        
        return {
            'aggregate': round(aggregate, 3),
            'method': 'weighted_average',
            'weights_used': self.default_weights,
            'confidence_interval': (
                round(max(aggregate - std_dev, 0), 3),
                round(min(aggregate + std_dev, 1), 3)
            ),
            'agreement_score': self._calculate_agreement(confidences)
        }
    
    def _harmonic_mean(self, agent_confidences: List[Tuple[str, float]]) -> Dict:
        """Harmonic mean - more conservative, penalizes low confidence"""
        confidences = [c for _, c in agent_confidences]
        
        # Avoid division by zero
        safe_confidences = [max(c, 0.01) for c in confidences]
        
        harmonic = len(safe_confidences) / sum(1/c for c in safe_confidences)
        
        return {
            'aggregate': round(harmonic, 3),
            'method': 'harmonic_mean',
            'confidence_interval': (
                round(harmonic * 0.9, 3),
                round(harmonic * 1.1, 3)
            )
        }
    
    def _bayesian_aggregation(self, agent_confidences: List[Tuple[str, float]]) -> Dict:
        """Bayesian aggregation with prior"""
        # Start with prior (neutral)
        prior = 0.5
        
        # Update with each agent's confidence (simplified Bayesian)
        posterior = prior
        for agent_name, confidence in agent_confidences:
            weight = self.default_weights.get(agent_name, 0.25)
            # Weighted Bayesian update
            posterior = (posterior * (1 - weight)) + (confidence * weight)
        
        return {
            'aggregate': round(posterior, 3),
            'method': 'bayesian',
            'prior': prior,
            'confidence_interval': (
                round(posterior * 0.85, 3),
                round(posterior * 1.15 if posterior < 0.87 else 1.0, 3)
            )
        }
    
    def _calculate_agreement(self, confidences: List[float]) -> float:
        """Calculate agreement score (inverse of variance)"""
        if len(confidences) < 2:
            return 1.0
        
        variance = np.var(confidences)
        # Convert to 0-1 scale (low variance = high agreement)
        agreement = 1.0 / (1.0 + variance * 10)
        
        return round(agreement, 3)
    
    def set_custom_weights(self, weights: Dict[str, float]):
        """Set custom agent weights"""
        self.default_weights.update(weights)
