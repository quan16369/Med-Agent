"""
Deep Confidence with Token-Level Confidence Tracking
Inspired by DeepConf paper - dynamically filters low-quality reasoning
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging


class TokenConfidenceTracker:
    """
    Track confidence at token level during generation
    Implements group confidence from DeepConf paper
    """
    
    def __init__(self, group_size: int = 16, threshold: float = 0.75):
        """
        Args:
            group_size: Number of tokens per confidence group
            threshold: Minimum acceptable group confidence
        """
        self.group_size = group_size
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Token-level tracking
        self.token_confidences: List[float] = []
        self.token_texts: List[str] = []
        self.group_confidences: List[float] = []
        
        # Statistics
        self.early_stops = 0
        self.total_generations = 0
    
    def add_token(self, token: str, logprob: float):
        """
        Add a generated token with its log probability
        
        Args:
            token: Generated token text
            logprob: Log probability from model
        """
        # Convert logprob to confidence (0-1 scale)
        confidence = self._logprob_to_confidence(logprob)
        
        self.token_confidences.append(confidence)
        self.token_texts.append(token)
    
    def calculate_group_confidence(self) -> float:
        """
        Calculate confidence for current token group
        Returns average confidence over last group_size tokens
        """
        if len(self.token_confidences) < self.group_size:
            # Not enough tokens yet
            return 1.0
        
        # Get last group_size tokens
        recent_confidences = self.token_confidences[-self.group_size:]
        
        # Calculate group confidence (average)
        group_conf = np.mean(recent_confidences)
        
        self.group_confidences.append(group_conf)
        
        return group_conf
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """
        Determine if generation should stop early due to low confidence
        
        Returns:
            (should_stop: bool, reason: str)
        """
        if len(self.token_confidences) < self.group_size:
            return False, "Not enough tokens"
        
        group_conf = self.calculate_group_confidence()
        
        if group_conf < self.threshold:
            reason = (
                f"Group confidence {group_conf:.3f} below threshold "
                f"{self.threshold:.3f} at token {len(self.token_confidences)}"
            )
            self.early_stops += 1
            return True, reason
        
        return False, f"Confidence acceptable: {group_conf:.3f}"
    
    def get_confidence_stats(self) -> Dict:
        """Get statistics about generation confidence"""
        if not self.token_confidences:
            return {}
        
        return {
            'mean_confidence': np.mean(self.token_confidences),
            'min_confidence': np.min(self.token_confidences),
            'max_confidence': np.max(self.token_confidences),
            'std_confidence': np.std(self.token_confidences),
            'total_tokens': len(self.token_confidences),
            'num_groups': len(self.group_confidences),
            'group_confidences': self.group_confidences,
            'confidence_trend': self._analyze_trend()
        }
    
    def _logprob_to_confidence(self, logprob: float) -> float:
        """
        Convert log probability to confidence score (0-1)
        
        logprob typically ranges from -inf to 0
        -0.01 (very confident) → 0.99
        -2.0 (uncertain) → 0.13
        -5.0 (very uncertain) → 0.007
        """
        # Use exponential transformation
        prob = np.exp(logprob)
        
        # Clip to reasonable range
        return np.clip(prob, 0.0, 1.0)
    
    def _analyze_trend(self) -> str:
        """Analyze confidence trend over generation"""
        if len(self.group_confidences) < 2:
            return "insufficient_data"
        
        # Check if confidence is declining
        recent = self.group_confidences[-3:]
        if len(recent) >= 2:
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                return "declining"
            elif all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                return "improving"
        
        return "stable"
    
    def reset(self):
        """Reset tracker for new generation"""
        self.token_confidences = []
        self.token_texts = []
        self.group_confidences = []
        self.total_generations += 1


class ParallelThinkingFilter:
    """
    Filter low-quality reasoning traces from parallel generation
    Implements weighted confidence majority voting from DeepConf
    """
    
    def __init__(self, num_samples: int = 8):
        """
        Args:
            num_samples: Number of parallel reasoning traces to generate
        """
        self.num_samples = num_samples
        self.logger = logging.getLogger(__name__)
    
    def generate_parallel_traces(self, 
                                 model,
                                 prompt: str,
                                 temperature: float = 0.8) -> List[Dict]:
        """
        Generate multiple reasoning traces in parallel
        
        Args:
            model: Language model
            prompt: Input prompt
            temperature: Sampling temperature
        
        Returns:
            List of reasoning traces with confidence scores
        """
        traces = []
        
        for i in range(self.num_samples):
            # Generate trace with confidence tracking
            tracker = TokenConfidenceTracker(group_size=16, threshold=0.70)
            
            # Simulate generation (in practice, integrate with actual model)
            trace_text, confidence = self._generate_with_tracking(
                model, prompt, tracker, temperature
            )
            
            traces.append({
                'trace_id': i,
                'text': trace_text,
                'confidence': confidence,
                'stats': tracker.get_confidence_stats(),
                'length': len(trace_text.split())
            })
        
        return traces
    
    def filter_by_confidence(self, 
                            traces: List[Dict],
                            mode: str = 'high') -> List[Dict]:
        """
        Filter traces based on confidence
        
        Args:
            traces: List of reasoning traces
            mode: 'high' (top 50%), 'low' (bottom 50%), 'threshold' (>0.75)
        
        Returns:
            Filtered traces
        """
        if mode == 'high':
            # Keep top 50% by confidence
            threshold = np.median([t['confidence'] for t in traces])
            filtered = [t for t in traces if t['confidence'] >= threshold]
        
        elif mode == 'low':
            # Keep bottom 50% (for comparison)
            threshold = np.median([t['confidence'] for t in traces])
            filtered = [t for t in traces if t['confidence'] < threshold]
        
        elif mode == 'threshold':
            # Keep traces above fixed threshold
            filtered = [t for t in traces if t['confidence'] > 0.75]
        
        else:
            filtered = traces
        
        self.logger.info(
            f"Filtered {len(traces)} to {len(filtered)} traces "
            f"(mode={mode})"
        )
        
        return filtered
    
    def weighted_confidence_voting(self, 
                                   traces: List[Dict],
                                   answers: List[str]) -> Dict:
        """
        Perform weighted majority voting using confidence scores
        
        Args:
            traces: Reasoning traces with confidence
            answers: Final answers extracted from each trace
        
        Returns:
            Best answer with aggregated confidence
        """
        # Group by answer
        answer_groups = {}
        for trace, answer in zip(traces, answers):
            if answer not in answer_groups:
                answer_groups[answer] = []
            answer_groups[answer].append(trace)
        
        # Calculate weighted votes
        weighted_votes = {}
        for answer, group in answer_groups.items():
            # Sum of confidence scores
            total_weight = sum(t['confidence'] for t in group)
            weighted_votes[answer] = {
                'weight': total_weight,
                'count': len(group),
                'avg_confidence': total_weight / len(group),
                'traces': group
            }
        
        # Select answer with highest weighted vote
        best_answer = max(
            weighted_votes.items(),
            key=lambda x: x[1]['weight']
        )
        
        return {
            'answer': best_answer[0],
            'confidence': best_answer[1]['avg_confidence'],
            'vote_weight': best_answer[1]['weight'],
            'num_traces': best_answer[1]['count'],
            'total_traces': len(traces),
            'agreement': best_answer[1]['count'] / len(traces),
            'all_votes': weighted_votes
        }
    
    def _generate_with_tracking(self,
                               model,
                               prompt: str,
                               tracker: TokenConfidenceTracker,
                               temperature: float) -> Tuple[str, float]:
        """
        Generate text with confidence tracking
        
        In practice, integrate with actual model generation
        """
        # Placeholder - actual implementation would call model
        generated_text = f"Reasoning trace for: {prompt[:50]}..."
        
        # Simulate token-by-token generation
        tokens = generated_text.split()
        for token in tokens:
            # Simulate logprob (would come from actual model)
            logprob = np.random.normal(-0.5, 0.5)
            tracker.add_token(token, logprob)
            
            # Check for early stopping
            should_stop, reason = tracker.should_stop_early()
            if should_stop:
                self.logger.info(f"Early stop: {reason}")
                break
        
        stats = tracker.get_confidence_stats()
        overall_confidence = stats.get('mean_confidence', 0.0)
        
        return generated_text, overall_confidence
    
    def deep_think_with_confidence(self,
                                   model,
                                   query: str,
                                   num_samples: int = 512,
                                   filter_mode: str = 'high') -> Dict:
        """
        Main DeepConf algorithm:
        1. Generate N parallel reasoning traces
        2. Filter by confidence (keep high-quality only)
        3. Weighted confidence voting on remaining traces
        
        Args:
            model: Language model
            query: Medical query
            num_samples: Number of parallel traces (e.g., 512)
            filter_mode: How to filter traces
        
        Returns:
            Best answer with confidence metrics
        """
        self.logger.info(
            f"DeepConf: Generating {num_samples} parallel traces"
        )
        
        # 1. Generate parallel traces
        all_traces = self.generate_parallel_traces(
            model, query, temperature=0.8
        )
        
        # 2. Filter by confidence
        filtered_traces = self.filter_by_confidence(
            all_traces, mode=filter_mode
        )
        
        token_savings = 1.0 - (len(filtered_traces) / len(all_traces))
        self.logger.info(
            f"Token savings: {token_savings:.1%} "
            f"({len(all_traces)} to {len(filtered_traces)} traces)"
        )
        
        # 3. Extract answers from traces
        answers = [
            self._extract_answer(t['text']) 
            for t in filtered_traces
        ]
        
        # 4. Weighted confidence voting
        result = self.weighted_confidence_voting(filtered_traces, answers)
        
        # Add statistics
        result['token_savings'] = token_savings
        result['traces_generated'] = len(all_traces)
        result['traces_used'] = len(filtered_traces)
        
        return result
    
    def _extract_answer(self, trace_text: str) -> str:
        """Extract final answer from reasoning trace"""
        # Simple extraction - look for "Answer:" or last sentence
        if "Answer:" in trace_text:
            return trace_text.split("Answer:")[-1].strip()
        else:
            sentences = trace_text.split('.')
            return sentences[-1].strip() if sentences else trace_text


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test token-level confidence tracking
    print("=== Token Confidence Tracking ===")
    tracker = TokenConfidenceTracker(group_size=8, threshold=0.75)
    
    # Simulate token generation with varying confidence
    test_tokens = [
        ("The", -0.05),
        ("patient", -0.1),
        ("presents", -0.15),
        ("with", -0.08),
        ("fever", -0.2),
        ("and", -0.1),
        ("cough", -0.3),
        ("suggesting", -1.5),  # Low confidence
        ("pneumonia", -2.0),   # Very low confidence
    ]
    
    for token, logprob in test_tokens:
        tracker.add_token(token, logprob)
        
        if len(tracker.token_confidences) >= tracker.group_size:
            should_stop, reason = tracker.should_stop_early()
            if should_stop:
                print(f"Early stop triggered: {reason}")
                break
    
    stats = tracker.get_confidence_stats()
    print(f"Generation stats: {stats}")
    
    # Test parallel thinking filter
    print("\n=== Parallel Thinking Filter ===")
    filter = ParallelThinkingFilter(num_samples=8)
    
    # Simulate traces (in practice, generated by model)
    mock_traces = [
        {'trace_id': i, 'text': f'Trace {i}', 
         'confidence': np.random.uniform(0.6, 0.95),
         'stats': {}, 'length': 100}
        for i in range(8)
    ]
    
    # Filter traces
    high_conf_traces = filter.filter_by_confidence(mock_traces, mode='high')
    print(f"High confidence traces: {len(high_conf_traces)}/{len(mock_traces)}")
    
    # Weighted voting
    answers = [f"Answer_{i % 3}" for i in range(len(high_conf_traces))]
    result = filter.weighted_confidence_voting(high_conf_traces, answers)
    
    print(f"Best answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Agreement: {result['agreement']:.2%}")
