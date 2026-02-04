"""
Hierarchical Feature Optimization for Medical RAG.
Inspired by Code RAG paper - applies to medical knowledge graphs.
"""

from typing import List, Dict, Any, Set, Tuple
import numpy as np
from dataclasses import dataclass

from medassist.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalCandidate:
    """Candidate for retrieval with features"""
    content: str
    entity_name: str
    entity_type: str
    embedding: np.ndarray
    graph_context: Dict[str, Any]
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    proximity_score: float = 0.0


class SemanticAlignmentDistillation:
    """
    Semantic Alignment Distillation (from Code RAG).
    Aligns query embeddings with knowledge graph embeddings.
    """
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        logger.info("Semantic Alignment Distillation initialized")
    
    def align_embeddings(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Compute semantic alignment scores using distillation.
        
        Similar to teacher-student distillation in Code RAG.
        """
        scores = []
        
        for candidate_emb in candidate_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb)
            )
            
            # Temperature scaling (distillation)
            score = np.exp(similarity / self.temperature)
            scores.append(float(score))
        
        # Normalize
        total = sum(scores)
        if total > 0:
            scores = [s / total for s in scores]
        
        return scores


class RedundancyAwarePruning:
    """
    Redundancy-Aware Pruning (from Code RAG).
    Removes redundant/duplicate medical entities and relationships.
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        logger.info("Redundancy-Aware Pruning initialized")
    
    def prune_redundant(
        self,
        candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        """
        Remove redundant candidates based on semantic similarity.
        
        Keeps highest-scoring candidate in each cluster.
        """
        if not candidates:
            return []
        
        # Build similarity matrix
        n = len(candidates)
        pruned = []
        used = set()
        
        # Sort by relevance score (keep best)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        for i, candidate in enumerate(sorted_candidates):
            if i in used:
                continue
            
            # Add this candidate
            pruned.append(candidate)
            used.add(i)
            
            # Mark similar candidates as used (redundant)
            for j, other in enumerate(sorted_candidates):
                if j <= i or j in used:
                    continue
                
                # Check similarity
                similarity = self._compute_similarity(candidate, other)
                if similarity > self.similarity_threshold:
                    used.add(j)
        
        logger.info(f"Pruned from {len(candidates)} to {len(pruned)} candidates")
        return pruned
    
    def _compute_similarity(
        self,
        cand1: RetrievalCandidate,
        cand2: RetrievalCandidate
    ) -> float:
        """Compute similarity between two candidates"""
        # Embedding similarity
        emb_sim = np.dot(cand1.embedding, cand2.embedding) / (
            np.linalg.norm(cand1.embedding) * np.linalg.norm(cand2.embedding)
        )
        
        # Entity name similarity (exact match bonus)
        name_sim = 1.0 if cand1.entity_name == cand2.entity_name else 0.0
        
        # Weighted combination
        return 0.7 * emb_sim + 0.3 * name_sim


class TopologicalProximityMetric:
    """
    Topological Proximity Metric (from Code RAG).
    Ranks based on graph structure and paths.
    """
    
    def __init__(self, decay_factor: float = 0.8):
        self.decay_factor = decay_factor
        logger.info("Topological Proximity Metric initialized")
    
    def compute_proximity_scores(
        self,
        candidates: List[RetrievalCandidate],
        query_entities: List[str],
        knowledge_graph: Any
    ) -> List[float]:
        """
        Compute topological proximity scores.
        
        Measures graph distance and path quality from query entities.
        """
        scores = []
        
        for candidate in candidates:
            # Find shortest paths from query entities to candidate
            min_distance = float('inf')
            path_quality = 0.0
            
            for query_entity in query_entities:
                try:
                    # Get path in knowledge graph
                    path = knowledge_graph.find_shortest_path(
                        query_entity,
                        candidate.entity_name
                    )
                    
                    if path:
                        distance = len(path) - 1
                        if distance < min_distance:
                            min_distance = distance
                            # Path quality = confidence along path
                            path_quality = self._compute_path_quality(
                                path,
                                knowledge_graph
                            )
                except:
                    continue
            
            # Proximity score with decay
            if min_distance == float('inf'):
                proximity = 0.0
            else:
                proximity = (self.decay_factor ** min_distance) * path_quality
            
            scores.append(proximity)
        
        return scores
    
    def _compute_path_quality(
        self,
        path: List[str],
        knowledge_graph: Any
    ) -> float:
        """Compute quality of path based on edge confidences"""
        if len(path) < 2:
            return 1.0
        
        confidences = []
        for i in range(len(path) - 1):
            try:
                edge_data = knowledge_graph.graph[path[i]][path[i+1]]
                confidence = edge_data.get('confidence', 0.5)
                confidences.append(confidence)
            except:
                confidences.append(0.3)  # Default low confidence
        
        # Geometric mean of confidences
        if confidences:
            product = np.prod(confidences)
            return product ** (1.0 / len(confidences))
        return 0.5


class DiversityAwareReranking:
    """
    Diversity-Aware Reranking (from Code RAG).
    Ensures diverse results across entity types and relationships.
    """
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
        logger.info("Diversity-Aware Reranking initialized")
    
    def rerank(
        self,
        candidates: List[RetrievalCandidate],
        top_k: int = 10
    ) -> List[RetrievalCandidate]:
        """
        Rerank candidates to maximize diversity.
        
        Balances relevance with diversity across entity types.
        """
        if len(candidates) <= top_k:
            return candidates
        
        # Initialize with highest-scoring candidate
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # Greedily select diverse candidates
        while len(selected) < top_k and remaining:
            # Compute diversity scores for remaining candidates
            diversity_scores = []
            
            for candidate in remaining:
                diversity = self._compute_diversity(candidate, selected)
                # Combined score: relevance + diversity
                combined = (
                    (1 - self.diversity_weight) * candidate.relevance_score +
                    self.diversity_weight * diversity
                )
                diversity_scores.append(combined)
            
            # Select best combined score
            best_idx = np.argmax(diversity_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        logger.info(f"Reranked to {len(selected)} diverse candidates")
        return selected
    
    def _compute_diversity(
        self,
        candidate: RetrievalCandidate,
        selected: List[RetrievalCandidate]
    ) -> float:
        """
        Compute diversity of candidate relative to selected set.
        
        Higher score = more diverse (different entity types, distant in graph).
        """
        if not selected:
            return 1.0
        
        # Entity type diversity
        selected_types = set(s.entity_type for s in selected)
        type_diversity = 1.0 if candidate.entity_type not in selected_types else 0.3
        
        # Embedding diversity (average distance to selected)
        distances = []
        for s in selected:
            distance = 1.0 - np.dot(candidate.embedding, s.embedding) / (
                np.linalg.norm(candidate.embedding) * np.linalg.norm(s.embedding)
            )
            distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 1.0
        
        # Combined diversity
        return 0.6 * avg_distance + 0.4 * type_diversity


class HierarchicalRetrievalOptimizer:
    """
    Complete hierarchical retrieval optimizer (Code RAG → Medical RAG).
    
    Pipeline:
    1. Semantic Alignment Distillation
    2. Redundancy-Aware Pruning
    3. Topological Proximity Metric
    4. Diversity-Aware Reranking
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        similarity_threshold: float = 0.9,
        decay_factor: float = 0.8,
        diversity_weight: float = 0.3
    ):
        self.semantic_aligner = SemanticAlignmentDistillation(temperature)
        self.pruner = RedundancyAwarePruning(similarity_threshold)
        self.proximity_metric = TopologicalProximityMetric(decay_factor)
        self.reranker = DiversityAwareReranking(diversity_weight)
        
        logger.info("Hierarchical Retrieval Optimizer initialized")
    
    def optimize_retrieval(
        self,
        query_embedding: np.ndarray,
        candidates: List[RetrievalCandidate],
        query_entities: List[str],
        knowledge_graph: Any,
        top_k: int = 10
    ) -> List[RetrievalCandidate]:
        """
        Complete optimization pipeline.
        
        Returns top-k optimized candidates.
        """
        logger.info(f"Starting hierarchical optimization with {len(candidates)} candidates")
        
        # Step 1: Semantic Alignment
        embeddings = [c.embedding for c in candidates]
        alignment_scores = self.semantic_aligner.align_embeddings(
            query_embedding,
            embeddings
        )
        
        for i, candidate in enumerate(candidates):
            candidate.relevance_score = alignment_scores[i]
        
        logger.info("✓ Semantic alignment complete")
        
        # Step 2: Redundancy Pruning
        pruned_candidates = self.pruner.prune_redundant(candidates)
        logger.info(f"✓ Pruned to {len(pruned_candidates)} candidates")
        
        # Step 3: Topological Proximity
        proximity_scores = self.proximity_metric.compute_proximity_scores(
            pruned_candidates,
            query_entities,
            knowledge_graph
        )
        
        for i, candidate in enumerate(pruned_candidates):
            candidate.proximity_score = proximity_scores[i]
            # Update relevance with proximity
            candidate.relevance_score = (
                0.6 * candidate.relevance_score +
                0.4 * candidate.proximity_score
            )
        
        logger.info("✓ Topological proximity computed")
        
        # Step 4: Diversity-Aware Reranking
        # Sort by relevance first
        sorted_candidates = sorted(
            pruned_candidates,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        reranked = self.reranker.rerank(sorted_candidates, top_k)
        logger.info(f"✓ Reranked to top-{len(reranked)} diverse results")
        
        return reranked


# Example usage
if __name__ == "__main__":
    print("Hierarchical Retrieval Optimizer Demo")
    print("=" * 60)
    
    # Mock data
    np.random.seed(42)
    
    query_emb = np.random.rand(768)
    candidates = [
        RetrievalCandidate(
            content=f"Entity {i}",
            entity_name=f"entity_{i}",
            entity_type="disease" if i % 2 == 0 else "treatment",
            embedding=np.random.rand(768),
            graph_context={}
        )
        for i in range(50)
    ]
    
    # Initialize optimizer
    optimizer = HierarchicalRetrievalOptimizer()
    
    # Optimize (without KG for demo)
    class MockKG:
        def find_shortest_path(self, a, b):
            return None
    
    optimized = optimizer.optimize_retrieval(
        query_emb,
        candidates,
        ["diabetes"],
        MockKG(),
        top_k=10
    )
    
    print(f"\nOptimized to {len(optimized)} candidates:")
    for i, cand in enumerate(optimized):
        print(f"{i+1}. {cand.entity_name} ({cand.entity_type})")
        print(f"   Relevance: {cand.relevance_score:.3f}")
        print(f"   Proximity: {cand.proximity_score:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
