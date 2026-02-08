"""
Hierarchical Feature Optimization for AMG-RAG
Inspired by SaraCoder's diversity-maximizing retrieval approach.

This module optimizes entity and evidence retrieval by:
1. Semantic deduplication
2. Structural similarity analysis (graph-based)
3. Diversity-based reranking (MMR)
4. Topological importance weighting
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizedEntity:
    """Entity with optimization metadata."""
    name: str
    entity_type: str
    confidence: float
    relevance: int  # 1-10 scale
    semantic_cluster: int
    structural_importance: float
    diversity_score: float


@dataclass
class OptimizedEvidence:
    """Evidence with diversity metadata."""
    text: str
    source: str
    relevance_score: float
    semantic_cluster: int
    diversity_score: float
    novelty: float


class HierarchicalOptimizer:
    """
    Hierarchical optimization for resource-constrained medical RAG.
    
    Following SaraCoder principles:
    - Maximize information diversity and representativeness
    - Graph-based structural similarity
    - Topological importance weighting
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        diversity_lambda: float = 0.5,
        max_cluster_size: int = 3
    ):
        """
        Args:
            similarity_threshold: Threshold for duplicate detection (0-1)
            diversity_lambda: Balance between relevance and diversity (0-1)
            max_cluster_size: Max entities per semantic cluster
        """
        self.similarity_threshold = similarity_threshold
        self.diversity_lambda = diversity_lambda
        self.max_cluster_size = max_cluster_size
    
    # ==================== Level 1: Semantic Deduplication ====================
    
    def deduplicate_entities(
        self,
        entities: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Remove exact and near-duplicate entities.
        
        SaraCoder: "systematically refines candidates by distilling deep
        semantic relationships, pruning exact duplicates"
        """
        if not entities:
            return []
        
        # Group by exact name match (case-insensitive)
        name_groups = defaultdict(list)
        for entity in entities:
            key = entity['name'].lower().strip()
            name_groups[key].append(entity)
        
        deduplicated = []
        for name, group in name_groups.items():
            # Keep highest confidence entity from each group
            best_entity = max(group, key=lambda e: e.get('confidence', 0.5))
            deduplicated.append(best_entity)
        
        logger.info(f"Deduplicated {len(entities)} → {len(deduplicated)} entities")
        return deduplicated
    
    def semantic_clustering(
        self,
        entities: List[Dict[str, any]],
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[int, List[Dict[str, any]]]:
        """
        Cluster semantically similar entities.
        
        Uses simple type-based clustering (can be enhanced with embeddings).
        """
        if embeddings is not None:
            return self._cluster_by_embeddings(entities, embeddings)
        
        # Type-based clustering (fallback)
        clusters = defaultdict(list)
        type_to_cluster = {}
        cluster_id = 0
        
        for entity in entities:
            entity_type = entity.get('entity_type', 'unknown')
            if entity_type not in type_to_cluster:
                type_to_cluster[entity_type] = cluster_id
                cluster_id += 1
            clusters[type_to_cluster[entity_type]].append(entity)
        
        return dict(clusters)
    
    def _cluster_by_embeddings(
        self,
        entities: List[Dict[str, any]],
        embeddings: np.ndarray
    ) -> Dict[int, List[Dict[str, any]]]:
        """Cluster entities using embedding similarity."""
        # Simple greedy clustering
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, entity in enumerate(entities):
            if i in assigned:
                continue
            
            cluster = [entity]
            assigned.add(i)
            
            # Find similar entities
            for j in range(i + 1, len(entities)):
                if j in assigned:
                    continue
                
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity > self.similarity_threshold:
                    cluster.append(entities[j])
                    assigned.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    # ==================== Level 2: Structural Similarity ====================
    
    def compute_structural_importance(
        self,
        entities: List[Dict[str, any]],
        knowledge_graph: any  # MedicalKnowledgeGraph instance
    ) -> Dict[str, float]:
        """
        Compute topological importance of entities in KG.
        
        SaraCoder: "assessing structural similarity with a novel graph-based
        metric that weighs edits by their topological importance"
        
        Uses graph centrality metrics:
        - Degree centrality: How connected is the entity?
        - Betweenness centrality: How important for connecting other entities?
        - PageRank: Global importance in the graph
        """
        importance_scores = {}
        
        if not hasattr(knowledge_graph, 'graph'):
            # No graph available, use uniform importance
            return {e['name']: 1.0 for e in entities}
        
        graph = knowledge_graph.graph
        
        # Compute centrality metrics
        try:
            import networkx as nx
            
            # Degree centrality (normalized)
            degree_centrality = nx.degree_centrality(graph)
            
            # Betweenness centrality (how often entity is on shortest paths)
            betweenness_centrality = nx.betweenness_centrality(graph)
            
            # PageRank (global importance)
            pagerank = nx.pagerank(graph)
            
            # Combine metrics (weighted average)
            for entity in entities:
                name = entity['name']
                if name not in graph.nodes:
                    importance_scores[name] = 0.1  # Low importance for isolated entities
                    continue
                
                # Weighted combination
                importance = (
                    0.4 * degree_centrality.get(name, 0) +
                    0.3 * betweenness_centrality.get(name, 0) +
                    0.3 * pagerank.get(name, 0)
                )
                importance_scores[name] = importance
        
        except ImportError:
            logger.warning("NetworkX not available, using uniform importance")
            importance_scores = {e['name']: 1.0 for e in entities}
        
        return importance_scores
    
    # ==================== Level 3: Diversity-Based Reranking ====================
    
    def mmr_rerank(
        self,
        candidates: List[Dict[str, any]],
        query_embedding: Optional[np.ndarray] = None,
        candidate_embeddings: Optional[np.ndarray] = None,
        k: int = 8
    ) -> List[Dict[str, any]]:
        """
        Maximal Marginal Relevance (MMR) reranking.
        
        SaraCoder: "reranking results to maximize both relevance and diversity"
        
        MMR = λ * Relevance(entity, query) - (1-λ) * max_similarity(entity, selected)
        
        Balances:
        - Relevance to query (high confidence/relevance scores)
        - Diversity from already selected entities
        """
        if not candidates:
            return []
        
        if query_embedding is None or candidate_embeddings is None:
            # Fallback: Use confidence/relevance scores only
            return self._mmr_without_embeddings(candidates, k)
        
        selected = []
        remaining = list(range(len(candidates)))
        
        # Select first: highest relevance
        first_idx = max(remaining, key=lambda i: candidates[i].get('relevance', 5))
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Iteratively select diverse entities
        while len(selected) < k and remaining:
            mmr_scores = {}
            
            for i in remaining:
                # Relevance score (query similarity)
                relevance = self._cosine_similarity(
                    query_embedding, 
                    candidate_embeddings[i]
                )
                
                # Diversity score (max similarity to already selected)
                diversity = max([
                    self._cosine_similarity(
                        candidate_embeddings[i],
                        candidate_embeddings[j]
                    ) for j in selected
                ])
                
                # MMR score
                mmr_scores[i] = (
                    self.diversity_lambda * relevance -
                    (1 - self.diversity_lambda) * diversity
                )
            
            # Select highest MMR score
            best_idx = max(mmr_scores.keys(), key=lambda i: mmr_scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [candidates[i] for i in selected]
    
    def _mmr_without_embeddings(
        self,
        candidates: List[Dict[str, any]],
        k: int
    ) -> List[Dict[str, any]]:
        """MMR fallback using type diversity and relevance scores."""
        selected = []
        remaining = candidates.copy()
        selected_types = set()
        
        while len(selected) < k and remaining:
            # Prefer entities with new types (diversity)
            scored = []
            for entity in remaining:
                relevance = entity.get('relevance', 5) / 10.0  # Normalize to 0-1
                entity_type = entity.get('entity_type', 'unknown')
                
                # Diversity bonus for new types
                diversity_bonus = 0.3 if entity_type not in selected_types else 0.0
                
                score = self.diversity_lambda * relevance + diversity_bonus
                scored.append((score, entity))
            
            # Select best
            best = max(scored, key=lambda x: x[0])[1]
            selected.append(best)
            remaining.remove(best)
            selected_types.add(best.get('entity_type', 'unknown'))
        
        return selected
    
    # ==================== Level 4: Integrated Optimization ====================
    
    def optimize_entities(
        self,
        entities: List[Dict[str, any]],
        knowledge_graph: any,
        max_entities: int = 8,
        embeddings: Optional[np.ndarray] = None
    ) -> List[Dict[str, any]]:
        """
        Full hierarchical optimization pipeline.
        
        Pipeline:
        1. Deduplicate exact/near duplicates
        2. Compute structural importance (graph centrality)
        3. Cluster by semantic similarity
        4. Rerank with MMR for diversity
        5. Select top-k with balanced representation
        """
        if not entities:
            return []
        
        # Level 1: Deduplication
        entities = self.deduplicate_entities(entities)
        
        # Level 2: Structural importance
        importance_scores = self.compute_structural_importance(entities, knowledge_graph)
        for entity in entities:
            entity['structural_importance'] = importance_scores.get(entity['name'], 0.1)
        
        # Level 3: Semantic clustering
        clusters = self.semantic_clustering(entities, embeddings)
        
        # Level 4: Diversity-based selection
        # Select representatives from each cluster
        optimized = []
        entities_per_cluster = max(1, max_entities // len(clusters))
        
        for cluster_id, cluster_entities in clusters.items():
            # Sort by combined score: relevance * structural_importance
            cluster_entities.sort(
                key=lambda e: e.get('relevance', 5) * e.get('structural_importance', 0.5),
                reverse=True
            )
            
            # Take top entities from each cluster
            optimized.extend(cluster_entities[:entities_per_cluster])
        
        # Final MMR reranking
        if len(optimized) > max_entities:
            optimized = self.mmr_rerank(optimized, k=max_entities)
        
        logger.info(f"Optimized {len(entities)} → {len(optimized)} entities (target: {max_entities})")
        return optimized[:max_entities]
    
    def optimize_evidence(
        self,
        evidence_list: List[Dict[str, any]],
        max_evidence: int = 5,
        embeddings: Optional[np.ndarray] = None
    ) -> List[Dict[str, any]]:
        """
        Optimize evidence selection for maximum diversity.
        
        Similar to entity optimization but focused on text evidence.
        """
        if not evidence_list:
            return []
        
        # Deduplicate by source/title
        seen = set()
        deduplicated = []
        for evidence in evidence_list:
            key = (evidence.get('source', ''), evidence.get('title', ''))
            if key not in seen:
                seen.add(key)
                deduplicated.append(evidence)
        
        # MMR reranking for diversity
        if embeddings is not None and len(deduplicated) > max_evidence:
            # Use MMR with embeddings
            deduplicated = self.mmr_rerank(deduplicated, k=max_evidence)
        else:
            # Simple diversity: prefer different sources
            deduplicated.sort(
                key=lambda e: (e.get('source', ''), -e.get('relevance', 0.5)),
                reverse=True
            )
            deduplicated = deduplicated[:max_evidence]
        
        logger.info(f"Optimized evidence: {len(evidence_list)} → {len(deduplicated)}")
        return deduplicated
