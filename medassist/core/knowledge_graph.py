"""
Medical Knowledge Graph using NetworkX.
Based on AMG-RAG architecture: https://github.com/MrRezaeiUofT/AMG-RAG
Paper: Agentic Medical Graph-RAG (EMNLP 2025)
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from medassist.models.entities import MedicalEntity, MedicalRelation


class MedicalKnowledgeGraph:
    """
    NetworkX-based knowledge graph for medical entities and relationships.
    Supports dynamic construction and path-based reasoning.
    """
    
    def __init__(self):
        """Initialize empty directed graph."""
        self.graph = nx.MultiDiGraph()  # Multi-edges for different relationship types
        self.entities: Dict[str, MedicalEntity] = {}
        
    def add_entity(self, entity: MedicalEntity) -> None:
        """
        Add medical entity to knowledge graph.
        
        Args:
            entity: MedicalEntity instance with name, type, and metadata
        """
        entity_id = entity.name.lower()
        
        # Add node if not exists, otherwise update attributes
        if entity_id not in self.graph:
            self.graph.add_node(
                entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                confidence=entity.confidence,
                sources=entity.sources
            )
        else:
            # Update with higher confidence data
            existing = self.graph.nodes[entity_id]
            if entity.confidence > existing.get("confidence", 0):
                self.graph.nodes[entity_id].update({
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "sources": entity.sources
                })
        
        # Store in entity registry
        self.entities[entity_id] = entity
    
    def add_relation(self, relation: MedicalRelation) -> None:
        """
        Add medical relationship between entities.
        Automatically adds bidirectional edges if both directions are valid.
        
        Args:
            relation: MedicalRelation with source, target, and metadata
        """
        source_id = relation.source.lower()
        target_id = relation.target.lower()
        
        # Ensure both entities exist in graph
        if source_id not in self.graph:
            self.add_entity(MedicalEntity(
                name=relation.source,
                description="",
                entity_type="unknown",
                confidence=0.5
            ))
        
        if target_id not in self.graph:
            self.add_entity(MedicalEntity(
                name=relation.target,
                description="",
                entity_type="unknown",
                confidence=0.5
            ))
        
        # Add edge with relationship metadata
        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            sources=relation.sources
        )
    
    def get_entity(self, entity_name: str) -> Optional[MedicalEntity]:
        """Retrieve entity by name."""
        entity_id = entity_name.lower()
        return self.entities.get(entity_id)
    
    def get_connected_entities(
        self,
        entity_name: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1
    ) -> List[Tuple[str, str, Dict]]:
        """
        Get entities connected to the given entity.
        
        Args:
            entity_name: Source entity name
            relation_types: Filter by specific relationship types
            max_depth: Maximum traversal depth (1 = direct neighbors)
            
        Returns:
            List of (source, target, edge_data) tuples
        """
        entity_id = entity_name.lower()
        
        if entity_id not in self.graph:
            return []
        
        connected = []
        
        if max_depth == 1:
            # Get direct neighbors
            for source, target, data in self.graph.edges(entity_id, data=True):
                if relation_types is None or data.get("relation_type") in relation_types:
                    connected.append((source, target, data))
            
            # Get incoming edges
            for source, target, data in self.graph.in_edges(entity_id, data=True):
                if relation_types is None or data.get("relation_type") in relation_types:
                    connected.append((source, target, data))
        else:
            # BFS traversal up to max_depth
            visited = set()
            queue = [(entity_id, 0)]
            
            while queue:
                current, depth = queue.pop(0)
                
                if current in visited or depth >= max_depth:
                    continue
                
                visited.add(current)
                
                for source, target, data in self.graph.edges(current, data=True):
                    if relation_types is None or data.get("relation_type") in relation_types:
                        connected.append((source, target, data))
                    
                    if depth + 1 < max_depth:
                        queue.append((target, depth + 1))
        
        return connected
    
    def explore_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_length: int = 3,
        relation_types: Optional[List[str]] = None
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find all paths between two entities (for chain-of-thought reasoning).
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_length: Maximum path length
            relation_types: Filter by specific relationship types
            
        Returns:
            List of paths, where each path is a list of (source, relation, target) tuples
        """
        start_id = start_entity.lower()
        end_id = end_entity.lower()
        
        if start_id not in self.graph or end_id not in self.graph:
            return []
        
        # Find all simple paths up to max_length
        try:
            paths = nx.all_simple_paths(
                self.graph,
                source=start_id,
                target=end_id,
                cutoff=max_length
            )
            
            result_paths = []
            
            for path in paths:
                # Convert node path to edge path with relation types
                edge_path = []
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    
                    # Get edge data (handle multi-edges)
                    edges = self.graph.get_edge_data(source, target)
                    if edges:
                        # Take first edge (or filter by relation_types)
                        edge_data = list(edges.values())[0]
                        relation = edge_data.get("relation_type", "related_to")
                        
                        if relation_types is None or relation in relation_types:
                            edge_path.append((source, relation, target))
                
                if edge_path:  # Only add if path has valid edges
                    result_paths.append(edge_path)
            
            return result_paths
            
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(
        self,
        entity_names: List[str],
        include_neighbors: bool = True
    ) -> nx.MultiDiGraph:
        """
        Extract subgraph containing specified entities.
        
        Args:
            entity_names: List of entity names to include
            include_neighbors: Whether to include direct neighbors
            
        Returns:
            NetworkX subgraph
        """
        entity_ids = [name.lower() for name in entity_names]
        
        if include_neighbors:
            # Add all neighbors of specified entities
            extended_ids = set(entity_ids)
            for entity_id in entity_ids:
                if entity_id in self.graph:
                    extended_ids.update(self.graph.neighbors(entity_id))
                    extended_ids.update(self.graph.predecessors(entity_id))
            entity_ids = list(extended_ids)
        
        return self.graph.subgraph(entity_ids).copy()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get knowledge graph statistics."""
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
            "entity_types": self._count_entity_types(),
            "relation_types": self._count_relation_types(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type."""
        type_counts = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get("entity_type", "unknown")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _count_relation_types(self) -> Dict[str, int]:
        """Count relationships by type."""
        type_counts = {}
        for source, target, data in self.graph.edges(data=True):
            relation_type = data.get("relation_type", "related_to")
            type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
        return type_counts
    
    def visualize_subgraph(
        self,
        entity_names: List[str],
        output_file: Optional[str] = None
    ) -> None:
        """
        Visualize subgraph (requires matplotlib).
        
        Args:
            entity_names: Entities to include in visualization
            output_file: Optional file path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            subgraph = self.get_subgraph(entity_names, include_neighbors=True)
            
            pos = nx.spring_layout(subgraph)
            
            plt.figure(figsize=(12, 8))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph,
                pos,
                node_color="lightblue",
                node_size=1000,
                alpha=0.9
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edge_color="gray",
                arrows=True,
                arrowsize=20,
                alpha=0.5
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                subgraph,
                pos,
                font_size=10,
                font_weight="bold"
            )
            
            # Draw edge labels (relation types)
            edge_labels = {
                (u, v): data.get("relation_type", "")
                for u, v, data in subgraph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                subgraph,
                pos,
                edge_labels,
                font_size=8
            )
            
            plt.title(f"Medical Knowledge Subgraph ({len(subgraph.nodes())} entities)")
            plt.axis("off")
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MedicalKnowledgeGraph("
            f"entities={stats['num_entities']}, "
            f"relations={stats['num_relations']}"
            f")"
        )
