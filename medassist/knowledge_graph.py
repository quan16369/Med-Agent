"""
Medical Knowledge Graph
Stores medical entities and their relationships in a graph structure
Enables multi-hop reasoning for better medical QA
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j driver not available. Using in-memory graph.")


@dataclass
class MedicalEntity:
    """Medical entity in knowledge graph"""
    id: str
    name: str
    type: str  # disease, symptom, treatment, anatomy, biomarker
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class MedicalRelationship:
    """Relationship between medical entities"""
    source_id: str
    target_id: str
    relation_type: str  # causes, treats, affects, indicates, contraindicated_with
    confidence: float  # 0-1
    evidence_count: int = 0
    source: str = "extracted"  # pubmed, wikipedia, textbook
    
    def __post_init__(self):
        assert 0 <= self.confidence <= 1, "Confidence must be between 0 and 1"


@dataclass
class KnowledgePath:
    """Path through knowledge graph"""
    nodes: List[MedicalEntity]
    edges: List[MedicalRelationship]
    confidence: float
    
    def __len__(self):
        return len(self.nodes)
    
    def to_string(self) -> str:
        """Convert path to readable string"""
        result = []
        for i, node in enumerate(self.nodes):
            result.append(node.name)
            if i < len(self.edges):
                result.append(f"--[{self.edges[i].relation_type}]-->")
        return " ".join(result)


class MedicalKnowledgeGraph:
    """
    Medical Knowledge Graph using Neo4j or in-memory representation
    
    Based on AMG-RAG paper architecture:
    - ~76K nodes, ~354K relationships
    - Confidence-scored edges
    - Multi-hop path finding
    - BFS/DFS traversal
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_memory: bool = False
    ):
        self.use_memory = use_memory or not NEO4J_AVAILABLE
        
        if self.use_memory:
            logger.info("Using in-memory knowledge graph")
            self._init_memory_graph()
        else:
            logger.info("Using Neo4j knowledge graph")
            self._init_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    
    def _init_memory_graph(self):
        """Initialize in-memory graph representation"""
        self.nodes: Dict[str, MedicalEntity] = {}
        self.edges: List[MedicalRelationship] = []
        self.adjacency: Dict[str, List[Tuple[str, MedicalRelationship]]] = {}
    
    def _init_neo4j(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        if not NEO4J_AVAILABLE:
            logger.error("Neo4j driver not installed")
            self._init_memory_graph()
            return
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.info("Falling back to in-memory graph")
            self.use_memory = True
            self._init_memory_graph()
    
    def add_entity(self, entity: MedicalEntity):
        """Add medical entity to graph"""
        if self.use_memory:
            self.nodes[entity.id] = entity
            if entity.id not in self.adjacency:
                self.adjacency[entity.id] = []
        else:
            self._add_entity_neo4j(entity)
    
    def _add_entity_neo4j(self, entity: MedicalEntity):
        """Add entity to Neo4j"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:MedicalEntity {id: $id})
                SET e.name = $name, e.type = $type, e.aliases = $aliases
                """,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases
            )
    
    def add_relationship(self, relationship: MedicalRelationship):
        """Add relationship between entities"""
        if self.use_memory:
            self.edges.append(relationship)
            
            # Update adjacency list
            if relationship.source_id in self.adjacency:
                self.adjacency[relationship.source_id].append(
                    (relationship.target_id, relationship)
                )
        else:
            self._add_relationship_neo4j(relationship)
    
    def _add_relationship_neo4j(self, rel: MedicalRelationship):
        """Add relationship to Neo4j"""
        with self.driver.session() as session:
            # Create relationship with type
            query = f"""
            MATCH (a:MedicalEntity {{id: $source_id}})
            MATCH (b:MedicalEntity {{id: $target_id}})
            MERGE (a)-[r:{rel.relation_type.upper()}]->(b)
            SET r.confidence = $confidence,
                r.evidence_count = $evidence_count,
                r.source = $source
            """
            session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                confidence=rel.confidence,
                evidence_count=rel.evidence_count,
                source=rel.source
            )
    
    def find_entity(self, name: str) -> Optional[MedicalEntity]:
        """Find entity by name or alias"""
        if self.use_memory:
            # Search by name
            for entity in self.nodes.values():
                if entity.name.lower() == name.lower():
                    return entity
                if name.lower() in [a.lower() for a in entity.aliases]:
                    return entity
            return None
        else:
            return self._find_entity_neo4j(name)
    
    def _find_entity_neo4j(self, name: str) -> Optional[MedicalEntity]:
        """Find entity in Neo4j"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:MedicalEntity)
                WHERE e.name =~ $pattern OR any(alias IN e.aliases WHERE alias =~ $pattern)
                RETURN e.id as id, e.name as name, e.type as type, e.aliases as aliases
                LIMIT 1
                """,
                pattern=f"(?i).*{name}.*"  # Case-insensitive contains
            )
            
            record = result.single()
            if record:
                return MedicalEntity(
                    id=record["id"],
                    name=record["name"],
                    type=record["type"],
                    aliases=record["aliases"] or []
                )
            return None
    
    def find_paths(
        self,
        start_entity: str,
        max_hops: int = 3,
        strategy: str = "bfs",
        min_confidence: float = 0.5
    ) -> List[KnowledgePath]:
        """
        Find knowledge paths from start entity
        
        Args:
            start_entity: Starting entity name
            max_hops: Maximum path length
            strategy: "bfs" (breadth-first) or "dfs" (depth-first)
            min_confidence: Minimum path confidence threshold
        
        Returns:
            List of knowledge paths sorted by confidence
        """
        start = self.find_entity(start_entity)
        if not start:
            logger.warning(f"Entity not found: {start_entity}")
            return []
        
        if self.use_memory:
            if strategy == "bfs":
                paths = self._bfs_traverse(start, max_hops)
            else:
                paths = self._dfs_traverse(start, max_hops)
        else:
            paths = self._find_paths_neo4j(start, max_hops, strategy)
        
        # Filter by confidence
        paths = [p for p in paths if p.confidence >= min_confidence]
        
        # Sort by confidence
        paths.sort(key=lambda x: x.confidence, reverse=True)
        
        return paths
    
    def _bfs_traverse(
        self,
        start: MedicalEntity,
        max_hops: int
    ) -> List[KnowledgePath]:
        """Breadth-first traversal for path finding"""
        paths = []
        queue = deque([(start, [], [])])  # (node, path_nodes, path_edges)
        visited = set([start.id])
        
        while queue:
            node, path_nodes, path_edges = queue.popleft()
            current_path = path_nodes + [node]
            
            # Add current path
            if len(current_path) > 1:
                confidence = self._compute_path_confidence(path_edges)
                paths.append(KnowledgePath(
                    nodes=current_path,
                    edges=path_edges,
                    confidence=confidence
                ))
            
            # Stop if max hops reached
            if len(path_edges) >= max_hops:
                continue
            
            # Explore neighbors
            if node.id in self.adjacency:
                for neighbor_id, edge in self.adjacency[node.id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor = self.nodes[neighbor_id]
                        queue.append((
                            neighbor,
                            current_path,
                            path_edges + [edge]
                        ))
        
        return paths
    
    def _dfs_traverse(
        self,
        start: MedicalEntity,
        max_hops: int
    ) -> List[KnowledgePath]:
        """Depth-first traversal for path finding"""
        paths = []
        visited = set()
        
        def dfs(node, path_nodes, path_edges, depth):
            if depth > max_hops:
                return
            
            current_path = path_nodes + [node]
            
            # Add current path
            if len(current_path) > 1:
                confidence = self._compute_path_confidence(path_edges)
                paths.append(KnowledgePath(
                    nodes=current_path,
                    edges=path_edges,
                    confidence=confidence
                ))
            
            # Explore neighbors
            visited.add(node.id)
            if node.id in self.adjacency:
                for neighbor_id, edge in self.adjacency[node.id]:
                    if neighbor_id not in visited:
                        neighbor = self.nodes[neighbor_id]
                        dfs(neighbor, current_path, path_edges + [edge], depth + 1)
            visited.remove(node.id)
        
        dfs(start, [], [], 0)
        return paths
    
    def _find_paths_neo4j(
        self,
        start: MedicalEntity,
        max_hops: int,
        strategy: str
    ) -> List[KnowledgePath]:
        """Find paths using Neo4j Cypher"""
        with self.driver.session() as session:
            # Cypher query for variable length paths
            result = session.run(
                """
                MATCH path = (start:MedicalEntity {id: $start_id})-[*1..%d]->(end:MedicalEntity)
                RETURN path
                LIMIT 100
                """ % max_hops,
                start_id=start.id
            )
            
            paths = []
            for record in result:
                path = record["path"]
                
                # Extract nodes
                nodes = []
                for node in path.nodes:
                    nodes.append(MedicalEntity(
                        id=node["id"],
                        name=node["name"],
                        type=node["type"],
                        aliases=node.get("aliases", [])
                    ))
                
                # Extract edges
                edges = []
                for rel in path.relationships:
                    edges.append(MedicalRelationship(
                        source_id=rel.start_node["id"],
                        target_id=rel.end_node["id"],
                        relation_type=rel.type.lower(),
                        confidence=rel.get("confidence", 0.5),
                        evidence_count=rel.get("evidence_count", 0),
                        source=rel.get("source", "unknown")
                    ))
                
                # Compute path confidence
                confidence = self._compute_path_confidence(edges)
                
                paths.append(KnowledgePath(
                    nodes=nodes,
                    edges=edges,
                    confidence=confidence
                ))
            
            return paths
    
    def _compute_path_confidence(self, edges: List[MedicalRelationship]) -> float:
        """
        Compute confidence score for a path
        Uses geometric mean of edge confidences
        """
        if not edges:
            return 1.0
        
        # Geometric mean
        product = 1.0
        for edge in edges:
            product *= edge.confidence
        
        return product ** (1.0 / len(edges))
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        if self.use_memory:
            return {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "node_types": self._count_node_types(),
                "edge_types": self._count_edge_types()
            }
        else:
            return self._get_statistics_neo4j()
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node in self.nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type"""
        counts = {}
        for edge in self.edges:
            counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
        return counts
    
    def _get_statistics_neo4j(self) -> Dict:
        """Get statistics from Neo4j"""
        with self.driver.session() as session:
            # Count nodes
            node_count = session.run("MATCH (n:MedicalEntity) RETURN count(n) as count").single()["count"]
            
            # Count relationships
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return {
                "num_nodes": node_count,
                "num_edges": edge_count
            }
    
    def close(self):
        """Close database connection"""
        if not self.use_memory and hasattr(self, 'driver'):
            self.driver.close()


if __name__ == "__main__":
    # Demo
    print("Medical Knowledge Graph Demo")
    print("="*60)
    
    # Initialize graph
    kg = MedicalKnowledgeGraph(use_memory=True)
    
    # Add entities
    diabetes = MedicalEntity(
        id="disease_001",
        name="Diabetes Mellitus",
        type="disease",
        aliases=["diabetes", "DM"]
    )
    
    hyperglycemia = MedicalEntity(
        id="symptom_001",
        name="Hyperglycemia",
        type="symptom",
        aliases=["high blood sugar"]
    )
    
    neuropathy = MedicalEntity(
        id="disease_002",
        name="Diabetic Neuropathy",
        type="disease",
        aliases=["nerve damage"]
    )
    
    kg.add_entity(diabetes)
    kg.add_entity(hyperglycemia)
    kg.add_entity(neuropathy)
    
    # Add relationships
    kg.add_relationship(MedicalRelationship(
        source_id="disease_001",
        target_id="symptom_001",
        relation_type="causes",
        confidence=0.95,
        evidence_count=150
    ))
    
    kg.add_relationship(MedicalRelationship(
        source_id="disease_001",
        target_id="disease_002",
        relation_type="causes",
        confidence=0.88,
        evidence_count=75
    ))
    
    # Find paths
    print("\nFinding paths from 'Diabetes Mellitus'...")
    paths = kg.find_paths("Diabetes Mellitus", max_hops=2)
    
    print(f"\nFound {len(paths)} paths:")
    for i, path in enumerate(paths[:5]):
        print(f"\n{i+1}. {path.to_string()}")
        print(f"   Confidence: {path.confidence:.3f}")
    
    # Statistics
    stats = kg.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Node types: {stats['node_types']}")
    print(f"  Edge types: {stats['edge_types']}")
