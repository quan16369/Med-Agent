"""
GraphRAG Document Ingestion Pipeline.
Processes documents and extracts knowledge graph objects.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from datetime import datetime

from medassist.tools.medical_ner import MedicalNER
from medassist.models.knowledge_graph import MedicalKnowledgeGraph
from medassist.tools.multimodal import MultimodalProcessor, MultimodalInput
from medassist.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class KGObject:
    """Knowledge Graph Object extracted from document"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    vector_metadata: Dict[str, Any]
    source_document: str
    timestamp: str
    multimodal_data: Optional[Dict[str, Any]] = None  # For images, embeddings


class DocumentProcessor:
    """
    Document processor for GraphRAG ingestion pipeline.
    Extracts entities, relationships, and generates embeddings.
    """
    
    def __init__(
        self,
        ner: Optional[MedicalNER] = None,
        kg: Optional[MedicalKnowledgeGraph] = None,
        multimodal: Optional[MultimodalProcessor] = None
    ):
        """Initialize document processor"""
        self.ner = ner or MedicalNER()
        self.kg = kg or MedicalKnowledgeGraph()
        self.multimodal = multimodal or MultimodalProcessor()
        logger.info("Document processor initialized with multimodal support")
    
    def process_document(
        self,
        document: Union[str, Dict[str, Any]],
        document_id: Optional[str] = None
    ) -> KGObject:
        """
        Process a document through the ingestion pipeline.
        Supports text-only or multimodal (text + images).
        
        Args:
            document: Text content or dict with 'text' and optional 'images'
            document_id: Optional document identifier
            
        Returns:
            KGObject with extracted entities, relationships, and metadata
        """
        logger.info(f"Processing document: {document_id or 'unnamed'}")
        
        # Handle multimodal input
        if isinstance(document, dict):
            text = document.get("text", "")
            images = document.get("images", [])
            multimodal_data = None
            
            if images:
                logger.info(f"Processing {len(images)} images")
                multimodal_data = self._process_images(text, images)
        else:
            text = document
            multimodal_data = None
        
        # Extract entities using NER
        entities = self._extract_entities(text)
        
        # Infer relationships between entities
        relationships = self._infer_relationships(entities, text)
        
        # Generate vector metadata
        vector_metadata = self._generate_vector_metadata(text, entities)
        
        # Create KG object
        kg_object = KGObject(
            entities=entities,
            relationships=relationships,
            vector_metadata=vector_metadata,
            source_document=document_id or "unknown",
            timestamp=datetime.now().isoformat(),
            multimodal_data=multimodal_data
        )
        
        logger.info(
            f"Extracted {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )
        
        return kg_object
    
    def _process_images(
        self,
        text: str,
        images: List[Union[str, bytes, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Process images associated with document"""
        processed_images = []
        
        for img in images:
            if isinstance(img, dict):
                # Dict with image_path, image_bytes, or image_type
                result = self.multimodal.process_multimodal_input(
                    MultimodalInput(
                        text=text,
                        image=img.get("image_bytes"),
                        image_path=img.get("image_path"),
                        metadata=img.get("metadata")
                    )
                )
            elif isinstance(img, str):
                # Image path
                result = self.multimodal.analyze_medical_image(img)
            else:
                # Image bytes
                result = self.multimodal.image_processor.analyze_medical_image(img)
            
            processed_images.append(result)
        
        return {
            "images_processed": len(processed_images),
            "results": processed_images
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text.
        Enhanced with AMG-RAG relevance scoring and descriptions.
        """
        extracted = self.ner.extract_entities(text)
        
        entities = []
        for entity in extracted:
            entity_dict = {
                "name": entity.text,
                "type": entity.label,
                "confidence": entity.confidence,
                "position": {
                    "start": entity.start,
                    "end": entity.end
                }
            }
            
            # AMG-RAG enhancements
            if hasattr(entity, 'relevance_score'):
                entity_dict["relevance_score"] = entity.relevance_score
            if hasattr(entity, 'description'):
                entity_dict["description"] = entity.description
            
            entities.append(entity_dict)
        
        return entities
    
    def _infer_relationships(
        self,
        entities: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Infer relationships between entities based on context.
        Enhanced with AMG-RAG bidirectional analysis.
        Uses simple heuristics and proximity.
        """
        relationships = []
        
        # Relationship patterns with bidirectional mapping (AMG-RAG)
        patterns = {
            "causes": {
                "patterns": [
                    r"(\w+)\s+causes?\s+(\w+)",
                    r"(\w+)\s+leads? to\s+(\w+)",
                    r"(\w+)\s+results? in\s+(\w+)"
                ],
                "reverse": "caused_by"  # AMG-RAG: Bidirectional
            },
            "treats": {
                "patterns": [
                    r"(\w+)\s+treats?\s+(\w+)",
                    r"(\w+)\s+for\s+(\w+)",
                    r"treat\s+(\w+)\s+with\s+(\w+)"
                ],
                "reverse": "treated_by"  # AMG-RAG: Bidirectional
            },
            "has_symptom": {
                "patterns": [
                    r"(\w+)\s+(?:has|have|with)\s+symptoms?\s+(?:of\s+)?(\w+)",
                    r"symptoms?\s+of\s+(\w+)\s+include\s+(\w+)"
                ],
                "reverse": "symptom_of"  # AMG-RAG: Bidirectional
            }
        }
        
        # Find relationships using patterns
        for relation_type, relation_data in patterns.items():
            for pattern in relation_data["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)
                    
                    # Verify entities exist
                    source_exists = any(
                        e["name"].lower() == source.lower()
                        for e in entities
                    )
                    target_exists = any(
                        e["name"].lower() == target.lower()
                        for e in entities
                    )
                    
                    if source_exists and target_exists:
                        # Forward relationship (A → B)
                        relationships.append({
                            "source": source,
                            "relation": relation_type,
                            "target": target,
                            "confidence": 0.7,
                            "evidence": match.group(0)  # AMG-RAG: Evidence
                        })
                        
                        # AMG-RAG: Bidirectional relationship (B → A)
                        reverse_relation = relation_data.get("reverse")
                        if reverse_relation:
                            relationships.append({
                                "source": target,
                                "relation": reverse_relation,
                                "target": source,
                                "confidence": 0.65,  # Slightly lower for inferred
                                "evidence": f"Inverse of: {match.group(0)}"
                            })
        
        # Proximity-based relationships
        # Entities close to each other likely have relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                distance = abs(
                    entity1["position"]["start"] - entity2["position"]["start"]
                )
                
                # If entities are within 50 characters, infer relationship
                if distance < 50:
                    # Determine relationship type based on entity types
                    rel_type = self._infer_relation_type(
                        entity1["type"],
                        entity2["type"]
                    )
                    
                    if rel_type:
                        # Forward relationship
                        relationships.append({
                            "source": entity1["name"],
                            "relation": rel_type,
                            "target": entity2["name"],
                            "confidence": 0.5,
                            "evidence": "proximity"
                        })
                        
                        # AMG-RAG: Add reverse if applicable
                        reverse_rel = self._get_reverse_relation(rel_type)
                        if reverse_rel:
                            relationships.append({
                                "source": entity2["name"],
                                "relation": reverse_rel,
                                "target": entity1["name"],
                                "confidence": 0.45,
                                "evidence": "proximity (inverse)"
                            })
        
        return relationships
    
    def _get_reverse_relation(self, relation: str) -> Optional[str]:
        """Get reverse relationship type (AMG-RAG enhancement)"""
        reverse_map = {
            "causes": "caused_by",
            "treats": "treated_by",
            "has_symptom": "symptom_of",
            "related_to": "related_to",  # Symmetric
            "associated_with": "associated_with"  # Symmetric
        }
        return reverse_map.get(relation)
    
    def _infer_relation_type(self, type1: str, type2: str) -> Optional[str]:
        """Infer relationship type based on entity types"""
        relation_map = {
            ("Disease", "Symptom"): "has_symptom",
            ("Disease", "Treatment"): "treated_by",
            ("Treatment", "Disease"): "treats",
            ("Disease", "Disease"): "related_to",
            ("Symptom", "Disease"): "indicates"
        }
        
        return relation_map.get((type1, type2))
    
    def _generate_vector_metadata(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate metadata for vector embeddings"""
        return {
            "text_length": len(text),
            "entity_count": len(entities),
            "entity_types": list(set(e["type"] for e in entities)),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "processed_at": datetime.now().isoformat()
        }
    
    def write_to_kg(self, kg_object: KGObject):
        """Write KG object to knowledge graph"""
        logger.info(f"Writing KG object from {kg_object.source_document} to graph")
        
        # Write entities
        for entity in kg_object.entities:
            self.kg.add_entity(
                entity["name"],
                entity["type"]
            )
        
        # Write relationships
        for rel in kg_object.relationships:
            self.kg.add_relationship(
                rel["source"],
                rel["relation"],
                rel["target"]
            )
        
        logger.info(
            f"Wrote {len(kg_object.entities)} entities, "
            f"{len(kg_object.relationships)} relationships to KG"
        )
    
    def batch_process(
        self,
        documents: List[Dict[str, str]]
    ) -> List[KGObject]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of {"id": str, "content": str} dicts
            
        Returns:
            List of KGObject instances
        """
        logger.info(f"Batch processing {len(documents)} documents")
        
        kg_objects = []
        for doc in documents:
            try:
                kg_obj = self.process_document(
                    doc["content"],
                    doc.get("id")
                )
                kg_objects.append(kg_obj)
                
                # Write to KG
                self.write_to_kg(kg_obj)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc.get('id')}: {e}")
        
        logger.info(f"Successfully processed {len(kg_objects)} documents")
        return kg_objects


class IngestionPipeline:
    """
    Complete GraphRAG ingestion pipeline.
    Coordinates document processing and KG writing.
    """
    
    def __init__(
        self,
        processor: Optional[DocumentProcessor] = None,
        kg: Optional[MedicalKnowledgeGraph] = None
    ):
        """Initialize ingestion pipeline"""
        self.kg = kg or MedicalKnowledgeGraph()
        self.processor = processor or DocumentProcessor(kg=self.kg)
        self.memory = []  # Store processed KG objects
        logger.info("Ingestion pipeline initialized")
    
    def ingest_document(self, document: str, document_id: Optional[str] = None):
        """Ingest a single document"""
        kg_object = self.processor.process_document(document, document_id)
        self.processor.write_to_kg(kg_object)
        self.memory.append(kg_object)
        return kg_object
    
    def ingest_batch(self, documents: List[Dict[str, str]]):
        """Ingest multiple documents"""
        kg_objects = self.processor.batch_process(documents)
        self.memory.extend(kg_objects)
        return kg_objects
    
    def search_memory(self, query: str) -> List[KGObject]:
        """Search ingested documents in memory"""
        results = []
        for kg_obj in self.memory:
            # Simple text match in entities
            for entity in kg_obj.entities:
                if query.lower() in entity["name"].lower():
                    results.append(kg_obj)
                    break
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "total_documents": len(self.memory),
            "total_entities": sum(len(obj.entities) for obj in self.memory),
            "total_relationships": sum(len(obj.relationships) for obj in self.memory),
            "entity_types": list(set(
                e["type"]
                for obj in self.memory
                for e in obj.entities
            ))
        }
