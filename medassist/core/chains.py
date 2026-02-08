"""
LLM chains for entity and relationship extraction.
Based on AMG-RAG paper methodology with 1-10 relevance scoring.
"""

from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from medassist.models.entities import MedicalEntity, MedicalRelation


class EntityExtractionOutput(BaseModel):
    """Structured output for entity extraction with relevance scoring."""
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type: disease, treatment, symptom, etc.")
    description: str = Field(description="Brief description of the entity")
    relevance: int = Field(description="Relevance score from 1-10 (AMG-RAG paper)")
    confidence: float = Field(description="Confidence score from 0-1")


class RelationExtractionOutput(BaseModel):
    """Structured output for relationship extraction."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Relationship type: treats, causes, symptom_of, etc.")
    evidence: str = Field(description="Evidence supporting this relationship")
    confidence: float = Field(description="Confidence score from 0-1")


class EntityExtractor:
    """
    Extract medical entities from text with relevance scoring.
    Following AMG-RAG paper: entities scored 1-10 for query relevance.
    """
    
    ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a medical knowledge extraction expert. Extract medical entities from the given text.

For each entity, provide:
- name: The entity name (normalized medical term)
- entity_type: One of: disease, treatment, symptom, risk_factor, diagnostic_test, body_part, gene, protein
- description: Brief medical description (1-2 sentences)
- relevance: Score from 1-10 indicating relevance to the query (10 = highly relevant, 1 = tangentially related)
- confidence: Score from 0-1 indicating extraction confidence (1 = very certain, 0 = uncertain)

Guidelines:
1. Extract only clinically significant entities
2. Use standard medical terminology (not lay terms)
3. Include both explicit and implicit entities
4. Prioritize entities directly related to the query

Output as JSON array of entities."""),
        ("human", "Query: {query}\n\nText to analyze:\n{text}")
    ])
    
    def __init__(self, llm):
        """
        Initialize entity extractor.
        
        Args:
            llm: LangChain chat model (MedGemma)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=EntityExtractionOutput)
        
        self.chain = (
            self.ENTITY_EXTRACTION_PROMPT
            | self.llm
            | self.parser
        )
    
    def extract(
        self,
        text: str,
        query: str,
        min_relevance: int = 5,
        source: str = "extracted"
    ) -> List[MedicalEntity]:
        """
        Extract entities from text with relevance filtering.
        
        Args:
            text: Medical text to extract from
            query: User query for relevance scoring
            min_relevance: Minimum relevance score (1-10)
            source: Source identifier for provenance
            
        Returns:
            List of MedicalEntity objects
        """
        try:
            result = self.chain.invoke({
                "text": text,
                "query": query
            })
            
            # Handle both single entity and array responses
            if isinstance(result, dict):
                result = [result]
            
            entities = []
            for item in result:
                # Filter by relevance score
                if item.get("relevance", 0) >= min_relevance:
                    entity = MedicalEntity(
                        name=item["name"],
                        description=item.get("description", ""),
                        entity_type=item.get("entity_type", "unknown"),
                        confidence=item.get("confidence", 0.5),
                        sources=[source]
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return []


class RelationExtractor:
    """
    Extract bidirectional relationships between medical entities.
    Following AMG-RAG: both A→B and B→A analyzed for validity.
    """
    
    RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a medical relationship extraction expert. Analyze relationships between entities.

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- relation_type: One of: treats, causes, symptom_of, risk_factor_for, diagnosed_by, associated_with, prevents, part_of, indicates, contraindicates
- evidence: Textual evidence supporting this relationship (quote from text)
- confidence: Score from 0-1 indicating relationship confidence

Guidelines:
1. Extract only relationships explicitly or implicitly stated in the text
2. Consider BIDIRECTIONAL relationships: if A treats B, check if B is_treated_by A
3. Use standard medical relationship types
4. Provide evidence from the source text
5. Assign confidence based on evidence strength

Output as JSON array of relationships."""),
        ("human", "Text to analyze:\n{text}\n\nExtracted entities:\n{entities}")
    ])
    
    def __init__(self, llm):
        """
        Initialize relationship extractor.
        
        Args:
            llm: LangChain chat model (MedGemma)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=RelationExtractionOutput)
        
        self.chain = (
            self.RELATION_EXTRACTION_PROMPT
            | self.llm
            | self.parser
        )
    
    def extract(
        self,
        text: str,
        entities: List[MedicalEntity],
        min_confidence: float = 0.5,
        source: str = "extracted"
    ) -> List[MedicalRelation]:
        """
        Extract relationships from text given known entities.
        
        Args:
            text: Medical text to extract from
            entities: List of known entities in the text
            min_confidence: Minimum confidence threshold
            source: Source identifier for provenance
            
        Returns:
            List of MedicalRelation objects
        """
        try:
            # Format entities for prompt
            entity_list = "\n".join([
                f"- {e.name} ({e.entity_type})"
                for e in entities
            ])
            
            result = self.chain.invoke({
                "text": text,
                "entities": entity_list
            })
            
            # Handle both single relation and array responses
            if isinstance(result, dict):
                result = [result]
            
            relations = []
            for item in result:
                # Filter by confidence
                if item.get("confidence", 0) >= min_confidence:
                    relation = MedicalRelation(
                        source=item["source"],
                        target=item["target"],
                        relation_type=item.get("relation_type", "related_to"),
                        confidence=item.get("confidence", 0.5),
                        evidence=item.get("evidence", ""),
                        sources=[source]
                    )
                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            print(f"Relation extraction failed: {e}")
            return []


class EntitySummarizer:
    """
    Generate concise summaries of medical entities from multiple sources.
    Used for consolidating information before reasoning.
    """
    
    SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a medical information synthesis expert. Create a concise summary of the given entity.

Guidelines:
1. Synthesize information from multiple sources
2. Focus on clinically relevant facts
3. Resolve contradictions by citing evidence strength
4. Keep summary under 200 words
5. Use clear, precise medical language

Output format:
{{
    "entity_name": "string",
    "summary": "string",
    "key_facts": ["fact1", "fact2", ...],
    "confidence": float
}}"""),
        ("human", "Entity: {entity_name}\nType: {entity_type}\n\nSources:\n{sources}")
    ])
    
    def __init__(self, llm):
        """
        Initialize entity summarizer.
        
        Args:
            llm: LangChain chat model (MedGemma)
        """
        self.llm = llm
        self.chain = self.SUMMARIZATION_PROMPT | self.llm
    
    def summarize(
        self,
        entity: MedicalEntity,
        additional_context: List[str]
    ) -> Dict[str, any]:
        """
        Generate entity summary from entity data and additional context.
        
        Args:
            entity: MedicalEntity to summarize
            additional_context: List of text snippets with additional information
            
        Returns:
            Dictionary with summary, key facts, and confidence
        """
        try:
            sources_text = f"Base description: {entity.description}\n\n"
            sources_text += "\n\n".join([
                f"Source {i+1}: {ctx}"
                for i, ctx in enumerate(additional_context)
            ])
            
            result = self.chain.invoke({
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "sources": sources_text
            })
            
            # Parse JSON response
            import json
            return json.loads(result.content if hasattr(result, "content") else str(result))
            
        except Exception as e:
            print(f"Summarization failed: {e}")
            return {
                "entity_name": entity.name,
                "summary": entity.description,
                "key_facts": [],
                "confidence": entity.confidence
            }
