"""
Unit tests for individual components.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.medical_ner import MedicalNER
from medassist.pubmed_retrieval import PubMedRetriever
from medassist.graph_retrieval import GraphConditionedRetriever
from medassist.knowledge_graph import MedicalKnowledgeGraph

class TestMedicalNER:
    """Test medical named entity recognition"""
    
    @pytest.fixture
    def ner(self):
        """Create NER instance"""
        return MedicalNER()
    
    def test_ner_initialization(self, ner):
        """Test NER initializes"""
        assert ner is not None
    
    def test_extract_entities_basic(self, ner):
        """Test basic entity extraction"""
        text = "Patient has diabetes and high blood pressure."
        entities = ner.extract_entities(text)
        
        assert entities is not None
        assert len(entities) > 0
        # Should find medical terms
        entity_texts = [e.text.lower() for e in entities]
        assert any("diabetes" in t or "blood pressure" in t for t in entity_texts)
    
    def test_extract_entities_symptoms(self, ner):
        """Test symptom extraction"""
        text = "Patient complains of fever, headache, and fatigue."
        entities = ner.extract_entities(text)
        
        assert len(entities) > 0
        symptoms = [e for e in entities if e.label == "Symptom"]
        assert len(symptoms) > 0

class TestPubMedRetriever:
    """Test PubMed retrieval"""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever instance"""
        return PubMedRetriever()
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initializes"""
        assert retriever is not None
    
    @pytest.mark.slow
    def test_search_pubmed(self, retriever):
        """Test PubMed search (requires internet)"""
        query = "diabetes treatment"
        results = retriever.search(query, max_results=3)
        
        # Should return some results
        assert results is not None
        assert isinstance(results, list)
        # Results should have structure
        if len(results) > 0:
            assert "title" in results[0]

class TestGraphRetriever:
    """Test graph retrieval"""
    
    @pytest.fixture
    def kg(self):
        """Create knowledge graph"""
        kg = MedicalKnowledgeGraph()
        # Add test data
        kg.add_entity("TestDisease", "Disease")
        kg.add_entity("TestSymptom", "Symptom")
        kg.add_entity("TestTreatment", "Treatment")
        kg.add_relationship("TestDisease", "HAS_SYMPTOM", "TestSymptom")
        kg.add_relationship("TestDisease", "TREATED_BY", "TestTreatment")
        return kg
    
    @pytest.fixture
    def retriever(self, kg):
        """Create retriever instance"""
        return GraphConditionedRetriever(kg)
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initializes"""
        assert retriever is not None
    
    def test_bfs_traversal(self, retriever):
        """Test BFS traversal"""
        results = retriever.bfs_retrieve("TestDisease", max_depth=2, max_width=5)
        
        assert results is not None
        assert len(results) > 0
        # Should find connected nodes
        result_names = [r["name"] for r in results]
        assert "TestDisease" in result_names
    
    def test_dfs_traversal(self, retriever):
        """Test DFS traversal"""
        results = retriever.dfs_retrieve("TestDisease", max_depth=2)
        
        assert results is not None
        assert len(results) > 0

class TestKnowledgeGraphOperations:
    """Test knowledge graph operations"""
    
    @pytest.fixture
    def kg(self):
        """Create empty knowledge graph"""
        return MedicalKnowledgeGraph()
    
    def test_add_entity(self, kg):
        """Test adding entity"""
        kg.add_entity("TestEntity", "Disease")
        result = kg.query_entity("TestEntity")
        assert result is not None
    
    def test_add_multiple_entities(self, kg):
        """Test adding multiple entities"""
        kg.add_entity("Entity1", "Disease")
        kg.add_entity("Entity2", "Symptom")
        kg.add_entity("Entity3", "Treatment")
        
        # All should be queryable
        assert kg.query_entity("Entity1") is not None
        assert kg.query_entity("Entity2") is not None
        assert kg.query_entity("Entity3") is not None
    
    def test_add_relationship(self, kg):
        """Test adding relationship"""
        kg.add_entity("Disease1", "Disease")
        kg.add_entity("Symptom1", "Symptom")
        kg.add_relationship("Disease1", "HAS_SYMPTOM", "Symptom1")
        
        # Relationship should be reflected in query
        result = kg.query_entity("Disease1")
        assert result is not None
    
    def test_query_nonexistent(self, kg):
        """Test querying non-existent entity"""
        result = kg.query_entity("NonExistentEntity")
        # Should handle gracefully
        assert result is None or result == {}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
