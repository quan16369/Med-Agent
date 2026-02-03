"""
Integration tests for the agentic medical workflow system.
Tests end-to-end functionality of all components.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.agentic_orchestrator import AgenticMedicalOrchestrator
from medassist.knowledge_graph import MedicalKnowledgeGraph
from medassist.config import get_config
from medassist.exceptions import MedAssistError

class TestAgenticWorkflow:
    """Test agentic workflow end-to-end"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return AgenticMedicalOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes all components"""
        assert orchestrator is not None
        assert orchestrator.kg is not None
        assert orchestrator.graph_retriever is not None
        assert orchestrator.ner is not None
        assert orchestrator.pubmed is not None
        assert orchestrator.orchestrator_agent is not None
    
    def test_simple_medical_query(self, orchestrator):
        """Test basic medical question"""
        question = "What causes diabetes and how is it treated?"
        result = orchestrator.execute_workflow(question)
        
        assert result is not None
        assert "answer" in result
        assert "confidence" in result
        assert len(result["answer"]) > 0
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_symptom_based_query(self, orchestrator):
        """Test symptom-based diagnostic question"""
        question = "Patient has fever and cough. What could be the diagnosis?"
        result = orchestrator.execute_workflow(question)
        
        assert result is not None
        assert "answer" in result
        # Check that answer contains medical reasoning
        assert len(result["answer"]) > 50
    
    def test_treatment_query(self, orchestrator):
        """Test treatment recommendation question"""
        question = "What are the treatment options for hypertension?"
        result = orchestrator.execute_workflow(question)
        
        assert result is not None
        assert "answer" in result
        # Should provide treatment information
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in ["treatment", "medication", "therapy"])
    
    def test_workflow_trace(self, orchestrator):
        """Test that workflow generates trace"""
        question = "What is the relationship between obesity and diabetes?"
        result = orchestrator.execute_workflow(question)
        
        assert "trace" in result
        assert len(result["trace"]) > 0
        # Trace should have agent information
        first_step = result["trace"][0]
        assert "agent" in first_step
        assert "action" in first_step
    
    def test_statistics_generation(self, orchestrator):
        """Test that statistics are generated"""
        question = "What causes asthma?"
        result = orchestrator.execute_workflow(question)
        
        assert "statistics" in result
        stats = result["statistics"]
        assert len(stats) > 0
        # Check statistics structure
        assert all("agent_name" in s for s in stats)
        assert all("calls" in s for s in stats)

class TestKnowledgeGraph:
    """Test knowledge graph functionality"""
    
    @pytest.fixture
    def kg(self):
        """Create knowledge graph instance"""
        return MedicalKnowledgeGraph()
    
    def test_kg_initialization(self, kg):
        """Test knowledge graph initializes"""
        assert kg is not None
    
    def test_add_and_query_entity(self, kg):
        """Test adding and querying entities"""
        # Add test entity
        kg.add_entity("TestDisease", "Disease")
        
        # Query should find it
        results = kg.query_entity("TestDisease")
        assert results is not None
    
    def test_add_relationship(self, kg):
        """Test adding relationships"""
        kg.add_entity("TestDisease1", "Disease")
        kg.add_entity("TestSymptom1", "Symptom")
        kg.add_relationship("TestDisease1", "HAS_SYMPTOM", "TestSymptom1")
        
        # Should be able to query relationship
        results = kg.query_entity("TestDisease1")
        assert results is not None

class TestConfiguration:
    """Test configuration management"""
    
    def test_config_loads(self):
        """Test that configuration loads"""
        config = get_config()
        assert config is not None
        assert config.model is not None
        assert config.retrieval is not None
        assert config.logging is not None
    
    def test_config_values(self):
        """Test configuration has valid values"""
        config = get_config()
        
        # Model config
        assert config.model.device in ["cuda", "cpu"]
        assert config.model.max_length > 0
        
        # Retrieval config
        assert config.retrieval.max_depth > 0
        assert config.retrieval.max_width > 0
        assert 0.0 <= config.retrieval.confidence_threshold <= 1.0

class TestErrorHandling:
    """Test error handling"""
    
    def test_empty_question_handling(self):
        """Test handling of empty questions"""
        orchestrator = AgenticMedicalOrchestrator()
        
        # Empty question should be handled gracefully
        result = orchestrator.execute_workflow("")
        assert result is not None
        # Should still return some response
        assert "answer" in result
    
    def test_invalid_question_handling(self):
        """Test handling of nonsensical questions"""
        orchestrator = AgenticMedicalOrchestrator()
        
        # Nonsensical question
        result = orchestrator.execute_workflow("asdfghjkl qwerty")
        assert result is not None
        assert "answer" in result

@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked slow)"""
    
    def test_query_response_time(self):
        """Test that queries complete in reasonable time"""
        import time
        
        orchestrator = AgenticMedicalOrchestrator()
        question = "What causes diabetes?"
        
        start = time.time()
        result = orchestrator.execute_workflow(question)
        duration = time.time() - start
        
        assert result is not None
        # Should complete within 30 seconds
        assert duration < 30.0
    
    def test_multiple_queries(self):
        """Test multiple sequential queries"""
        orchestrator = AgenticMedicalOrchestrator()
        questions = [
            "What causes diabetes?",
            "How is hypertension treated?",
            "What are symptoms of asthma?"
        ]
        
        for question in questions:
            result = orchestrator.execute_workflow(question)
            assert result is not None
            assert "answer" in result

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
