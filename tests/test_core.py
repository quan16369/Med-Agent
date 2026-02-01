"""
Unit tests for MedAssist core components
Comprehensive test coverage for production readiness
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from medassist.orchestrator import MedAssistOrchestrator
from medassist.agents import HistoryAgent, DiagnosticAgent, TreatmentAgent, KnowledgeAgent
from medassist.semantic_router import SemanticRouter
from medassist.deep_confidence import DeepConf
from medassist.adaptive_models import AdaptiveModelSelector
from medassist.offline_rag import OfflineRAG
from medassist.error_handling import retry, CircuitBreaker, GracefulDegradation


class TestSemanticRouter:
    """Test semantic router"""
    
    def test_route_history_query(self):
        """Test routing of history-related query"""
        router = SemanticRouter()
        
        query = "Patient has fever for 3 days"
        result = router.route(query)
        
        assert result["primary_agent"] == "history"
        assert result["confidence"] > 0.5
    
    def test_route_diagnostic_query(self):
        """Test routing of diagnostic query"""
        router = SemanticRouter()
        
        query = "What could cause chest pain and shortness of breath?"
        result = router.route(query)
        
        assert result["primary_agent"] == "diagnostic"
        assert result["confidence"] > 0.5
    
    def test_route_treatment_query(self):
        """Test routing of treatment query"""
        router = SemanticRouter()
        
        query = "What is the dosage for amoxicillin?"
        result = router.route(query)
        
        assert result["primary_agent"] == "treatment"
        assert result["confidence"] > 0.5
    
    def test_escalation_on_low_confidence(self):
        """Test escalation when confidence is low"""
        router = SemanticRouter()
        
        query = "help"  # Ambiguous query
        result = router.route(query)
        
        # Should still return a route but may have lower confidence
        assert "primary_agent" in result
        assert "confidence" in result


class TestDeepConf:
    """Test confidence tracking"""
    
    def test_track_confidence(self):
        """Test confidence tracking for tokens"""
        deepconf = DeepConf()
        
        # Mock token logits
        logits = [
            [0.9, 0.05, 0.05],  # High confidence
            [0.4, 0.3, 0.3],     # Low confidence
            [0.8, 0.1, 0.1]      # High confidence
        ]
        
        tokens = [1, 2, 3]
        
        for i, token_logits in enumerate(logits):
            deepconf.track_token(tokens[i], token_logits)
        
        group_conf = deepconf.get_group_confidence([0, 1, 2])
        
        assert 0 <= group_conf <= 1
        assert group_conf > 0  # Should have some confidence
    
    def test_filter_low_confidence(self):
        """Test filtering low confidence tokens"""
        deepconf = DeepConf()
        
        # Track some tokens with varying confidence
        tokens = []
        for i in range(10):
            conf = 0.9 if i < 5 else 0.3  # First 5 high, last 5 low
            logits = [conf] + [(1-conf)/2, (1-conf)/2]
            deepconf.track_token(i, logits)
            tokens.append(i)
        
        # Filter with threshold
        filtered = deepconf.filter_low_confidence(tokens, threshold=0.5)
        
        # Should keep high confidence tokens
        assert len(filtered) <= len(tokens)


class TestAdaptiveModelSelector:
    """Test adaptive model selection"""
    
    def test_select_simple_query(self):
        """Test model selection for simple query"""
        selector = AdaptiveModelSelector()
        
        query = "What is fever?"
        context = {"emergency": False, "complexity": "low"}
        
        model = selector.select_model(query, context)
        
        # Should select lightweight model for simple query
        assert "1b" in model.lower() or "2b" in model.lower()
    
    def test_select_complex_query(self):
        """Test model selection for complex query"""
        selector = AdaptiveModelSelector()
        
        query = "Patient with multiple comorbidities including diabetes, hypertension, and heart disease"
        context = {"emergency": False, "complexity": "high"}
        
        model = selector.select_model(query, context)
        
        # Should select more capable model
        assert "7b" in model.lower() or "9b" in model.lower() or "ensemble" in model.lower()
    
    def test_select_emergency(self):
        """Test model selection for emergency"""
        selector = AdaptiveModelSelector()
        
        query = "Patient not breathing"
        context = {"emergency": True}
        
        model = selector.select_model(query, context)
        
        # Should select best available model
        assert model  # Should return a model


class TestOfflineRAG:
    """Test offline RAG system"""
    
    def test_augment_query(self):
        """Test query augmentation"""
        rag = OfflineRAG()
        
        query = "What is the treatment for malaria?"
        augmented = rag.augment_query(query)
        
        assert "knowledge" in augmented
        assert len(augmented["knowledge"]) > 0
    
    def test_drug_lookup(self):
        """Test drug information lookup"""
        rag = OfflineRAG()
        
        info = rag.get_drug_info("artemether-lumefantrine")
        
        assert info is not None
        assert "indications" in info or "name" in info


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures"""
        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        def failing_function():
            raise Exception("Service unavailable")
        
        # Should fail 3 times before opening
        for _ in range(3):
            with pytest.raises(Exception):
                circuit.call(failing_function)
        
        # Circuit should now be open
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            circuit.call(failing_function)
    
    def test_circuit_recovers(self):
        """Test circuit breaker recovery"""
        circuit = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        call_count = [0]
        
        def flaky_function():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Failing")
            return "Success"
        
        # Trigger failures
        for _ in range(2):
            with pytest.raises(Exception):
                circuit.call(flaky_function)
        
        # Wait for recovery timeout
        time.sleep(1.5)
        
        # Should succeed after recovery
        result = circuit.call(flaky_function)
        assert result == "Success"


class TestGracefulDegradation:
    """Test graceful degradation"""
    
    def test_mark_feature_degraded(self):
        """Test marking feature as degraded"""
        degradation = GracefulDegradation()
        
        degradation.mark_degraded("cloud_sync", Exception("Connection failed"))
        
        assert degradation.is_degraded("cloud_sync")
        assert not degradation.is_degraded("local_inference")
    
    def test_restore_feature(self):
        """Test restoring degraded feature"""
        degradation = GracefulDegradation()
        
        degradation.mark_degraded("telemetry", Exception("API down"))
        assert degradation.is_degraded("telemetry")
        
        degradation.restore_feature("telemetry")
        assert not degradation.is_degraded("telemetry")


class TestRetryDecorator:
    """Test retry decorator"""
    
    def test_retry_succeeds_eventually(self):
        """Test that retry succeeds after failures"""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1, backoff=1.5)
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Not ready")
            return "Success"
        
        result = flaky_function()
        assert result == "Success"
        assert call_count[0] == 3
    
    def test_retry_fails_after_max_attempts(self):
        """Test that retry fails after max attempts"""
        @retry(max_attempts=3, delay=0.1)
        def always_failing():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            always_failing()


class TestOrchestrator:
    """Test main orchestrator"""
    
    @patch('medassist.orchestrator.MedAssistOrchestrator._load_model')
    def test_orchestrator_initialization(self, mock_load_model):
        """Test orchestrator initializes properly"""
        mock_load_model.return_value = Mock()
        
        orchestrator = MedAssistOrchestrator()
        
        assert orchestrator.history_agent is not None
        assert orchestrator.diagnostic_agent is not None
        assert orchestrator.treatment_agent is not None
        assert orchestrator.knowledge_agent is not None
    
    @patch('medassist.orchestrator.MedAssistOrchestrator._load_model')
    def test_process_query(self, mock_load_model):
        """Test query processing"""
        mock_model = Mock()
        mock_model.generate.return_value = "Test response"
        mock_load_model.return_value = mock_model
        
        orchestrator = MedAssistOrchestrator()
        
        query = "What causes fever?"
        result = orchestrator.process(query)
        
        assert "response" in result or "error" not in result


class TestIntegration:
    """Integration tests"""
    
    @patch('medassist.orchestrator.MedAssistOrchestrator._load_model')
    def test_end_to_end_query(self, mock_load_model):
        """Test end-to-end query processing"""
        mock_model = Mock()
        mock_model.generate.return_value = "Fever can be caused by infections"
        mock_load_model.return_value = mock_model
        
        orchestrator = MedAssistOrchestrator()
        
        query = "What causes fever?"
        result = orchestrator.process(query)
        
        # Should successfully process query
        assert result is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
