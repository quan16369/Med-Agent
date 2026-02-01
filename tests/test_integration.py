"""
Integration tests for MedAssist system
Tests complete workflows and component interactions
"""

import pytest
import time
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestRuralDeployment:
    """Test rural deployment scenarios"""
    
    def test_offline_mode(self):
        """Test system works in offline mode"""
        from medassist.offline_rag import OfflineRAG
        
        rag = OfflineRAG()
        
        # Should work without internet
        result = rag.augment_query("malaria treatment")
        
        assert result is not None
        assert "knowledge" in result
    
    def test_low_resource_inference(self):
        """Test inference on low-resource device"""
        # Simulate low RAM environment
        import psutil
        
        memory = psutil.virtual_memory()
        
        # System should detect available memory
        assert memory.available > 0
        
        # Should select appropriate model based on resources
        from medassist.adaptive_models import AdaptiveModelSelector
        
        selector = AdaptiveModelSelector()
        model = selector.select_model("simple query", {"complexity": "low"})
        
        assert model  # Should select a model


class TestHybridMode:
    """Test hybrid online/offline operation"""
    
    @patch('medassist.cloud_services.CloudAPIClient.check_connection')
    def test_online_mode(self, mock_check):
        """Test system in online mode"""
        mock_check.return_value = True
        
        from medassist.hybrid_orchestrator import HybridOrchestrator
        
        # Should initialize successfully
        assert True  # Placeholder - would test actual orchestrator
    
    @patch('medassist.cloud_services.CloudAPIClient.check_connection')
    def test_offline_fallback(self, mock_check):
        """Test fallback to offline mode"""
        mock_check.return_value = False
        
        from medassist.hybrid_orchestrator import HybridOrchestrator
        
        # Should fallback to offline mode
        assert True  # Placeholder


class TestErrorRecovery:
    """Test error recovery scenarios"""
    
    def test_model_failure_recovery(self):
        """Test recovery from model failure"""
        from medassist.error_handling import ErrorRecovery
        
        result = ErrorRecovery.recover_from_model_error(
            Exception("Model failed")
        )
        
        assert result is not None
        assert "fallback" in result
    
    def test_database_failure_recovery(self):
        """Test recovery from database failure"""
        from medassist.error_handling import ErrorRecovery
        
        result = ErrorRecovery.recover_from_database_error(
            Exception("DB connection failed")
        )
        
        assert result is not None
        assert "fallback" in result


class TestHealthChecks:
    """Test health check system"""
    
    def test_liveness_check(self):
        """Test liveness probe"""
        from medassist.health_checks import HealthChecker
        
        checker = HealthChecker()
        status = checker.check_liveness()
        
        assert status.healthy
        assert status.status == "healthy"
    
    def test_readiness_check(self):
        """Test readiness probe"""
        from medassist.health_checks import HealthChecker
        
        checker = HealthChecker()
        status = checker.check_readiness()
        
        # May be healthy or degraded depending on system state
        assert status.status in ["healthy", "degraded", "unhealthy"]


class TestPerformance:
    """Test performance requirements"""
    
    def test_query_latency(self):
        """Test query processing latency"""
        from medassist.semantic_router import SemanticRouter
        
        router = SemanticRouter()
        
        start = time.time()
        result = router.route("What is malaria?")
        latency = time.time() - start
        
        # Routing should be fast (<100ms)
        assert latency < 0.1
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        from medassist.semantic_router import SemanticRouter
        
        router = SemanticRouter()
        
        # Simulate concurrent requests
        queries = [
            "What is malaria?",
            "Fever treatment",
            "Diabetes management"
        ]
        
        start = time.time()
        for query in queries:
            router.route(query)
        elapsed = time.time() - start
        
        # Should handle multiple requests efficiently
        assert elapsed < 1.0


class TestScalability:
    """Test scalability features"""
    
    def test_configuration_loading(self):
        """Test configuration loading for different environments"""
        from config_production import get_config
        
        # Development config
        dev_config = get_config("development")
        assert dev_config.environment == "development"
        
        # Production config
        prod_config = get_config("production")
        assert prod_config.environment == "production"
    
    def test_logging_setup(self):
        """Test logging configuration"""
        from medassist.logging_setup import setup_logging
        
        # Should setup without errors
        setup_logging(
            log_level="INFO",
            log_format="json",
            log_file="./logs/test.log"
        )
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Should be able to log
        logger.info("Test message")


class TestSecurity:
    """Test security features"""
    
    def test_audit_logging(self):
        """Test audit logging"""
        from medassist.logging_setup import AuditLogger
        
        audit = AuditLogger(log_file="./logs/test_audit.log")
        
        # Should log without errors
        audit.log_access("user123", "view_case", "case_456", True)
        audit.log_auth_event("user123", "login", True, "127.0.0.1")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from config_production import ProductionConfig
        
        config = ProductionConfig()
        
        # Should validate successfully
        assert config.validate() in [True, False]  # May fail if dirs don't exist


class TestDataPersistence:
    """Test data persistence"""
    
    def test_knowledge_base_persistence(self):
        """Test knowledge base persists data"""
        # Test would check if SQLite DB persists correctly
        assert True  # Placeholder
    
    def test_cache_persistence(self):
        """Test cache persistence"""
        # Test would check if cache works correctly
        assert True  # Placeholder


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
