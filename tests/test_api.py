"""
Tests for API endpoints.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
    
    def test_liveness_probe(self):
        """Test liveness probe"""
        response = client.get("/health/liveness")
        assert response.status_code == 200
    
    def test_readiness_probe(self):
        """Test readiness probe"""
        response = client.get("/health/readiness")
        # May be 200 or 503 depending on orchestrator state
        assert response.status_code in [200, 503]
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "orchestrator_loaded" in data
    
    def test_query_endpoint_valid(self):
        """Test query endpoint with valid request"""
        request_data = {
            "question": "What causes diabetes and how is it treated?",
            "include_trace": False,
            "include_stats": False
        }
        response = client.post("/query", json=request_data)
        
        # Should return 200 or 503 (if orchestrator not ready)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert "execution_time" in data
    
    def test_query_endpoint_with_trace(self):
        """Test query endpoint with trace"""
        request_data = {
            "question": "What are symptoms of hypertension?",
            "include_trace": True,
            "include_stats": True
        }
        response = client.post("/query", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "trace" in data or "answer" in data
    
    def test_query_endpoint_short_question(self):
        """Test query endpoint with too short question"""
        request_data = {
            "question": "What?",
            "include_trace": False
        }
        response = client.post("/query", json=request_data)
        
        # Should reject short questions
        assert response.status_code == 422
    
    def test_query_endpoint_empty_question(self):
        """Test query endpoint with empty question"""
        request_data = {
            "question": "",
            "include_trace": False
        }
        response = client.post("/query", json=request_data)
        
        # Should reject empty questions
        assert response.status_code == 422
    
    def test_query_endpoint_missing_question(self):
        """Test query endpoint without question field"""
        request_data = {
            "include_trace": False
        }
        response = client.post("/query", json=request_data)
        
        # Should reject missing field
        assert response.status_code == 422
    
    @pytest.mark.slow
    def test_rate_limiting(self):
        """Test rate limiting (marked slow)"""
        # Make many requests
        request_data = {
            "question": "What causes diabetes?"
        }
        
        responses = []
        for _ in range(15):  # Exceed rate limit of 10/min
            response = client.post("/query", json=request_data)
            responses.append(response.status_code)
        
        # Should eventually get 429 (rate limited)
        assert 429 in responses

class TestAPIDocs:
    """Test API documentation"""
    
    def test_openapi_docs(self):
        """Test OpenAPI documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_docs(self):
        """Test ReDoc documentation is available"""
        response = client.get("/redoc")
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
