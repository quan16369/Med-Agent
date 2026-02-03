"""
Health Check Endpoints
For Kubernetes liveness and readiness probes
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    message: str
    details: Dict[str, Any]


class HealthChecker:
    """Health check system"""
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.startup_complete = False
    
    def liveness(self) -> HealthCheckResult:
        """
        Liveness probe - is the application running?
        Returns unhealthy only if application is completely broken
        """
        try:
            # Basic check - can we execute Python code?
            import sys
            python_version = sys.version
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Application is running",
                details={"python_version": python_version}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Liveness check failed: {e}",
                details={}
            )
    
    def readiness(self) -> HealthCheckResult:
        """
        Readiness probe - is the application ready to serve requests?
        Returns unhealthy if dependencies are not available
        """
        checks = {}
        overall_status = HealthStatus.HEALTHY
        
        # Check if startup is complete
        if not self.startup_complete:
            overall_status = HealthStatus.UNHEALTHY
            checks["startup"] = "incomplete"
        else:
            checks["startup"] = "complete"
        
        # Check orchestrator
        if self.orchestrator:
            try:
                stats = self.orchestrator.get_statistics()
                checks["knowledge_graph"] = f"{stats['knowledge_graph']['num_nodes']} nodes"
                checks["agents"] = f"{stats['agent_count']} agents"
            except Exception as e:
                overall_status = HealthStatus.DEGRADED
                checks["orchestrator"] = f"error: {e}"
        else:
            checks["orchestrator"] = "not initialized"
        
        # Check dependencies
        try:
            import torch
            checks["torch"] = "available"
        except ImportError:
            checks["torch"] = "not available"
        
        try:
            import transformers
            checks["transformers"] = "available"
        except ImportError:
            checks["transformers"] = "not available"
            overall_status = HealthStatus.DEGRADED
        
        return HealthCheckResult(
            status=overall_status,
            message=f"Readiness: {overall_status.value}",
            details=checks
        )
    
    def startup(self) -> HealthCheckResult:
        """
        Startup probe - has initialization completed?
        """
        if self.startup_complete:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Startup complete",
                details={"ready": True}
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Startup in progress",
                details={"ready": False}
            )
    
    def mark_startup_complete(self):
        """Mark startup as complete"""
        self.startup_complete = True
        logger.info("Startup marked as complete")


def format_health_response(result: HealthCheckResult) -> Dict:
    """Format health check result as dictionary"""
    return {
        "status": result.status.value,
        "message": result.message,
        "details": result.details
    }
