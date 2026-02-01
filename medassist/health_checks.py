"""
Health check endpoints for monitoring and orchestration
Implements readiness and liveness probes
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status"""
    healthy: bool
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    checks: Dict[str, Any]
    version: str = "1.0.0"


@dataclass
class ComponentHealth:
    """Individual component health"""
    name: str
    status: str  # pass, warn, fail
    message: str
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """
    Production health checker
    
    Implements:
    - Liveness probe: Is the application running?
    - Readiness probe: Can the application serve requests?
    - Startup probe: Has the application started successfully?
    """
    
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.last_check = None
        self.check_history: List[HealthStatus] = []
        self.max_history = 100
        
        # Component checkers
        self.component_checkers = {}
        
        # Thresholds
        self.cpu_threshold = 90.0  # percent
        self.memory_threshold = 90.0  # percent
        self.disk_threshold = 90.0  # percent
        
        # Startup grace period
        self.startup_grace_period = timedelta(minutes=5)
    
    def register_component_checker(self, name: str, checker_func):
        """Register a component health checker"""
        self.component_checkers[name] = checker_func
        logger.info(f"Registered health checker: {name}")
    
    def check_liveness(self) -> HealthStatus:
        """
        Liveness probe - Is the application alive?
        Should only fail if application is deadlocked/crashed
        """
        checks = {
            "alive": self._check_alive(),
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds()
        }
        
        healthy = checks["alive"]["status"] == "pass"
        
        status = HealthStatus(
            healthy=healthy,
            status="healthy" if healthy else "unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            checks=checks
        )
        
        self._record_check(status)
        return status
    
    def check_readiness(self) -> HealthStatus:
        """
        Readiness probe - Can the application serve requests?
        Should fail if dependencies are unavailable
        """
        checks = {
            "alive": self._check_alive(),
            "database": self._check_database(),
            "model": self._check_model(),
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory()
        }
        
        # Add custom component checks
        for name, checker in self.component_checkers.items():
            try:
                checks[name] = asdict(checker())
            except Exception as e:
                checks[name] = asdict(ComponentHealth(
                    name=name,
                    status="fail",
                    message=f"Check failed: {str(e)}"
                ))
        
        # Determine overall health
        failed_checks = [k for k, v in checks.items() if v.get("status") == "fail"]
        warn_checks = [k for k, v in checks.items() if v.get("status") == "warn"]
        
        if failed_checks:
            healthy = False
            status = "unhealthy"
        elif warn_checks:
            healthy = True
            status = "degraded"
        else:
            healthy = True
            status = "healthy"
        
        health_status = HealthStatus(
            healthy=healthy,
            status=status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            checks=checks
        )
        
        self._record_check(health_status)
        return health_status
    
    def check_startup(self) -> HealthStatus:
        """
        Startup probe - Has the application started successfully?
        Should fail if startup is taking too long
        """
        uptime = datetime.utcnow() - self.startup_time
        
        checks = {
            "alive": self._check_alive(),
            "uptime_seconds": uptime.total_seconds(),
            "startup_complete": uptime > self.startup_grace_period
        }
        
        # Check critical components
        if uptime > timedelta(seconds=30):
            checks["database"] = self._check_database()
            checks["model"] = self._check_model()
        
        failed_checks = [k for k, v in checks.items() 
                        if isinstance(v, dict) and v.get("status") == "fail"]
        
        healthy = not failed_checks
        
        status = HealthStatus(
            healthy=healthy,
            status="healthy" if healthy else "unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            checks=checks
        )
        
        return status
    
    def _check_alive(self) -> Dict[str, Any]:
        """Check if application is alive"""
        return asdict(ComponentHealth(
            name="alive",
            status="pass",
            message="Application is running",
            latency_ms=0.0
        ))
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        start = time.time()
        
        try:
            # Try to import and check database
            from medassist.knowledge_sync import KnowledgeSync
            
            kb = KnowledgeSync(db_path="./data/medical_kb/knowledge.db")
            
            # Simple query to test connection
            result = kb.search_guidelines("test", limit=1)
            
            latency_ms = (time.time() - start) * 1000
            
            return asdict(ComponentHealth(
                name="database",
                status="pass",
                message="Database accessible",
                latency_ms=latency_ms
            ))
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Database check failed: {e}")
            
            return asdict(ComponentHealth(
                name="database",
                status="fail",
                message=f"Database check failed: {str(e)}",
                latency_ms=latency_ms
            ))
    
    def _check_model(self) -> Dict[str, Any]:
        """Check model availability"""
        start = time.time()
        
        try:
            # Check if model directory exists
            model_cache = os.getenv("MODEL_CACHE_DIR", "./models/cache")
            
            if not os.path.exists(model_cache):
                return asdict(ComponentHealth(
                    name="model",
                    status="warn",
                    message="Model cache directory not found",
                    latency_ms=(time.time() - start) * 1000
                ))
            
            latency_ms = (time.time() - start) * 1000
            
            return asdict(ComponentHealth(
                name="model",
                status="pass",
                message="Model available",
                latency_ms=latency_ms
            ))
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            
            return asdict(ComponentHealth(
                name="model",
                status="fail",
                message=f"Model check failed: {str(e)}",
                latency_ms=latency_ms
            ))
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            percent_used = disk.percent
            
            if percent_used > self.disk_threshold:
                status = "fail"
                message = f"Disk usage critical: {percent_used:.1f}%"
            elif percent_used > 80:
                status = "warn"
                message = f"Disk usage high: {percent_used:.1f}%"
            else:
                status = "pass"
                message = f"Disk usage normal: {percent_used:.1f}%"
            
            return asdict(ComponentHealth(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "percent_used": percent_used,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            ))
        except Exception as e:
            return asdict(ComponentHealth(
                name="disk_space",
                status="fail",
                message=f"Disk check failed: {str(e)}"
            ))
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            if percent_used > self.memory_threshold:
                status = "fail"
                message = f"Memory usage critical: {percent_used:.1f}%"
            elif percent_used > 80:
                status = "warn"
                message = f"Memory usage high: {percent_used:.1f}%"
            else:
                status = "pass"
                message = f"Memory usage normal: {percent_used:.1f}%"
            
            return asdict(ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "percent_used": percent_used,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            ))
        except Exception as e:
            return asdict(ComponentHealth(
                name="memory",
                status="fail",
                message=f"Memory check failed: {str(e)}"
            ))
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.cpu_threshold:
                status = "fail"
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = "warn"
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = "pass"
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return asdict(ComponentHealth(
                name="cpu",
                status=status,
                message=message,
                details={
                    "percent_used": cpu_percent,
                    "core_count": psutil.cpu_count()
                }
            ))
        except Exception as e:
            return asdict(ComponentHealth(
                name="cpu",
                status="fail",
                message=f"CPU check failed: {str(e)}"
            ))
    
    def _record_check(self, status: HealthStatus):
        """Record health check in history"""
        self.check_history.append(status)
        
        # Keep only recent checks
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
        
        self.last_check = status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        if not self.check_history:
            return {}
        
        # Calculate uptime
        uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        
        # Calculate success rate
        recent_checks = self.check_history[-20:]
        healthy_count = sum(1 for c in recent_checks if c.healthy)
        success_rate = healthy_count / len(recent_checks) if recent_checks else 0
        
        return {
            "uptime_seconds": uptime,
            "success_rate": success_rate,
            "total_checks": len(self.check_history),
            "last_check": asdict(self.last_check) if self.last_check else None
        }


class PeriodicHealthChecker:
    """Run health checks periodically in background"""
    
    def __init__(self, health_checker: HealthChecker, interval_seconds: int = 60):
        self.health_checker = health_checker
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
    
    def start(self):
        """Start periodic health checks"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_checks, daemon=True)
        self.thread.start()
        logger.info(f"Started periodic health checks (interval={self.interval_seconds}s)")
    
    def stop(self):
        """Stop periodic health checks"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Stopped periodic health checks")
    
    def _run_checks(self):
        """Run checks periodically"""
        while self.running:
            try:
                status = self.health_checker.check_readiness()
                
                if not status.healthy:
                    logger.warning(f"Health check failed: {status.status}")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.interval_seconds)


if __name__ == "__main__":
    # Demo
    print("Health Check Demo")
    print("="*60)
    
    checker = HealthChecker()
    
    # Liveness check
    liveness = checker.check_liveness()
    print(f"\nLiveness: {liveness.status}")
    print(f"  Healthy: {liveness.healthy}")
    
    # Readiness check
    readiness = checker.check_readiness()
    print(f"\nReadiness: {readiness.status}")
    print(f"  Healthy: {readiness.healthy}")
    print(f"  Checks:")
    for name, result in readiness.checks.items():
        if isinstance(result, dict):
            print(f"    {name}: {result.get('status')} - {result.get('message')}")
    
    # Metrics
    metrics = checker.get_metrics()
    print(f"\nMetrics:")
    print(f"  Uptime: {metrics['uptime_seconds']:.1f}s")
    print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
