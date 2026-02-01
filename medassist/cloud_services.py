"""
Cloud Services Integration for Online Mode
Provides enhanced features when internet is available
"""

import requests
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Network connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"  # Slow/intermittent connection


class CloudAPIClient:
    """
    Client for cloud-based medical API services
    Provides enhanced capabilities when online
    """
    
    def __init__(
        self,
        api_endpoint: str = "https://api.medassist.health",
        api_key: Optional[str] = None,
        timeout: int = 10
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.connection_status = ConnectionStatus.OFFLINE
        
        # Cache for offline fallback
        self.cache = {}
        
    def check_connectivity(self) -> ConnectionStatus:
        """Check internet connectivity and API availability"""
        try:
            response = requests.get(
                f"{self.api_endpoint}/health",
                timeout=3
            )
            if response.status_code == 200:
                self.connection_status = ConnectionStatus.ONLINE
                return ConnectionStatus.ONLINE
            else:
                self.connection_status = ConnectionStatus.DEGRADED
                return ConnectionStatus.DEGRADED
        except Exception as e:
            logger.debug(f"Connectivity check failed: {e}")
            self.connection_status = ConnectionStatus.OFFLINE
            return ConnectionStatus.OFFLINE
    
    def get_drug_interactions(
        self,
        drug_list: List[str]
    ) -> Optional[Dict]:
        """
        Query cloud drug interaction database (more comprehensive than local)
        Falls back to cache if offline
        """
        cache_key = f"interactions_{'-'.join(sorted(drug_list))}"
        
        try:
            if self.connection_status == ConnectionStatus.ONLINE:
                response = requests.post(
                    f"{self.api_endpoint}/interactions",
                    json={"drugs": drug_list},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.cache[cache_key] = result  # Cache for offline use
                    return result
            
            # Fallback to cache
            if cache_key in self.cache:
                logger.info("Using cached drug interaction data")
                return self.cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning(f"Cloud API call failed: {e}")
            return self.cache.get(cache_key)
    
    def get_latest_guidelines(
        self,
        topic: str,
        region: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Fetch latest clinical guidelines from cloud
        Can be region-specific (e.g., malaria guidelines for Southeast Asia)
        """
        try:
            if self.connection_status == ConnectionStatus.ONLINE:
                params = {"topic": topic}
                if region:
                    params["region"] = region
                
                response = requests.get(
                    f"{self.api_endpoint}/guidelines/latest",
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch guidelines: {e}")
            return None
    
    def query_medical_llm(
        self,
        query: str,
        model: str = "medgemma-7b",
        context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Query larger cloud-hosted model for complex cases
        Useful when local model confidence is low
        """
        try:
            if self.connection_status != ConnectionStatus.ONLINE:
                return None
            
            payload = {
                "query": query,
                "model": model,
                "context": context or {}
            }
            
            response = requests.post(
                f"{self.api_endpoint}/inference",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30  # Longer timeout for inference
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.warning(f"Cloud inference failed: {e}")
            return None
    
    def get_outbreak_alerts(
        self,
        location: str,
        radius_km: int = 50
    ) -> Optional[List[Dict]]:
        """
        Get real-time disease outbreak alerts for region
        Helps with diagnosis (e.g., dengue outbreak nearby)
        """
        try:
            if self.connection_status == ConnectionStatus.ONLINE:
                response = requests.get(
                    f"{self.api_endpoint}/outbreaks",
                    params={"location": location, "radius": radius_km},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json().get("alerts", [])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch outbreak alerts: {e}")
            return None
    
    def upload_anonymized_case(
        self,
        case_data: Dict,
        outcome: Optional[str] = None
    ) -> bool:
        """
        Upload anonymized case for research/learning
        Helps improve system over time
        """
        try:
            if self.connection_status != ConnectionStatus.ONLINE:
                return False
            
            # Ensure no PII
            anonymized = self._anonymize_case(case_data)
            
            payload = {
                "case": anonymized,
                "outcome": outcome,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            response = requests.post(
                f"{self.api_endpoint}/cases/upload",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Failed to upload case: {e}")
            return False
    
    def _anonymize_case(self, case: Dict) -> Dict:
        """Remove all PII from case data"""
        anonymized = case.copy()
        
        # Remove PII fields
        pii_fields = [
            "name", "patient_name", "id", "patient_id",
            "phone", "email", "address", "location"
        ]
        
        for field in pii_fields:
            if field in anonymized:
                del anonymized[field]
        
        # Hash any remaining identifiers
        if "case_id" in anonymized:
            anonymized["case_id"] = f"anon_{hash(anonymized['case_id'])}"
        
        return anonymized


class TelemetryService:
    """
    Telemetry and analytics service
    Tracks system performance and usage patterns
    """
    
    def __init__(
        self,
        endpoint: str = "https://telemetry.medassist.health",
        enabled: bool = True
    ):
        self.endpoint = endpoint
        self.enabled = enabled
        self.buffer = []  # Buffer events when offline
        self.max_buffer_size = 1000
    
    def log_event(
        self,
        event_type: str,
        data: Dict,
        priority: str = "normal"
    ):
        """Log telemetry event"""
        if not self.enabled:
            return
        
        event = {
            "type": event_type,
            "data": data,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Try immediate send if high priority
        if priority == "high":
            self._send_event(event)
        else:
            self.buffer.append(event)
            
            # Flush buffer if full
            if len(self.buffer) >= self.max_buffer_size:
                self.flush_buffer()
    
    def flush_buffer(self):
        """Send buffered events to server"""
        if not self.buffer:
            return
        
        try:
            response = requests.post(
                f"{self.endpoint}/events",
                json={"events": self.buffer},
                timeout=5
            )
            
            if response.status_code == 200:
                self.buffer.clear()
                logger.info(f"Flushed {len(self.buffer)} telemetry events")
        
        except Exception as e:
            logger.debug(f"Telemetry flush failed: {e}")
    
    def _send_event(self, event: Dict):
        """Send single event immediately"""
        try:
            requests.post(
                f"{self.endpoint}/event",
                json=event,
                timeout=2
            )
        except Exception:
            pass  # Silent fail for telemetry


class ModelServingService:
    """
    Cloud model serving for larger models
    Falls back to local model when offline
    """
    
    def __init__(
        self,
        endpoint: str = "https://inference.medassist.health",
        api_key: Optional[str] = None
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.available_models = [
            "medgemma-1b",  # Local
            "medgemma-2b",  # Local
            "medgemma-7b",  # Cloud-preferred
            "gemma-2-9b-it",  # Cloud-only
            "medgemma-27b"  # Cloud-only (hypothetical larger model)
        ]
    
    def query_model(
        self,
        prompt: str,
        model: str = "medgemma-7b",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Optional[Dict]:
        """
        Query cloud-hosted model
        Returns None if offline (caller should use local model)
        """
        try:
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.debug(f"Cloud inference unavailable: {e}")
            return None
    
    def get_model_status(self) -> Dict:
        """Check which models are available locally vs cloud"""
        return {
            "local": ["medgemma-1b", "medgemma-2b"],
            "cloud": ["medgemma-7b", "gemma-2-9b-it", "medgemma-27b"],
            "recommended_offline": "medgemma-2b",
            "recommended_online": "medgemma-7b"
        }


class HybridKnowledgeService:
    """
    Hybrid knowledge service
    Combines local KB with cloud-based updates
    """
    
    def __init__(
        self,
        cloud_api: CloudAPIClient,
        local_kb_path: str = "./data/medical_kb"
    ):
        self.cloud_api = cloud_api
        self.local_kb_path = local_kb_path
    
    def search(
        self,
        query: str,
        prefer_cloud: bool = True
    ) -> List[Dict]:
        """
        Search knowledge base
        Uses cloud for latest data when available
        """
        results = []
        
        # Try cloud first if preferred and online
        if prefer_cloud and self.cloud_api.connection_status == ConnectionStatus.ONLINE:
            try:
                cloud_results = self._search_cloud(query)
                if cloud_results:
                    results.extend(cloud_results)
            except Exception as e:
                logger.warning(f"Cloud search failed: {e}")
        
        # Always search local as backup
        local_results = self._search_local(query)
        results.extend(local_results)
        
        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r.get('type')}_{r.get('id', '')}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results
    
    def _search_cloud(self, query: str) -> List[Dict]:
        """Search cloud knowledge base"""
        # Placeholder - would make API call
        return []
    
    def _search_local(self, query: str) -> List[Dict]:
        """Search local knowledge base"""
        # Placeholder - would search local DB
        return []


class AutoScalingCoordinator:
    """
    Coordinates multiple clinic instances
    Enables horizontal scaling
    """
    
    def __init__(
        self,
        coordinator_url: str = "https://coordinator.medassist.health",
        clinic_id: str = None
    ):
        self.coordinator_url = coordinator_url
        self.clinic_id = clinic_id or self._generate_clinic_id()
        self.registered = False
    
    def register_clinic(
        self,
        location: str,
        capabilities: List[str]
    ) -> bool:
        """Register this clinic instance with coordinator"""
        try:
            response = requests.post(
                f"{self.coordinator_url}/register",
                json={
                    "clinic_id": self.clinic_id,
                    "location": location,
                    "capabilities": capabilities,
                    "timestamp": datetime.utcnow().isoformat()
                },
                timeout=10
            )
            
            self.registered = response.status_code == 200
            return self.registered
            
        except Exception as e:
            logger.warning(f"Clinic registration failed: {e}")
            return False
    
    def share_load_metrics(
        self,
        metrics: Dict
    ):
        """Share load metrics with coordinator for load balancing"""
        if not self.registered:
            return
        
        try:
            requests.post(
                f"{self.coordinator_url}/metrics/{self.clinic_id}",
                json=metrics,
                timeout=5
            )
        except Exception:
            pass  # Silent fail
    
    def get_referral_recommendations(
        self,
        case_severity: str
    ) -> Optional[List[Dict]]:
        """
        Get recommendations for patient referral
        Coordinator knows which facilities have capacity
        """
        try:
            response = requests.get(
                f"{self.coordinator_url}/referrals",
                params={
                    "from_clinic": self.clinic_id,
                    "severity": case_severity
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("recommendations", [])
            
            return None
            
        except Exception:
            return None
    
    def _generate_clinic_id(self) -> str:
        """Generate unique clinic ID"""
        import uuid
        return f"clinic_{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Cloud Services Demo")
    print("="*60)
    
    # Test connectivity
    client = CloudAPIClient()
    status = client.check_connectivity()
    print(f"\nConnection status: {status.value}")
    
    # Test drug interactions (would use cached data offline)
    interactions = client.get_drug_interactions(["warfarin", "aspirin"])
    print(f"\nDrug interactions: {interactions is not None}")
    
    # Test telemetry
    telemetry = TelemetryService()
    telemetry.log_event("system_start", {"version": "1.0"})
    print(f"\nTelemetry events buffered: {len(telemetry.buffer)}")
