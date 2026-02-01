"""
Hybrid Orchestrator - Seamlessly switches between Online and Offline modes
Provides best experience regardless of connectivity
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import time

from .orchestrator import MedAssistOrchestrator
from .cloud_services import (
    CloudAPIClient,
    ConnectionStatus,
    TelemetryService,
    ModelServingService,
    HybridKnowledgeService,
    AutoScalingCoordinator
)
from .offline_rag import OfflineRAG
from .knowledge_sync import KnowledgeDatabase

logger = logging.getLogger(__name__)


class OperationalMode:
    """Current operational mode"""
    ONLINE = "online"  # Full cloud features
    OFFLINE = "offline"  # Local-only operation
    HYBRID = "hybrid"  # Intelligent mix
    DEGRADED = "degraded"  # Limited connectivity


class HybridOrchestrator(MedAssistOrchestrator):
    """
    Enhanced orchestrator with online/offline capabilities
    Automatically adapts to network conditions
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        device: str = "auto",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        rural_mode: bool = False,
        offline_mode: bool = False,
        cloud_api_key: Optional[str] = None,
        clinic_id: Optional[str] = None,
        auto_mode_switch: bool = True
    ):
        """
        Initialize hybrid orchestrator
        
        Args:
            model_name: Local model to use
            device: Device for local model
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            rural_mode: Enable rural optimizations
            offline_mode: Force offline mode
            cloud_api_key: API key for cloud services
            clinic_id: Unique clinic identifier
            auto_mode_switch: Automatically switch modes based on connectivity
        """
        # Initialize base orchestrator
        super().__init__(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            rural_mode=rural_mode,
            offline_mode=offline_mode
        )
        
        self.cloud_api_key = cloud_api_key
        self.clinic_id = clinic_id
        self.auto_mode_switch = auto_mode_switch
        
        # Initialize cloud services
        self.cloud_api = CloudAPIClient(api_key=cloud_api_key) if not offline_mode else None
        self.telemetry = TelemetryService(enabled=not offline_mode)
        self.model_serving = ModelServingService(api_key=cloud_api_key) if not offline_mode else None
        self.coordinator = AutoScalingCoordinator(clinic_id=clinic_id) if not offline_mode else None
        
        # Enhanced knowledge service
        if self.cloud_api and self.rag:
            self.hybrid_kb = HybridKnowledgeService(
                cloud_api=self.cloud_api,
                local_kb_path="./data/medical_kb"
            )
        else:
            self.hybrid_kb = None
        
        # Operational mode
        self.current_mode = OperationalMode.OFFLINE if offline_mode else OperationalMode.HYBRID
        self.mode_override = offline_mode  # Manual override
        
        # Connectivity monitoring
        self.last_connectivity_check = None
        self.connectivity_check_interval = 60  # Check every minute
        
        # Performance tracking
        self.online_request_count = 0
        self.offline_request_count = 0
        self.mode_switches = 0
        
        # Start background monitoring if auto-switch enabled
        if auto_mode_switch and not offline_mode:
            self._start_connectivity_monitor()
        
        logger.info(f"Hybrid orchestrator initialized (Mode: {self.current_mode}, Auto-switch: {auto_mode_switch})")
    
    def _start_connectivity_monitor(self):
        """Start background thread to monitor connectivity"""
        def monitor():
            while True:
                try:
                    self._check_and_update_mode()
                    time.sleep(self.connectivity_check_interval)
                except Exception as e:
                    logger.error(f"Connectivity monitor error: {e}")
                    time.sleep(self.connectivity_check_interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Connectivity monitor started")
    
    def _check_and_update_mode(self):
        """Check connectivity and update operational mode"""
        if self.mode_override or not self.cloud_api:
            return
        
        status = self.cloud_api.check_connectivity()
        self.last_connectivity_check = datetime.now()
        
        old_mode = self.current_mode
        
        if status == ConnectionStatus.ONLINE:
            self.current_mode = OperationalMode.ONLINE
        elif status == ConnectionStatus.DEGRADED:
            self.current_mode = OperationalMode.DEGRADED
        else:
            self.current_mode = OperationalMode.OFFLINE
        
        if old_mode != self.current_mode:
            self.mode_switches += 1
            logger.info(f"Mode switched: {old_mode} -> {self.current_mode}")
            self.telemetry.log_event(
                "mode_switch",
                {"from": old_mode, "to": self.current_mode},
                priority="high"
            )
    
    def process_case(
        self,
        case: Dict,
        workflow: str = "standard_diagnosis",
        prefer_cloud: bool = True
    ) -> Dict:
        """
        Process patient case with hybrid online/offline approach
        
        Args:
            case: Patient case data
            workflow: Workflow type
            prefer_cloud: Prefer cloud processing when available
        
        Returns:
            Assessment results
        """
        start_time = time.time()
        
        # Check mode
        if self.auto_mode_switch:
            self._check_and_update_mode()
        
        # Determine processing strategy
        use_cloud = (
            self.current_mode in [OperationalMode.ONLINE, OperationalMode.HYBRID]
            and prefer_cloud
            and not self.mode_override
        )
        
        # Log telemetry
        self.telemetry.log_event(
            "case_start",
            {
                "mode": self.current_mode,
                "use_cloud": use_cloud,
                "workflow": workflow
            }
        )
        
        try:
            if use_cloud:
                result = self._process_case_hybrid(case, workflow)
                self.online_request_count += 1
            else:
                result = super().process_case(case, workflow)
                self.offline_request_count += 1
            
            # Add metadata
            result["processing_metadata"] = {
                "mode": self.current_mode,
                "used_cloud": use_cloud,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Upload anonymized case if online
            if self.current_mode == OperationalMode.ONLINE:
                self.cloud_api.upload_anonymized_case(case)
            
            # Log success
            self.telemetry.log_event(
                "case_complete",
                {
                    "success": True,
                    "processing_time": time.time() - start_time,
                    "mode": self.current_mode
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Case processing failed: {e}")
            
            # Try fallback to offline if cloud failed
            if use_cloud:
                logger.info("Falling back to offline processing")
                result = super().process_case(case, workflow)
                result["processing_metadata"] = {
                    "mode": "offline_fallback",
                    "original_mode": self.current_mode,
                    "fallback_reason": str(e)
                }
                return result
            
            raise
    
    def _process_case_hybrid(
        self,
        case: Dict,
        workflow: str
    ) -> Dict:
        """
        Process case using hybrid online/offline approach
        Uses cloud for enhancement, local for core processing
        """
        # Step 1: Use local model for initial assessment (fast)
        local_result = super().process_case(case, workflow)
        
        # Step 2: Check if cloud enhancement is needed
        local_confidence = local_result.get("final_assessment", {}).get("confidence", 0)
        
        if local_confidence < 0.85 and self.model_serving:
            # Low confidence - query cloud for second opinion
            logger.info("Querying cloud model for verification (low confidence)")
            
            cloud_result = self._query_cloud_model(case)
            
            if cloud_result:
                # Merge results with confidence weighting
                merged = self._merge_assessments(local_result, cloud_result)
                merged["enhancement"] = "cloud_verification"
                return merged
        
        # Step 3: Enhance with cloud knowledge if available
        if self.hybrid_kb and self.current_mode == OperationalMode.ONLINE:
            # Get latest outbreak alerts
            if "location" in case:
                alerts = self.cloud_api.get_outbreak_alerts(case["location"])
                if alerts:
                    local_result["outbreak_alerts"] = alerts
            
            # Get latest guidelines
            if "symptoms" in case:
                guidelines = self.cloud_api.get_latest_guidelines(
                    topic=case["symptoms"]
                )
                if guidelines:
                    local_result["latest_guidelines"] = guidelines
        
        local_result["enhancement"] = "local_only" if not self.hybrid_kb else "local_with_cloud_data"
        return local_result
    
    def _query_cloud_model(self, case: Dict) -> Optional[Dict]:
        """Query larger cloud-hosted model"""
        if not self.model_serving:
            return None
        
        # Build prompt
        prompt = self._build_case_prompt(case)
        
        # Query cloud model (larger than local)
        response = self.model_serving.query_model(
            prompt=prompt,
            model="medgemma-7b"  # Larger model in cloud
        )
        
        if response:
            return self._parse_cloud_response(response)
        
        return None
    
    def _merge_assessments(
        self,
        local: Dict,
        cloud: Dict
    ) -> Dict:
        """Merge local and cloud assessments with confidence weighting"""
        local_conf = local.get("final_assessment", {}).get("confidence", 0.5)
        cloud_conf = cloud.get("confidence", 0.8)  # Cloud model usually more confident
        
        total_weight = local_conf + cloud_conf
        local_weight = local_conf / total_weight
        cloud_weight = cloud_conf / total_weight
        
        merged = {
            "final_assessment": {
                "diagnosis": {
                    "local": local.get("final_assessment", {}).get("diagnosis"),
                    "cloud": cloud.get("diagnosis"),
                    "consensus": self._find_consensus(
                        local.get("final_assessment", {}).get("diagnosis"),
                        cloud.get("diagnosis")
                    )
                },
                "confidence": local_conf * local_weight + cloud_conf * cloud_weight,
                "method": "hybrid_consensus"
            },
            "reasoning": {
                "local": local.get("reasoning", []),
                "cloud": cloud.get("reasoning", [])
            }
        }
        
        return merged
    
    def _find_consensus(self, local_dx: Dict, cloud_dx: Dict) -> Dict:
        """Find consensus between local and cloud diagnoses"""
        # Simple approach: check for overlap
        if not local_dx or not cloud_dx:
            return local_dx or cloud_dx
        
        local_primary = local_dx.get("most_likely", "").lower()
        cloud_primary = cloud_dx.get("most_likely", "").lower()
        
        if local_primary == cloud_primary:
            return {"most_likely": local_dx.get("most_likely"), "agreement": "strong"}
        else:
            return {
                "most_likely": local_dx.get("most_likely"),
                "alternative": cloud_dx.get("most_likely"),
                "agreement": "weak"
            }
    
    def _build_case_prompt(self, case: Dict) -> str:
        """Build prompt for cloud model"""
        parts = [f"Patient Case:\n"]
        
        if "age" in case:
            parts.append(f"Age: {case['age']}")
        if "gender" in case:
            parts.append(f"Gender: {case['gender']}")
        if "symptoms" in case:
            parts.append(f"Symptoms: {case['symptoms']}")
        if "history" in case:
            parts.append(f"History: {case['history']}")
        
        parts.append("\nProvide diagnosis and treatment recommendations:")
        
        return "\n".join(parts)
    
    def _parse_cloud_response(self, response: Dict) -> Dict:
        """Parse cloud model response"""
        # Simplified parser
        return {
            "diagnosis": response.get("diagnosis", {}),
            "confidence": response.get("confidence", 0.8),
            "reasoning": response.get("reasoning", [])
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "operational_mode": self.current_mode,
            "connectivity": self.cloud_api.connection_status.value if self.cloud_api else "offline",
            "last_connectivity_check": self.last_connectivity_check.isoformat() if self.last_connectivity_check else None,
            "model": {
                "local": self.model_name,
                "cloud_available": self.model_serving is not None
            },
            "statistics": {
                "online_requests": self.online_request_count,
                "offline_requests": self.offline_request_count,
                "mode_switches": self.mode_switches,
                "total_requests": self.online_request_count + self.offline_request_count
            },
            "features": {
                "rag": self.rag is not None,
                "cloud_api": self.cloud_api is not None,
                "telemetry": self.telemetry.enabled,
                "auto_mode_switch": self.auto_mode_switch
            }
        }
    
    def set_mode(self, mode: str, override: bool = False):
        """
        Manually set operational mode
        
        Args:
            mode: Desired mode (online/offline/hybrid)
            override: If True, prevent auto-switching
        """
        self.current_mode = mode
        self.mode_override = override
        logger.info(f"Mode manually set to {mode} (override: {override})")
    
    def sync_knowledge(self, force: bool = False):
        """
        Sync knowledge base with cloud
        
        Args:
            force: Force sync even if recently synced
        """
        if not self.rag or not hasattr(self.rag.kb, 'sync_from_cloud'):
            logger.warning("Knowledge sync not available")
            return
        
        if self.current_mode == OperationalMode.OFFLINE:
            logger.warning("Cannot sync in offline mode")
            return
        
        logger.info("Starting knowledge sync...")
        self.rag.kb.sync_from_cloud(force=force)
        logger.info("Knowledge sync complete")


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Hybrid Orchestrator Demo")
    print("="*60)
    
    # Initialize
    orchestrator = HybridOrchestrator(
        model_name="mock",  # Use mock for demo
        offline_mode=False,
        auto_mode_switch=True
    )
    
    # Check status
    status = orchestrator.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Mode: {status['operational_mode']}")
    print(f"  Connectivity: {status['connectivity']}")
    print(f"  Auto-switch: {status['features']['auto_mode_switch']}")
    
    # Process test case
    test_case = {
        "age": 35,
        "gender": "female",
        "symptoms": "fever, headache, body aches",
        "location": "rural_clinic_a"
    }
    
    print(f"\nProcessing test case...")
    result = orchestrator.process_case(test_case)
    
    print(f"\nResult metadata:")
    print(f"  Mode used: {result['processing_metadata']['mode']}")
    print(f"  Used cloud: {result['processing_metadata']['used_cloud']}")
    print(f"  Processing time: {result['processing_metadata']['processing_time']:.2f}s")
