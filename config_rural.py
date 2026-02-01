"""
Low-Resource Configuration for Rural/Resource-Constrained Settings
Optimized for CPU-only, low RAM, and offline deployment
"""

# Model Configuration for Resource-Constrained Environments
LOW_RESOURCE_CONFIG = {
    # Use smaller model or quantized version
    "model_name": "google/medgemma-2b",  # Already small at 2B params
    "quantization": "4bit",  # 4-bit quantization (vs 8-bit) → 50% less RAM
    "device": "cpu",  # CPU-only for areas without GPU
    "max_memory": {
        "cpu": "4GB"  # Work on machines with just 4GB RAM
    },
    
    # Reduce batch size and context
    "max_length": 512,  # Shorter context → faster inference
    "batch_size": 1,
    "num_beams": 1,  # Greedy decoding instead of beam search
    
    # Disable expensive features
    "use_embeddings": False,  # No sentence-transformers (saves 500MB)
    "parallel_thinking": False,  # No 512 parallel traces
    "use_cache": True,  # Cache common queries
}

# Simplified Workflow for Rural Settings
RURAL_WORKFLOW_CONFIG = {
    # Start with essential agents only
    "default_agents": ["history", "diagnostic"],  # Skip treatment/knowledge initially
    
    # Escalation strategy optimized for speed
    "escalation_threshold": 0.70,  # Lower threshold (accept faster)
    "max_escalation_levels": 2,  # Max 2 levels (faster completion)
    
    # Confidence settings
    "parallel_samples": 1,  # No parallel generation (save compute)
    "confidence_method": "simple",  # Simple logprob, not group confidence
    
    # Offline mode
    "offline_mode": True,
    "cache_guidelines": True,  # Pre-cache common guidelines
    "use_local_knowledge_base": True,
}

# Focus on Common Rural Health Issues
RURAL_HEALTH_PRIORITIES = {
    "common_conditions": [
        # Infectious diseases
        "malaria", "tuberculosis", "dengue", "typhoid",
        "respiratory_infections", "diarrheal_diseases",
        
        # Chronic conditions
        "hypertension", "diabetes", "malnutrition",
        
        # Maternal/child health
        "prenatal_complications", "childhood_diseases",
        
        # Injuries
        "wounds", "fractures", "snake_bites",
    ],
    
    "limited_diagnostics": [
        # Focus on clinical diagnosis without advanced tests
        "symptom_based_diagnosis",
        "physical_examination_findings",
        "basic_vital_signs",
    ],
    
    "treatment_constraints": [
        # Consider limited drug availability
        "essential_medicines_list",
        "alternative_low_cost_treatments",
        "non_pharmaceutical_interventions",
    ],
}

# Offline Knowledge Base
OFFLINE_RESOURCES = {
    "guidelines": [
        "WHO Essential Medicines List",
        "WHO IMAI (Integrated Management of Adult/Adolescent Illness)",
        "IMCI (Integrated Management of Childhood Illness)",
        "Basic Emergency Care",
    ],
    
    "decision_support": [
        "Symptom-based triage protocols",
        "Danger signs recognition",
        "Referral criteria",
    ],
    
    "local_adaptations": True,  # Adapt to local disease prevalence
}

# Simplified Interface for Non-Specialists
SIMPLE_UI_CONFIG = {
    "language_options": ["english", "vietnamese", "local_language"],
    "visual_guides": True,  # Picture-based symptom selection
    "voice_input": True,  # For low-literacy users
    "sms_integration": True,  # Work via SMS if no internet
    
    "output_format": {
        "simple_language": True,  # Avoid medical jargon
        "actionable_steps": True,  # Clear next steps
        "red_flags": True,  # Highlight danger signs
        "referral_guidance": True,  # When to refer to hospital
    },
}

# Network Optimization
CONNECTIVITY_CONFIG = {
    "offline_first": True,
    "low_bandwidth_mode": True,
    
    # Sync strategies
    "sync_when_available": {
        "update_guidelines": "weekly",
        "upload_anonymized_cases": "batch",
        "download_model_updates": "manual",
    },
    
    # Data efficiency
    "compress_data": True,
    "text_only_mode": True,  # No images unless critical
}

# Cost Optimization
COST_CONSTRAINTS = {
    "target_cost_per_consultation": "$0.01",  # Ultra-low cost
    "optimize_for": "inference_cost",  # Not training
    
    "hardware_requirements": {
        "min_ram": "4GB",
        "min_storage": "10GB",
        "gpu": "optional",
        "internet": "intermittent",
    },
}

# Safety & Ethics for Low-Resource Settings
SAFETY_CONFIG = {
    "conservative_recommendations": True,  # Err on side of caution
    "emphasize_limitations": True,  # Clear about AI limitations
    "mandatory_referral_conditions": [
        "severe_symptoms",
        "pregnancy_complications",
        "pediatric_emergencies",
        "trauma",
    ],
    
    "human_oversight": {
        "telemedicine_backup": True,  # Connect to remote doctor if available
        "community_health_worker_review": True,
        "escalation_protocols": True,
    },
}

# Performance Targets for Rural Deployment
RURAL_PERFORMANCE_TARGETS = {
    "inference_time": "< 10 seconds",  # Fast enough on CPU
    "memory_usage": "< 4GB RAM",  # Work on basic laptops
    "model_size": "< 2GB",  # Fits on small storage
    "offline_capable": True,
    "cost_per_query": "< $0.01",
    
    # Accuracy targets
    "triage_accuracy": "> 85%",  # Correctly identify urgent cases
    "common_disease_accuracy": "> 80%",
    "referral_sensitivity": "> 95%",  # Don't miss serious cases
}

# Deployment Scenarios
RURAL_DEPLOYMENT_SCENARIOS = {
    "health_post": {
        "device": "laptop or tablet",
        "connectivity": "intermittent 3G",
        "user": "community health worker",
        "use_cases": [
            "initial_triage",
            "basic_diagnosis",
            "treatment_guidance",
            "referral_decisions",
        ],
    },
    
    "mobile_clinic": {
        "device": "smartphone",
        "connectivity": "offline primary",
        "user": "visiting nurse",
        "use_cases": [
            "field_assessment",
            "outbreak_detection",
            "follow_up_care",
        ],
    },
    
    "telemedicine_point": {
        "device": "basic computer",
        "connectivity": "low bandwidth internet",
        "user": "health volunteer",
        "use_cases": [
            "remote_consultation_support",
            "data_collection",
            "health_education",
        ],
    },
}
