"""
MedAssist: Agentic Medical Workflow System
Configuration file for model and agent settings
"""

# Model Configuration
MODEL_CONFIG = {
    "model_name": "google/medgemma-2b",  # or path to local model
    "device": "auto",  # auto, cuda, cpu
    "load_in_8bit": True,  # for memory efficiency
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 10,
    "timeout_seconds": 120,
    "verbose": True,
    "return_intermediate_steps": True,
}

# Orchestrator settings
ORCHESTRATOR_CONFIG = {
    "agent_timeout": 60,
    "parallel_execution": True,
    "max_parallel_agents": 3,
}

# Agent roles and descriptions
AGENT_ROLES = {
    "orchestrator": {
        "name": "Medical Orchestrator",
        "description": "Coordinates the diagnostic workflow and delegates tasks to specialized agents",
        "system_prompt": """You are a medical orchestrator agent. Your role is to:
1. Analyze the patient case
2. Determine which specialized agents to engage
3. Coordinate information flow between agents
4. Synthesize findings into a comprehensive assessment
Always prioritize patient safety and evidence-based medicine."""
    },
    
    "history": {
        "name": "Medical History Analyst",
        "description": "Analyzes patient medical history and identifies relevant risk factors",
        "system_prompt": """You are a medical history analysis agent. Your role is to:
1. Review patient demographics, past medical history, medications, and allergies
2. Identify relevant risk factors for current presentation
3. Flag important historical findings that may impact diagnosis
4. Summarize pertinent positive and negative findings
Be thorough and systematic in your analysis."""
    },
    
    "diagnostic": {
        "name": "Diagnostic Agent",
        "description": "Processes symptoms and generates differential diagnoses",
        "system_prompt": """You are a diagnostic reasoning agent. Your role is to:
1. Analyze presenting symptoms and signs
2. Generate a prioritized differential diagnosis list
3. Suggest relevant diagnostic tests or examinations
4. Reason through the most likely diagnoses
Use clinical reasoning frameworks and consider both common and serious causes."""
    },
    
    "treatment": {
        "name": "Treatment Planning Agent",
        "description": "Recommends evidence-based treatment plans",
        "system_prompt": """You are a treatment planning agent. Your role is to:
1. Review confirmed or suspected diagnoses
2. Recommend evidence-based treatment options
3. Consider contraindications and drug interactions
4. Provide patient education points
Always follow clinical guidelines and consider patient-specific factors."""
    },
    
    "knowledge": {
        "name": "Medical Knowledge Agent",
        "description": "Queries medical literature and clinical guidelines",
        "system_prompt": """You are a medical knowledge retrieval agent. Your role is to:
1. Search medical literature and guidelines
2. Provide evidence-based recommendations
3. Cite relevant studies or guidelines
4. Clarify medical concepts
Provide accurate, up-to-date medical information from reliable sources."""
    }
}

# Tool definitions
TOOLS_CONFIG = {
    "lab_lookup": {
        "enabled": True,
        "description": "Look up lab test reference ranges and interpretations"
    },
    "drug_interaction": {
        "enabled": True,
        "description": "Check for drug-drug interactions"
    },
    "guideline_search": {
        "enabled": True,
        "description": "Search clinical practice guidelines"
    },
    "icd_search": {
        "enabled": True,
        "description": "Search ICD-10 diagnosis codes"
    },
    "calculator": {
        "enabled": True,
        "description": "Medical calculators (BMI, eGFR, risk scores, etc.)"
    }
}

# Workflow templates
WORKFLOW_TEMPLATES = {
    "standard_diagnosis": [
        "history",
        "diagnostic",
        "treatment"
    ],
    "complex_case": [
        "history",
        "diagnostic",
        "knowledge",
        "treatment"
    ],
    "triage": [
        "diagnostic",
        "treatment"
    ]
}

# Privacy and security
PRIVACY_CONFIG = {
    "deidentify_output": True,
    "audit_logging": True,
    "local_only": True,  # No external API calls
}
