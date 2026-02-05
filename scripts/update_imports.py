#!/usr/bin/env python3
"""
Update imports to new folder structure
"""

import os
import re
from pathlib import Path

# Mapping old imports to new imports
IMPORT_MAPPING = {
    'from medassist.knowledge_graph': 'from medassist.models.knowledge_graph',
    'from medassist.multimodal_models': 'from medassist.models.multimodal_models',
    'from medassist.medical_ner': 'from medassist.tools.medical_ner',
    'from medassist.pubmed_retrieval': 'from medassist.tools.pubmed_retrieval',
    'from medassist.graph_retrieval': 'from medassist.tools.graph_retrieval',
    'from medassist.hierarchical_retrieval': 'from medassist.tools.hierarchical_retrieval',
    'from medassist.multimodal': 'from medassist.tools.multimodal',
    'from medassist.medical_image_search': 'from medassist.tools.medical_image_search',
    'from medassist.agentic_orchestrator': 'from medassist.core.agentic_orchestrator',
    'from medassist.langgraph_orchestrator': 'from medassist.core.langgraph_orchestrator',
    'from medassist.amg_rag_orchestrator': 'from medassist.core.amg_rag_orchestrator',
    'from medassist.agentic_workflow': 'from medassist.agents.agentic_workflow',
    'from medassist.ingestion_pipeline': 'from medassist.services.ingestion_pipeline',
    'from medassist.mcp_server': 'from medassist.services.mcp_server',
    'from medassist.mcp_client': 'from medassist.services.mcp_client',
}

def update_file_imports(filepath):
    """Update imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        updated = False
        
        # Replace imports
        for old_import, new_import in IMPORT_MAPPING.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                updated = True
        
        # Write back if changed
        if updated and content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        return False
        
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    """Update all Python files"""
    base_dir = Path('/home/quan/MedGemma')
    
    # Files to update
    patterns = [
        'simple_agent.py',
        'api.py',
        'run_agent.py',
        'demo_interface.py',
        'tests/*.py',
        'examples/*.py',
        'medassist/services/*.py',
    ]
    
    updated_count = 0
    
    for pattern in patterns:
        for filepath in base_dir.glob(pattern):
            if filepath.is_file() and filepath.suffix == '.py':
                if update_file_imports(filepath):
                    updated_count += 1
    
    print(f"\nUpdated {updated_count} files")

if __name__ == '__main__':
    main()
