"""
MedAssist: Agentic Medical Graph-RAG with MedGemma

Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
HAI-DEF Models: https://developers.google.com/health-ai-developer-foundations
Original Repository: https://github.com/MrRezaeiUofT/AMG-RAG

An agentic workflow system for medical question answering using:
- MedGemma models (Google HAI-DEF)
- LangChain & LangGraph for orchestration
- NetworkX for knowledge graphs
- PubMed for evidence retrieval
"""

__version__ = "0.2.0"

from medassist.amg_rag import AMG_RAG_System
from medassist.llm.medgemma import get_medgemma_llm
from medassist.core.knowledge_graph import MedicalKnowledgeGraph
from medassist.models.entities import MedicalEntity, MedicalRelation

__all__ = [
    "AMG_RAG_System",
    "get_medgemma_llm",
    "MedicalKnowledgeGraph",
    "MedicalEntity",
    "MedicalRelation"
]

