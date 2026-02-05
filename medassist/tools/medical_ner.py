"""
Medical Named Entity Recognition - AMG-RAG Style
LLM-based entity extraction with flexible provider support
Following AMG-RAG paper: https://github.com/MrRezaeiUofT/AMG-RAG
"""

import logging
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")

# LLM providers
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class MedicalEntity:
    """
    Medical entity with AMG-RAG enhancements
    
    AMG-RAG features:
    - Relevance scoring (1-10)
    - Context-aware descriptions  
    - Confidence scoring
    """
    text: str
    entity_type: str
    confidence: float = 1.0
    relevance_score: float = 5.0
    description: Optional[str] = None
    sources: List[str] = field(default_factory=list)


class AMGRAGEntityExtractor:
    """
    AMG-RAG Entity Extraction with Flexible LLM Support
    
    Auto-detects best available provider:
    - Groq (fast, free API)
    - OpenAI (reliable)
    - Ollama (local)
    
    Paper: "AMG-RAG: Agentic Medical Graph-RAG"
    """
    
    def __init__(
        self,
        llm_provider: str = "auto",  # Auto-detect best available
        model_name: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize AMG-RAG entity extractor
        
        Args:
            llm_provider: "auto", "groq", "openai", or "ollama"
            model_name: Model name (auto-selected if None)
            temperature: LLM temperature
        """
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("Install: pip install langchain")
        
        # Initialize LLM with auto-detection
        self.llm = self._init_llm(llm_provider, model_name, temperature)
        self._setup_extraction_chain()
    
    def _init_llm(self, provider: str, model_name: Optional[str], temperature: float):
        """Initialize LLM with best available provider"""
        
        # Auto-detect: Groq > OpenAI > Ollama
        if provider == "auto":
            if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
                provider = "groq"
                model_name = model_name or "llama-3.3-70b-versatile"
            elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                provider = "openai"
                model_name = model_name or "gpt-4o-mini"
            elif OLLAMA_AVAILABLE:
                provider = "ollama"
                model_name = model_name or "llama3.2"
            else:
                raise RuntimeError("No LLM provider available. Install langchain-groq or langchain-openai")
        
        # Initialize provider
        if provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Install: pip install langchain-groq")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            logger.info(f"NER using Groq: {model_name}")
            return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)
        
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("Install: pip install langchain-openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            logger.info(f"NER using OpenAI: {model_name}")
            return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Install: pip install langchain-ollama")
            logger.info(f"NER using Ollama: {model_name}")
            return ChatOllama(model=model_name, temperature=temperature)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _setup_extraction_chain(self):
        """Setup LLM extraction chain"""
        
        entity_schemas = [
            ResponseSchema(name="entities", description="Medical entities list", type="array"),
            ResponseSchema(name="scores", description="Relevance scores (1-10)", type="array"),
            ResponseSchema(name="descriptions", description="Entity descriptions", type="array"),
            ResponseSchema(name="types", description="Entity types", type="array")
        ]
        
        parser = StructuredOutputParser.from_response_schemas(entity_schemas)
        
        self.extraction_chain = PromptTemplate(
            template="""Extract medical entities from text with relevance scoring.

Text: {text}
Context: {context}

For each entity provide:
1. Entity name
2. Relevance score (1-10)
3. Brief description
4. Type (disease, symptom, treatment, medication, anatomy, biomarker)

Return JSON:
{format_instructions}""",
            input_variables=["text", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        ) | self.llm | parser
    
    def extract_entities(
        self,
        text: str,
        context: Optional[str] = None,
        max_entities: int = 8
    ) -> List[MedicalEntity]:
        """Extract entities using LLM"""
        
        if not text or not text.strip():
            return []
        
        try:
            result = self.extraction_chain.invoke({
                "text": text,
                "context": context or ""
            })
            
            entities = result.get("entities", [])
            scores = result.get("scores", [])
            descriptions = result.get("descriptions", [])
            types = result.get("types", [])
            
            medical_entities = []
            for i, name in enumerate(entities[:max_entities]):
                entity = MedicalEntity(
                    text=name,
                    entity_type=types[i] if i < len(types) else "medical_concept",
                    confidence=0.9,
                    relevance_score=scores[i] if i < len(scores) else 5.0,
                    description=descriptions[i] if i < len(descriptions) else f"Medical entity: {name}",
                    sources=["LLM"]
                )
                medical_entities.append(entity)
            
            logger.info(f"Extracted {len(medical_entities)} entities")
            return medical_entities
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []


class HybridMedicalNER:
    """Backward compatible wrapper"""
    
    def __init__(self, llm_provider: str = "auto", model_name: Optional[str] = None):
        self.extractor = AMGRAGEntityExtractor(llm_provider, model_name)
    
    def extract_entities(self, text: str, context: Optional[str] = None) -> List[MedicalEntity]:
        return self.extractor.extract_entities(text, context)


# Aliases
MedicalNER = HybridMedicalNER
BioBERTNER = HybridMedicalNER


class MedicalEmbeddings:
    """SentenceTransformer embeddings (open-source)"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embeddings: {model_name}")
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query, convert_to_tensor=False).tolist()
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents, convert_to_tensor=False).tolist()
    
    def __call__(self, texts):
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)
