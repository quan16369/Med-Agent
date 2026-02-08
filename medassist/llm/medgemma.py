"""
MedGemma LLM Integration for HAI-DEF Models
Supports multiple deployment options:
1. Google GenAI API (cloud)
2. Vertex AI (GCP)
3. Ollama (local)
4. vLLM (local server)

Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
"""

from typing import Optional, Dict, Any, Literal, List, Union
from pathlib import Path
from decouple import config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class MedGemmaLLM:
    """
    Factory class for MedGemma model initialization.
    Auto-detects available providers and returns appropriate LLM instance.
    """
    
    MEDGEMMA_MODELS = {
        # Text-only models
        "medgemma-3-8b": "MedGemma 3 8B - Medical text comprehension",
        "medgemma-3-27b": "MedGemma 3 27B - Advanced clinical reasoning",
        "medgemma-2b": "MedGemma 2B - Lightweight medical tasks",
        
        # Multimodal models (text + medical images)
        "medgemma-1.5-4b": "MedGemma 1.5 4B Multimodal - X-ray, CT, MRI, histopathology",
        "medgemma-1-4b": "MedGemma 1 4B Multimodal - Medical image interpretation",
        "medgemma-1-27b": "MedGemma 1 27B Multimodal - Advanced multimodal reasoning",
    }
    
    @staticmethod
    def create(
        model_name: str = "medgemma-3-8b",
        provider: Literal["google", "vertex", "ollama", "vllm", "auto"] = "auto",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> BaseChatModel:
        """
        Create MedGemma LLM instance with specified provider.
        
        Args:
            model_name: MedGemma model variant
            provider: LLM provider (auto-detects if "auto")
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum generation length
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized LangChain chat model
        """
        if provider == "auto":
            provider = MedGemmaLLM._detect_provider()
        
        if provider == "google":
            return MedGemmaLLM._create_google_genai(
                model_name, temperature, max_tokens, **kwargs
            )
        elif provider == "vertex":
            return MedGemmaLLM._create_vertex_ai(
                model_name, temperature, max_tokens, **kwargs
            )
        elif provider == "ollama":
            return MedGemmaLLM._create_ollama(
                model_name, temperature, max_tokens, **kwargs
            )
        elif provider == "vllm":
            return MedGemmaLLM._create_vllm(
                model_name, temperature, max_tokens, **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _detect_provider() -> str:
        """Auto-detect available LLM provider."""
        # Check for Google API key
        google_api_key = config("GOOGLE_API_KEY", default=None)
        if google_api_key:
            return "google"
        
        # Check for Vertex AI credentials
        vertex_project = config("GOOGLE_CLOUD_PROJECT", default=None)
        if vertex_project:
            return "vertex"
        
        # Check for local Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return "ollama"
        except Exception:
            pass
        
        # Check for vLLM endpoint
        vllm_endpoint = config("VLLM_ENDPOINT", default=None)
        if vllm_endpoint:
            return "vllm"
        
        raise RuntimeError(
            "No MedGemma provider detected. Please configure:\n"
            "1. GOOGLE_API_KEY for Google GenAI\n"
            "2. GOOGLE_CLOUD_PROJECT for Vertex AI\n"
            "3. Local Ollama server at localhost:11434\n"
            "4. VLLM_ENDPOINT for vLLM server"
        )
    
    @staticmethod
    def _create_google_genai(
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> BaseChatModel:
        """Create Google GenAI API instance."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Map MedGemma names to Google API model IDs
        model_mapping = {
            "medgemma-3-8b": "models/medgemma-3-8b",
            "medgemma-3-27b": "models/medgemma-3-27b",
            "medgemma-2b": "models/medgemma-2b",
        }
        
        api_model = model_mapping.get(model_name, model_name)
        
        return ChatGoogleGenerativeAI(
            model=api_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=config("GOOGLE_API_KEY"),
            **kwargs
        )
    
    @staticmethod
    def _create_vertex_ai(
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> BaseChatModel:
        """Create Vertex AI instance."""
        from langchain_google_vertexai import ChatVertexAI
        
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            project=config("GOOGLE_CLOUD_PROJECT"),
            location=config("GOOGLE_CLOUD_LOCATION", default="us-central1"),
            **kwargs
        )
    
    @staticmethod
    def _create_ollama(
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> BaseChatModel:
        """Create Ollama local instance."""
        from langchain_ollama import ChatOllama
        
        # Ollama uses model name directly
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=config("OLLAMA_BASE_URL", default="http://localhost:11434"),
            **kwargs
        )
    
    @staticmethod
    def _create_vllm(
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> BaseChatModel:
        """Create vLLM server instance."""
        from langchain_openai import ChatOpenAI
        
        # vLLM uses OpenAI-compatible API
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=config("VLLM_ENDPOINT"),
            api_key="EMPTY",  # vLLM doesn't require API key
            **kwargs
        )


def get_medgemma_llm(
    model: str = "medgemma-3-8b",
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """
    Convenience function to get MedGemma LLM with auto-detection.
    
    Example:
        # Text-only
        llm = get_medgemma_llm(model="medgemma-3-8b", temperature=0.0)
        response = llm.invoke("What is type 2 diabetes?")
        
        # Multimodal
        llm = get_medgemma_llm(model="medgemma-1.5-4b", temperature=0.0)
        response = llm.invoke([
            {"type": "text", "text": "What abnormalities are visible?"},
            {"type": "image_url", "image_url": {"url": "path/to/xray.jpg"}}
        ])
    """
    return MedGemmaLLM.create(
        model_name=model,
        provider="auto",
        temperature=temperature,
        **kwargs
    )


def is_multimodal_model(model_name: str) -> bool:
    """Check if a MedGemma model supports multimodal input."""
    multimodal_models = ["medgemma-1.5-4b", "medgemma-1-4b", "medgemma-1-27b"]
    return model_name in multimodal_models


def create_image_message(
    text: str,
    image_paths: Union[str, List[str]],
    detail: str = "high"
) -> HumanMessage:
    """
    Create a message with text and medical images for multimodal models.
    
    Args:
        text: Question or instruction about the images
        image_paths: Path(s) to medical image files (X-ray, CT, MRI, etc.)
        detail: Image detail level ("low", "high", "auto")
        
    Returns:
        HumanMessage with text and image content
        
    Example:
        message = create_image_message(
            text="Describe the abnormalities in this chest X-ray.",
            image_paths="chest_xray.jpg"
        )
        response = llm.invoke([message])
    """
    import base64
    from mimetypes import guess_type
    
    # Normalize to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Build content list
    content = [{"type": "text", "text": text}]
    
    for image_path in image_paths:
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read and encode image
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Get MIME type
        mime_type = guess_type(str(path))[0] or "image/jpeg"
        
        # Add to content
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": detail
            }
        })
    
    return HumanMessage(content=content)
