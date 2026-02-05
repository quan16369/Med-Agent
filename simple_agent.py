"""
Simple Agent Interface - Focus on Core Agent Functionality
Loại bỏ các thành phần production-ready, chỉ tập trung vào agent workflow
"""

from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from medassist.config import get_config
from medassist.logging_utils import setup_logging, get_logger
from medassist.agentic_orchestrator import AgenticMedicalOrchestrator
from medassist.ingestion_pipeline import IngestionPipeline
from medassist.multimodal_models import MultimodalMessage, TextContent, ImageUrlContent

# Setup logging
setup_logging()
logger = get_logger(__name__)


class SimpleAgent:
    """
    Simple agent interface tập trung vào core functionality
    Không có rate limiting, health checks, hay production overhead
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize agent with basic configuration"""
        self.config = get_config(config_path)
        self.orchestrator = AgenticMedicalOrchestrator(self.config)
        self.ingestion_pipeline = IngestionPipeline(self.config)
        logger.info("✅ Simple Agent initialized")
    
    def ask(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Đơn giản hóa: hỏi câu hỏi và nhận câu trả lời
        
        Args:
            question: Câu hỏi medical
            context: Context bổ sung (optional)
            
        Returns:
            Dict với answer, reasoning, entities, relationships
        """
        try:
            logger.info(f"❓ Question: {question}")
            
            # Execute agent workflow
            result = self.orchestrator.process_query(
                question=question,
                context=context
            )
            
            logger.info(f"✅ Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    def ask_with_image(
        self, 
        question: str, 
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Hỏi với image (X-ray, CT, MRI, etc.)
        
        Args:
            question: Câu hỏi về image
            image_path: Đường dẫn đến image file
            image_base64: Base64 encoded image
            
        Returns:
            Dict với answer và visual analysis
        """
        try:
            # Convert image to multimodal format
            if image_path:
                with open(image_path, 'rb') as f:
                    import base64
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            elif image_base64:
                image_data = image_base64
            else:
                raise ValueError("Cần image_path hoặc image_base64")
            
            # Create multimodal message
            message = MultimodalMessage(content=[
                TextContent(text=question),
                ImageUrlContent(url=f"data:image/jpeg;base64,{image_data}")
            ])
            
            logger.info(f"🖼️ Multimodal question: {question}")
            
            # Process with multimodal support
            result = self.orchestrator.process_query(
                question=question,
                multimodal_content=message
            )
            
            logger.info(f"✅ Multimodal answer generated")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multimodal error: {e}")
            return {
                "answer": f"Error processing multimodal question: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    def ingest_document(
        self, 
        text: str, 
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Đưa document vào knowledge graph
        
        Args:
            text: Nội dung document
            doc_id: ID của document
            metadata: Metadata bổ sung
            
        Returns:
            Dict với entities và relationships extracted
        """
        try:
            logger.info(f"📄 Ingesting document: {doc_id or 'unnamed'}")
            
            # Process document through ingestion pipeline
            result = self.ingestion_pipeline.process_document(
                text=text,
                doc_id=doc_id,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Document ingested: {result.get('entity_count', 0)} entities, {result.get('relationship_count', 0)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def ingest_multimodal_document(
        self,
        text: str,
        images: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ingest document with images (case studies, medical reports)
        
        Args:
            text: Document text
            images: List of {base64, metadata} dicts
            doc_id: Document ID
            metadata: Additional metadata
            
        Returns:
            Dict with extraction results
        """
        try:
            logger.info(f"📄🖼️ Ingesting multimodal document: {doc_id or 'unnamed'} with {len(images)} images")
            
            # Convert to multimodal format
            multimodal_content = {
                "text": text,
                "images": images
            }
            
            result = self.ingestion_pipeline.process_document(
                text=multimodal_content,
                doc_id=doc_id,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Multimodal document ingested")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multimodal ingestion error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def explore_knowledge_graph(
        self, 
        entity_name: str, 
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Explore knowledge graph around an entity
        
        Args:
            entity_name: Tên entity cần explore
            max_depth: Độ sâu tối đa để traverse
            
        Returns:
            Dict với entities và relationships liên quan
        """
        try:
            logger.info(f"🔍 Exploring KG for: {entity_name}")
            
            # Use graph retrieval
            from medassist.graph_retrieval import GraphConditionalRetrieval
            graph_retriever = GraphConditionalRetrieval(
                self.orchestrator.knowledge_graph
            )
            
            result = graph_retriever.explore_entity(
                entity_name=entity_name,
                max_depth=max_depth
            )
            
            logger.info(f"✅ Found {len(result.get('related_entities', []))} related entities")
            return result
            
        except Exception as e:
            logger.error(f"❌ Exploration error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def demo_simple_agent():
    """Demo usage của Simple Agent"""
    
    print("=" * 60)
    print("🤖 Simple Medical Agent Demo")
    print("=" * 60)
    
    # Initialize agent
    agent = SimpleAgent()
    
    # Example 1: Simple Q&A
    print("\n1️⃣ Simple Medical Q&A")
    print("-" * 60)
    result1 = agent.ask(
        question="What are the symptoms of diabetes?",
        context="Patient is 45 years old with family history"
    )
    print(f"Answer: {result1.get('answer', 'N/A')}")
    
    # Example 2: Ingest document
    print("\n2️⃣ Document Ingestion")
    print("-" * 60)
    result2 = agent.ingest_document(
        text="Diabetes mellitus causes hyperglycemia. Metformin treats diabetes by reducing glucose production.",
        doc_id="doc_001",
        metadata={"source": "medical_textbook"}
    )
    print(f"Extracted: {result2.get('entity_count', 0)} entities, {result2.get('relationship_count', 0)} relationships")
    
    # Example 3: Explore knowledge graph
    print("\n3️⃣ Knowledge Graph Exploration")
    print("-" * 60)
    result3 = agent.explore_knowledge_graph(
        entity_name="diabetes",
        max_depth=2
    )
    print(f"Found {len(result3.get('related_entities', []))} related entities")
    
    # Example 4: Multimodal Q&A (if image available)
    print("\n4️⃣ Multimodal Q&A (với image)")
    print("-" * 60)
    print("Skipped - requires actual medical image")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_simple_agent()
