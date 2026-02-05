#!/usr/bin/env python3
"""
Medical AI Agent Demo Interface
Interactive web UI using Gradio for demonstrating medical knowledge graph and Q&A
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError:
    logger.error("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from simple_agent import SimpleAgent


class MedicalAgentDemo:
    """Demo interface for Medical AI Agent"""
    
    def __init__(self):
        """Initialize the demo with SimpleAgent"""
        logger.info("Initializing Medical Agent Demo...")
        
        # Initialize agent
        self.agent = SimpleAgent()
        
        # Track conversation history
        self.conversation_history = []
        
        logger.info("Demo initialized successfully!")
    
    def ask_question(
        self, 
        question: str, 
        context: str = "", 
        history: list = None
    ) -> Tuple[str, list]:
        """
        Ask a medical question
        
        Args:
            question: Medical question
            context: Optional additional context
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not question.strip():
            return "Please enter a question.", history or []
        
        try:
            # Get answer from agent
            answer = self.agent.ask(question, context if context.strip() else None)
            
            # Update history
            if history is None:
                history = []
            
            history.append((question, answer))
            
            return answer, history
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return error_msg, history or []
    
    def ask_with_image(
        self,
        question: str,
        image,
        history: list = None
    ) -> Tuple[str, list]:
        """
        Ask a question about a medical image
        
        Args:
            question: Question about the image
            image: Image file (PIL Image or path)
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not question.strip():
            return "Please enter a question about the image.", history or []
        
        if image is None:
            return "Please upload an image.", history or []
        
        try:
            # Convert image to path if PIL Image
            if hasattr(image, 'filename'):
                image_path = image.filename
            else:
                # Assume it's already a path or PIL image
                image_path = image
            
            # Get answer from agent
            answer = self.agent.ask_with_image(question, image_path)
            
            # Update history
            if history is None:
                history = []
            
            history.append((f"[Image] {question}", answer))
            
            return answer, history
            
        except Exception as e:
            error_msg = f"Error processing image question: {str(e)}"
            logger.error(error_msg)
            return error_msg, history or []
    
    def ingest_document(self, text: str, doc_id: str = "") -> str:
        """
        Ingest medical document into knowledge graph
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            Status message
        """
        if not text.strip():
            return "Please enter document text."
        
        try:
            # Generate doc_id if not provided
            if not doc_id.strip():
                import hashlib
                doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
            
            # Ingest document
            self.agent.ingest_document(text, doc_id)
            
            return f"Document ingested successfully! (ID: {doc_id})\n\nEntities and relationships extracted and added to knowledge graph."
            
        except Exception as e:
            error_msg = f"Error ingesting document: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def explore_graph(self, entity_name: str, max_depth: int = 2) -> str:
        """
        Explore knowledge graph around an entity
        
        Args:
            entity_name: Entity to explore
            max_depth: Maximum traversal depth
            
        Returns:
            Graph exploration results
        """
        if not entity_name.strip():
            return "Please enter an entity name to explore."
        
        try:
            # Explore graph
            result = self.agent.explore_knowledge_graph(entity_name, max_depth)
            
            # Format output
            output = f"Exploring Knowledge Graph: {entity_name}\n\n"
            
            if "entities" in result:
                output += f"Found {len(result['entities'])} related entities:\n"
                for entity in result['entities'][:10]:  # Limit to 10
                    output += f"  • [{entity['type']}] {entity['text']}\n"
                
                if len(result['entities']) > 10:
                    output += f"  ... and {len(result['entities']) - 10} more\n"
            
            if "relationships" in result:
                output += f"\nFound {len(result['relationships'])} relationships:\n"
                for rel in result['relationships'][:10]:  # Limit to 10
                    output += f"  • {rel['source']} → {rel['type']} → {rel['target']}\n"
                
                if len(result['relationships']) > 10:
                    output += f"  ... and {len(result['relationships']) - 10} more\n"
            
            if "paths" in result:
                output += f"\n🛤️ Found {len(result['paths'])} paths\n"
            
            return output
            
        except Exception as e:
            error_msg = f"Error exploring graph: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .tabs {
            margin-top: 20px;
        }
        #title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        #subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Medical AI Agent Demo") as interface:
            
            # Header
            gr.Markdown(
                """
                # 🏥 Medical AI Agent Demo
                ### Powered by AMG-RAG + Code RAG + Multimodal AI
                
                This demo showcases an advanced medical AI agent with:
                - 🧠 Medical Knowledge Graph (AMG-RAG)
                - Hierarchical Retrieval (Code RAG)
                - 👁️ Vision Support (Multimodal)
                - 📚 PubMed Literature Integration
                """,
                elem_id="title"
            )
            
            # Tabs for different functionalities
            with gr.Tabs() as tabs:
                
                # Tab 1: Medical Q&A
                with gr.Tab("💬 Ask Medical Questions"):
                    gr.Markdown("Ask medical questions with optional context from the knowledge graph.")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            question_input = gr.Textbox(
                                label="Your Question",
                                placeholder="e.g., What are the symptoms of diabetes?",
                                lines=2
                            )
                            context_input = gr.Textbox(
                                label="Additional Context (Optional)",
                                placeholder="Provide any additional context or patient information...",
                                lines=3
                            )
                            ask_btn = gr.Button("Ask", variant="primary")
                        
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=500
                            )
                    
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    
                    # Event handlers
                    ask_btn.click(
                        fn=self.ask_question,
                        inputs=[question_input, context_input, chatbot],
                        outputs=[gr.Textbox(visible=False), chatbot]
                    )
                    
                    clear_btn.click(
                        fn=lambda: ([], "", ""),
                        inputs=[],
                        outputs=[chatbot, question_input, context_input]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["What are the symptoms of type 2 diabetes?", ""],
                            ["How is diabetic neuropathy treated?", ""],
                            ["What is the relationship between HbA1c and diabetes control?", ""],
                            ["What causes cardiovascular complications in diabetic patients?", ""],
                        ],
                        inputs=[question_input, context_input]
                    )
                
                # Tab 2: Multimodal Q&A
                with gr.Tab("Medical Image Analysis"):
                    gr.Markdown("Upload medical images and ask questions about them.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                label="Medical Image",
                                type="filepath"
                            )
                            image_question = gr.Textbox(
                                label="Question about Image",
                                placeholder="e.g., What abnormalities are visible in this X-ray?",
                                lines=3
                            )
                            image_ask_btn = gr.Button("Analyze", variant="primary")
                        
                        with gr.Column(scale=2):
                            image_chatbot = gr.Chatbot(
                                label="Image Analysis",
                                height=500
                            )
                    
                    image_clear_btn = gr.Button("🗑️ Clear")
                    
                    # Event handlers
                    image_ask_btn.click(
                        fn=self.ask_with_image,
                        inputs=[image_question, image_input, image_chatbot],
                        outputs=[gr.Textbox(visible=False), image_chatbot]
                    )
                    
                    image_clear_btn.click(
                        fn=lambda: ([], None, ""),
                        inputs=[],
                        outputs=[image_chatbot, image_input, image_question]
                    )
                
                # Tab 3: Document Ingestion
                with gr.Tab("Ingest Medical Documents"):
                    gr.Markdown("Add medical documents to the knowledge graph for enhanced Q&A.")
                    
                    with gr.Row():
                        with gr.Column():
                            doc_text = gr.Textbox(
                                label="Document Text",
                                placeholder="Paste medical text, case reports, or literature...",
                                lines=10
                            )
                            doc_id = gr.Textbox(
                                label="Document ID (Optional)",
                                placeholder="e.g., case_001",
                                lines=1
                            )
                            ingest_btn = gr.Button("📥 Ingest Document", variant="primary")
                        
                        with gr.Column():
                            ingest_output = gr.Textbox(
                                label="Status",
                                lines=10,
                                interactive=False
                            )
                    
                    # Event handler
                    ingest_btn.click(
                        fn=self.ingest_document,
                        inputs=[doc_text, doc_id],
                        outputs=[ingest_output]
                    )
                    
                    # Example
                    gr.Examples(
                        examples=[
                            ["Patient with type 2 diabetes mellitus presents with peripheral neuropathy and retinopathy. HbA1c level is 8.5%. Started on metformin and lifestyle modifications.", "case_001"],
                            ["Diabetic nephropathy is characterized by proteinuria and declining renal function. Associated with poor glycemic control and hypertension.", "lit_001"],
                        ],
                        inputs=[doc_text, doc_id]
                    )
                
                # Tab 4: Knowledge Graph Explorer
                with gr.Tab("🕸️ Explore Knowledge Graph"):
                    gr.Markdown("Explore relationships and connections in the medical knowledge graph.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            entity_input = gr.Textbox(
                                label="Entity Name",
                                placeholder="e.g., diabetes, metformin, neuropathy",
                                lines=1
                            )
                            depth_slider = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1,
                                label="Exploration Depth"
                            )
                            explore_btn = gr.Button("Explore", variant="primary")
                        
                        with gr.Column(scale=2):
                            explore_output = gr.Textbox(
                                label="Graph Exploration Results",
                                lines=15,
                                interactive=False
                            )
                    
                    # Event handler
                    explore_btn.click(
                        fn=self.explore_graph,
                        inputs=[entity_input, depth_slider],
                        outputs=[explore_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["diabetes", 2],
                            ["neuropathy", 2],
                            ["metformin", 1],
                        ],
                        inputs=[entity_input, depth_slider]
                    )
            
            # Footer
            gr.Markdown(
                """
                ---
                **Research Papers Implemented:**
                - [AMG-RAG](https://arxiv.org/abs/2410.03883): Agentic Medical Knowledge Graphs
                - [Code RAG](https://arxiv.org/abs/2508.10068): Hierarchical Retrieval Optimization
                - [Kubrick AI](https://github.com/kubrick-ai/multimodal-agents-course): Multimodal Agents
                
                **Note:** This is a research demo. Not for clinical use.
                """,
                elem_id="subtitle"
            )
        
        return interface


def main():
    """Launch the demo interface"""
    
    print("="*70)
    print(" 🏥 Medical AI Agent Demo ")
    print("="*70)
    
    # Check for API keys
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: No API keys found!")
        print("Set GROQ_API_KEY or OPENAI_API_KEY environment variable")
        print("\nExample:")
        print("  export GROQ_API_KEY='your-key-here'")
        print()
    
    # Initialize demo
    try:
        demo = MedicalAgentDemo()
        interface = demo.create_interface()
        
        # Launch
        print("\nStarting demo interface...")
        print("\n🌐 Opening in browser...")
        print("\nPress Ctrl+C to stop\n")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Failed to start demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
