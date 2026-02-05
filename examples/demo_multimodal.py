#!/usr/bin/env python3
"""
Demo of multimodal medical processing.
Shows image analysis, diagram generation, and multimodal embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.tools.multimodal import (
    MultimodalProcessor,
    MultimodalInput,
    ImageProcessor,
    MultimodalEmbedder,
    process_multimodal
)
from medassist.services.mcp_server import MCPServer
from medassist.services.mcp_client import MCPClient


def demo_image_processing():
    """Demonstrate medical image processing"""
    print("=" * 70)
    print("MEDICAL IMAGE PROCESSING DEMO")
    print("=" * 70)
    
    processor = ImageProcessor()
    
    # Show supported formats
    print("\n[SUPPORTED FORMATS]")
    print(f"  {', '.join(processor.supported_formats)}")
    
    # Placeholder analysis (would use actual images in production)
    print("\n[IMAGE ANALYSIS]")
    print("  Analyzing chest X-ray...")
    result = processor.analyze_medical_image(
        image=b"placeholder_image_bytes",
        image_type="xray"
    )
    print(f"  Image type: {result['image_type']}")
    print(f"  Requires vision model: {result['requires_vision_model']}")
    print(f"  Suggested models:")
    for model in result['suggested_models']:
        print(f"    - {model}")


def demo_diagram_generation():
    """Demonstrate medical diagram generation"""
    print("\n" + "=" * 70)
    print("MEDICAL DIAGRAM GENERATION DEMO")
    print("=" * 70)
    
    processor = ImageProcessor()
    
    # Generate anatomy diagram
    print("\n[GENERATE ANATOMY DIAGRAM]")
    print("  Description: 'Human heart with labeled chambers and valves'")
    result = processor.generate_medical_diagram(
        description="Human heart with labeled chambers and valves",
        diagram_type="anatomy"
    )
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  Suggested models:")
    for model in result['suggested_models']:
        print(f"    - {model}")
    
    # Generate mechanism diagram
    print("\n[GENERATE MECHANISM DIAGRAM]")
    print("  Description: 'Insulin signaling pathway in diabetes'")
    result = processor.generate_medical_diagram(
        description="Insulin signaling pathway in diabetes",
        diagram_type="mechanism"
    )
    print(f"  Status: {result['status']}")
    print(f"  Diagram type: {result['diagram_type']}")


def demo_multimodal_embeddings():
    """Demonstrate multimodal embeddings"""
    print("\n" + "=" * 70)
    print("MULTIMODAL EMBEDDINGS DEMO")
    print("=" * 70)
    
    embedder = MultimodalEmbedder()
    
    # Text embedding
    print("\n[TEXT EMBEDDING]")
    text = "Patient has diabetes with symptoms of increased thirst and fatigue"
    text_emb = embedder.embed_text(text)
    print(f"  Text: '{text[:50]}...'")
    print(f"  Embedding dimension: {len(text_emb)}")
    
    # Image embedding
    print("\n[IMAGE EMBEDDING]")
    image_emb = embedder.embed_image(b"placeholder_image")
    print(f"  Image embedding dimension: {len(image_emb)}")
    
    # Multimodal fusion
    print("\n[MULTIMODAL FUSION]")
    result = embedder.embed_multimodal(
        text=text,
        image=b"placeholder_image"
    )
    print(f"  Modalities: {result['modalities']}")
    print(f"  Dimensions:")
    for modality, dim in result['dimensions'].items():
        print(f"    {modality}: {dim}")


def demo_multimodal_pipeline():
    """Demonstrate complete multimodal pipeline"""
    print("\n" + "=" * 70)
    print("COMPLETE MULTIMODAL PIPELINE DEMO")
    print("=" * 70)
    
    # Process multimodal input
    print("\n[PROCESS MULTIMODAL INPUT]")
    
    input_data = MultimodalInput(
        text="Chest X-ray shows infiltrates consistent with pneumonia",
        image=b"placeholder_xray_image",
        metadata={"image_type": "xray", "patient_id": "P12345"}
    )
    
    print(f"  Modalities: {[m.value for m in input_data.get_modalities()]}")
    print(f"  Is multimodal: {input_data.is_multimodal()}")
    
    processor = MultimodalProcessor()
    result = processor.process_multimodal_input(input_data)
    
    print(f"\n  Processing results:")
    print(f"    Modalities processed: {result['modalities']}")
    print(f"    Is multimodal: {result['is_multimodal']}")
    
    if 'text' in result['analysis']:
        print(f"    Text analysis:")
        print(f"      Length: {result['analysis']['text']['length']}")
    
    if 'image' in result['analysis']:
        print(f"    Image analysis:")
        print(f"      Type: {result['analysis']['image']['image_type']}")
        print(f"      Requires model: {result['analysis']['image']['requires_vision_model']}")


def demo_mcp_multimodal():
    """Demonstrate MCP Server with multimodal tools"""
    print("\n" + "=" * 70)
    print("MCP SERVER MULTIMODAL TOOLS DEMO")
    print("=" * 70)
    
    # Initialize MCP Server with multimodal support
    mcp_server = MCPServer()
    client = MCPClient(server=mcp_server)
    
    # List tools
    print("\n[MCP TOOLS]")
    capabilities = mcp_server.get_capabilities()
    print(f"  Total tools: {len(capabilities['tools'])}")
    
    multimodal_tools = [
        t for t in capabilities['tools']
        if 'image' in t['name'] or 'generate' in t['name']
    ]
    print(f"  Multimodal tools:")
    for tool in multimodal_tools:
        print(f"    - {tool['name']}: {tool['description']}")
    
    # Test image generation
    print("\n[TEST IMAGE GENERATION TOOL]")
    response = client.send_request(
        tool="generate_image",
        parameters={
            "description": "Diagram of respiratory system",
            "diagram_type": "anatomy"
        }
    )
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Result: {response.result['message']}")
    
    # Test image analysis
    print("\n[TEST IMAGE ANALYSIS TOOL]")
    response = client.send_request(
        tool="analyze_image",
        parameters={
            "image": b"placeholder_ct_scan",
            "image_type": "ct"
        }
    )
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Image type: {response.result['image_type']}")
        print(f"  Requires vision model: {response.result['requires_vision_model']}")


def demo_production_integration():
    """Show production integration notes"""
    print("\n" + "=" * 70)
    print("PRODUCTION INTEGRATION NOTES")
    print("=" * 70)
    
    print("\n[REQUIRED COMPONENTS FOR FULL MULTIMODAL]")
    print("""
  Vision Models:
    • CheXNet - Chest X-ray interpretation
    • BiomedCLIP - Multi-modal medical understanding
    • MedSAM - Medical image segmentation
    • RadImageNet - Radiology-specific pretrained model
    
  Generative Models:
    • DALL-E 3 / GPT-4 Vision - General medical illustrations
    • Stable Diffusion XL - Custom medical diagrams
    • BiomedGPT - Medical-specific generation
    
  Embedding Models:
    • PubMedBERT - Medical text embeddings
    • BioLinkBERT - Biomedical entity linking
    • Vision Transformer (ViT) - Medical image embeddings
    
  Integration:
    • Hugging Face Transformers - Model loading
    • PyTorch / TensorFlow - Inference
    • ONNX Runtime - Optimized deployment
    • Ray Serve - Distributed serving
    """)
    
    print("\n[CURRENT STATUS]")
    print("  [OK] Multimodal architecture implemented")
    print("  [OK] Image processing pipeline")
    print("  [OK] Diagram generation interface")
    print("  [OK] Multimodal embedding framework")
    print("  [OK] MCP tools integration")
    print("  [PLACEHOLDER] Vision model inference (needs model integration)")
    print("  [PLACEHOLDER] Image generation (needs generative model)")
    
    print("\n[API ENDPOINTS]")
    print("  POST /mcp - Tool: 'generate_image'")
    print("  POST /mcp - Tool: 'analyze_image'")
    print("  POST /ingest/document - Supports {'text': '...', 'images': [...]}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTIMODAL MEDICAL AI DEMO")
    print("Text + Image Processing for Medical Applications")
    print("=" * 70)
    
    # Run all demos
    demo_image_processing()
    demo_diagram_generation()
    demo_multimodal_embeddings()
    demo_multimodal_pipeline()
    demo_mcp_multimodal()
    demo_production_integration()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nFeatures demonstrated:")
    print("  [OK] Medical image processing (X-ray, CT, MRI)")
    print("  [OK] Medical diagram generation")
    print("  [OK] Multimodal embeddings (text + image)")
    print("  [OK] MCP Server multimodal tools")
    print("  [OK] Document ingestion with images")
    print("\nNote: Vision model integration requires additional setup")
    print("See docs/MCP_ARCHITECTURE.md for production deployment")
