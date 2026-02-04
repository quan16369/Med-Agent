"""
Example: Using multimodal API endpoints with medical images.
Based on Kubrick AI's multimodal patterns.
"""

import requests
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path


def encode_image_file(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def create_medical_query_with_image(
    question: str,
    image_path: str,
    image_type: str = "xray"
):
    """
    Send medical question with image to API.
    
    Example:
        question = "What abnormalities do you see in this chest X-ray?"
        image_path = "chest_xray.jpg"
        result = create_medical_query_with_image(question, image_path, "xray")
    """
    base_url = "http://localhost:8000"
    
    # Encode image
    image_base64 = encode_image_file(image_path)
    
    # Create request with image
    payload = {
        "question": question,
        "image_base64": image_base64,
        "image_metadata": {
            "image_type": image_type,
            "body_part": "chest" if image_type == "xray" else None
        },
        "include_trace": True
    }
    
    # Send request
    response = requests.post(f"{base_url}/query", json=payload)
    response.raise_for_status()
    
    return response.json()


def ingest_multimodal_document(
    text: str,
    image_paths: list[str],
    document_id: str = None
):
    """
    Ingest document with text and images.
    
    Example:
        text = "Patient presents with pneumonia. Chest X-ray shows infiltrates."
        images = ["xray_front.jpg", "xray_side.jpg"]
        result = ingest_multimodal_document(text, images, "case_001")
    """
    base_url = "http://localhost:8000"
    
    # Encode images
    images = []
    for img_path in image_paths:
        img_b64 = encode_image_file(img_path)
        images.append({
            "image_path": img_path,
            "image_base64": img_b64,
            "metadata": {
                "filename": Path(img_path).name
            }
        })
    
    # Create request
    payload = {
        "content": text,
        "document_id": document_id,
        "images": images
    }
    
    # Send request
    response = requests.post(f"{base_url}/ingest/document", json=payload)
    response.raise_for_status()
    
    return response.json()


def analyze_medical_image_via_mcp(
    image_path: str,
    image_type: str = "xray"
):
    """
    Analyze medical image using MCP tools.
    
    Example:
        result = analyze_medical_image_via_mcp("chest_xray.jpg", "xray")
    """
    base_url = "http://localhost:8000"
    
    # Encode image
    image_base64 = encode_image_file(image_path)
    
    # Call MCP tool
    payload = {
        "tool": "analyze_image",
        "arguments": {
            "image": image_base64,
            "image_type": image_type
        }
    }
    
    response = requests.post(f"{base_url}/mcp", json=payload)
    response.raise_for_status()
    
    return response.json()


def generate_medical_diagram_via_mcp(
    description: str,
    diagram_type: str = "anatomy"
):
    """
    Generate medical diagram using MCP tools.
    
    Example:
        result = generate_medical_diagram_via_mcp(
            "Heart anatomy with labeled chambers and valves",
            "anatomy"
        )
    """
    base_url = "http://localhost:8000"
    
    payload = {
        "tool": "generate_image",
        "arguments": {
            "description": description,
            "diagram_type": diagram_type
        }
    }
    
    response = requests.post(f"{base_url}/mcp", json=payload)
    response.raise_for_status()
    
    return response.json()


def demo_multimodal_workflow():
    """Complete multimodal workflow demo"""
    
    print("=== MedGemma Multimodal API Demo ===\n")
    
    # 1. Analyze image with question
    print("1. Analyzing chest X-ray with question...")
    # result = create_medical_query_with_image(
    #     "What abnormalities are visible in this X-ray?",
    #     "path/to/xray.jpg",
    #     "xray"
    # )
    # print(f"Answer: {result['answer']}\n")
    
    # 2. Ingest multimodal case study
    print("2. Ingesting multimodal case study...")
    # result = ingest_multimodal_document(
    #     "Patient presents with COVID-19 pneumonia. CT shows ground-glass opacities.",
    #     ["ct_scan_1.jpg", "ct_scan_2.jpg"],
    #     "covid_case_001"
    # )
    # print(f"Ingested: {result['entities_extracted']} entities\n")
    
    # 3. Analyze standalone image
    print("3. Analyzing image via MCP...")
    # result = analyze_medical_image_via_mcp("chest_xray.jpg", "xray")
    # print(f"Findings: {result}\n")
    
    # 4. Generate medical diagram
    print("4. Generating medical diagram...")
    # result = generate_medical_diagram_via_mcp(
    #     "Heart anatomy with chambers and valves labeled",
    #     "anatomy"
    # )
    # print(f"Generated: {result}\n")
    
    print("Demo complete! Uncomment code blocks to test with actual images.")


if __name__ == "__main__":
    # Run demo
    demo_multimodal_workflow()
    
    # Example usage patterns
    print("\n=== Usage Patterns ===\n")
    
    print("# 1. Medical question with image")
    print('result = create_medical_query_with_image(')
    print('    "Describe findings in this chest X-ray",')
    print('    "xray.jpg",')
    print('    "xray"')
    print(')\n')
    
    print("# 2. Ingest case with multiple images")
    print('result = ingest_multimodal_document(')
    print('    "Patient case description...",')
    print('    ["image1.jpg", "image2.jpg"],')
    print('    "case_id"')
    print(')\n')
    
    print("# 3. Analyze image via MCP")
    print('result = analyze_medical_image_via_mcp("scan.jpg", "ct")\n')
    
    print("# 4. Generate diagram via MCP")
    print('result = generate_medical_diagram_via_mcp(')
    print('    "Brain anatomy cross-section",')
    print('    "anatomy"')
    print(')')
