#!/usr/bin/env python3
"""
Demo of MCP (Model Context Protocol) implementation.
Shows GraphRAG ingestion pipeline and MCP server/client usage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.services.mcp_server import MCPServer
from medassist.services.mcp_client import MCPClient, InteractiveMCPClient
from medassist.services.ingestion_pipeline import IngestionPipeline
from medassist.models.knowledge_graph import MedicalKnowledgeGraph


def demo_ingestion_pipeline():
    """Demonstrate document ingestion pipeline"""
    print("=" * 60)
    print("Document Ingestion Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    kg = MedicalKnowledgeGraph()
    pipeline = IngestionPipeline(kg=kg)
    
    # Sample medical documents
    documents = [
        {
            "id": "doc1",
            "content": """
            Type 2 diabetes mellitus is a chronic metabolic disorder characterized 
            by high blood sugar levels. Common symptoms include increased thirst, 
            frequent urination, and fatigue. The disease is primarily caused by 
            insulin resistance and inadequate insulin production. Treatment options 
            include lifestyle modifications, oral medications like metformin, and 
            insulin therapy in advanced cases.
            """
        },
        {
            "id": "doc2",
            "content": """
            Hypertension, or high blood pressure, is a condition where blood pressure 
            in the arteries is persistently elevated. It can lead to serious 
            complications such as heart disease and stroke. Symptoms may include 
            headaches and shortness of breath. Treatment involves lifestyle changes 
            and medications such as ACE inhibitors and beta blockers.
            """
        }
    ]
    
    # Ingest documents
    print("\n[INGESTION] Processing documents...")
    kg_objects = pipeline.ingest_batch(documents)
    
    print(f"\n[RESULT] Processed {len(kg_objects)} documents")
    
    for i, kg_obj in enumerate(kg_objects, 1):
        print(f"\n  Document {i} ({kg_obj.source_document}):")
        print(f"    Entities: {len(kg_obj.entities)}")
        print(f"    Relationships: {len(kg_obj.relationships)}")
        
        # Show some entities
        if kg_obj.entities:
            print(f"    Sample entities:")
            for entity in kg_obj.entities[:3]:
                print(f"      - {entity['name']} ({entity['type']})")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"\n[STATISTICS]")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Entity types: {', '.join(stats['entity_types'])}")


def demo_mcp_server():
    """Demonstrate MCP Server tools and skills"""
    print("\n" + "=" * 60)
    print("MCP Server Demo")
    print("=" * 60)
    
    # Initialize MCP Server
    kg = MedicalKnowledgeGraph()
    mcp_server = MCPServer(knowledge_graph=kg)
    
    # Show capabilities
    print("\n[CAPABILITIES]")
    capabilities = mcp_server.get_capabilities()
    
    print(f"\n  Available Tools ({len(capabilities['tools'])}):")
    for tool in capabilities['tools']:
        print(f"    - {tool['name']}: {tool['description']}")
    
    print(f"\n  Available Skills ({len(capabilities['skills'])}):")
    for skill in capabilities['skills']:
        print(f"    - {skill['name']}: {skill['description']}")


def demo_mcp_client():
    """Demonstrate MCP Client usage"""
    print("\n" + "=" * 60)
    print("MCP Client Demo")
    print("=" * 60)
    
    # Initialize server and client
    kg = MedicalKnowledgeGraph()
    mcp_server = MCPServer(knowledge_graph=kg)
    client = MCPClient(server=mcp_server)
    interactive = InteractiveMCPClient(client)
    
    # Test 1: KG Search
    print("\n[TEST 1] Knowledge Graph Search")
    print("  Query: 'diabetes symptoms'")
    response = client.kg_search("diabetes symptoms", max_depth=2)
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Results: {len(response.result)} paths found")
    
    # Test 2: Update Memory
    print("\n[TEST 2] Update Memory Skill")
    content = "Aspirin is used to treat pain and reduce fever. It also prevents blood clots."
    print(f"  Content: '{content}'")
    response = client.update_memory(content)
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Entities extracted: {response.result.get('entities_extracted', 0)}")
    
    # Test 3: Generate Content
    print("\n[TEST 3] Write Content Skill")
    print("  Topic: 'diabetes treatment'")
    response = client.write_content("diabetes treatment", include_evidence=True)
    print(f"  Success: {response.success}")
    if response.success and response.result:
        kg_data = response.result.get('knowledge_graph_data', [])
        evidence = response.result.get('evidence', [])
        print(f"  Knowledge graph results: {len(kg_data)}")
        print(f"  Evidence articles: {len(evidence)}")
    
    # Test 4: Interactive Client
    print("\n[TEST 4] Interactive Client - Medical Query")
    result = interactive.query_medical_topic(
        "What causes hypertension?",
        include_evidence=False
    )
    print(f"  Question: {result['question']}")
    print(f"  KG Success: {result['kg_success']}")
    print(f"  KG Results: {len(result['knowledge_graph'])}")


def demo_full_workflow():
    """Demonstrate complete GraphRAG + MCP workflow"""
    print("\n" + "=" * 60)
    print("Complete GraphRAG + MCP Workflow")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\n[STEP 1] Initialize components")
    kg = MedicalKnowledgeGraph()
    pipeline = IngestionPipeline(kg=kg)
    mcp_server = MCPServer(knowledge_graph=kg)
    client = MCPClient(server=mcp_server)
    
    # Step 2: Ingest medical knowledge
    print("\n[STEP 2] Ingest medical documents")
    document = {
        "id": "covid19",
        "content": """
        COVID-19 is caused by the SARS-CoV-2 virus. Common symptoms include 
        fever, cough, and loss of taste or smell. Severe cases can lead to 
        pneumonia and respiratory failure. Vaccines have been developed to 
        prevent infection. Treatment includes supportive care and antiviral 
        medications like remdesivir.
        """
    }
    kg_obj = pipeline.ingest_document(document["content"], document["id"])
    print(f"  Extracted {len(kg_obj.entities)} entities")
    print(f"  Extracted {len(kg_obj.relationships)} relationships")
    
    # Step 3: Search knowledge using MCP
    print("\n[STEP 3] Search knowledge via MCP")
    response = client.kg_search("COVID-19 symptoms", max_depth=2)
    print(f"  Found {len(response.result) if response.success else 0} results")
    
    # Step 4: Retrieve evidence
    print("\n[STEP 4] Retrieve scientific evidence")
    response = client.web_search("COVID-19 treatment", max_results=3)
    if response.success and response.result:
        print(f"  Found {len(response.result)} PubMed articles")
    
    # Step 5: Generate comprehensive answer
    print("\n[STEP 5] Generate comprehensive answer")
    response = client.write_content("COVID-19", include_evidence=True)
    if response.success:
        print("  Generated medical report")
        print(f"  Topic: {response.result.get('topic')}")
        print(f"  Summary: {response.result.get('summary')}")
    
    print("\n[WORKFLOW COMPLETE]")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MCP + GraphRAG Pipeline Demo")
    print("Model Context Protocol Implementation")
    print("=" * 60)
    
    # Run all demos
    demo_ingestion_pipeline()
    demo_mcp_server()
    demo_mcp_client()
    demo_full_workflow()
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print("\nFeatures demonstrated:")
    print("  [OK] Document ingestion pipeline")
    print("  [OK] Entity and relationship extraction")
    print("  [OK] MCP Server with tools and skills")
    print("  [OK] MCP Client interface")
    print("  [OK] Complete GraphRAG workflow")
