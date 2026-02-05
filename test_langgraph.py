#!/usr/bin/env python3
"""
Test LangGraph Medical Orchestrator
Quick test để verify workflow hoạt động
"""

import os
from dotenv import load_dotenv
load_dotenv()

from simple_agent import SimpleAgent

def test_basic_query():
    """Test basic medical query"""
    print("\n" + "="*80)
    print("TEST 1: Basic Medical Query")
    print("="*80 + "\n")
    
    agent = SimpleAgent(use_memory_graph=True, load_sample_graph=True)
    
    question = "What is the first-line treatment for type 2 diabetes?"
    print(f"Question: {question}\n")
    
    result = agent.ask(question)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    print(f"Agents Used: {', '.join(result['agents_used'])}")
    
    # Show workflow trace
    print(f"\nWorkflow Trace:\n{result['workflow_trace']}")
    
    # Show findings
    print(f"\nFindings:")
    findings = result['findings']
    print(f"  - Entities: {len(findings.get('entities', []))}")
    print(f"  - Knowledge Paths: {len(findings.get('knowledge_paths', []))}")
    print(f"  - Symptoms: {len(findings.get('symptoms', []))}")
    print(f"  - Differential Diagnosis: {len(findings.get('differential_diagnosis', []))}")
    print(f"  - Treatments: {len(findings.get('treatments', []))}")
    print(f"  - Evidence: {len(findings.get('evidence', []))}")


def test_diagnostic_query():
    """Test diagnostic query with symptoms"""
    print("\n" + "="*80)
    print("TEST 2: Diagnostic Query")
    print("="*80 + "\n")
    
    agent = SimpleAgent(use_memory_graph=True, load_sample_graph=True)
    
    question = "A patient presents with chest pain and shortness of breath. What could be the diagnosis?"
    print(f"Question: {question}\n")
    
    result = agent.ask(question)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Agents Used: {', '.join(result['agents_used'])}")
    
    # Show differential diagnosis
    findings = result['findings']
    if findings.get('differential_diagnosis'):
        print(f"\nDifferential Diagnosis:")
        for i, dx in enumerate(findings['differential_diagnosis'][:3], 1):
            print(f"  {i}. {dx}")


def test_knowledge_graph_query():
    """Test knowledge graph retrieval"""
    print("\n" + "="*80)
    print("TEST 3: Knowledge Graph Query")
    print("="*80 + "\n")
    
    agent = SimpleAgent(use_memory_graph=True, load_sample_graph=True)
    
    question = "What medications treat hypertension?"
    print(f"Question: {question}\n")
    
    result = agent.ask(question)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    
    # Show entities and knowledge paths
    findings = result['findings']
    if findings.get('entities'):
        print(f"\nEntities Found:")
        for entity in findings['entities'][:5]:
            print(f"  - {entity['text']} ({entity['type']})")
    
    if findings.get('knowledge_paths'):
        print(f"\nKnowledge Paths:")
        for path in findings['knowledge_paths'][:3]:
            print(f"  Source: {path['source']}")
            for rel in path['related'][:3]:
                print(f"    → {rel['relationship']}: {rel['entity']}")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("LangGraph Medical Orchestrator - Test Suite")
    print("="*80)
    
    try:
        test_basic_query()
        test_diagnostic_query()
        test_knowledge_graph_query()
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
