#!/usr/bin/env python3
"""
Agentic Workflow Demo
Shows multi-agent collaboration for medical QA
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medassist.agentic_orchestrator import AgenticMedicalOrchestrator


def print_section(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print('='*80)


def main():
    print_section("MULTI-AGENT MEDICAL QA SYSTEM")
    print("\nSpecialized agents collaborate to answer medical questions")
    print("Features: Knowledge graph, evidence search, diagnostic reasoning, validation")
    
    # Initialize orchestrator
    print("\nInitializing Agentic Orchestrator...")
    orchestrator = AgenticMedicalOrchestrator(
        use_memory_graph=True,
        load_sample_graph=True
    )
    
    stats = orchestrator.get_statistics()
    print(f"[OK] {stats['agent_count']} agents initialized")
    print(f"[OK] Knowledge Graph: {stats['knowledge_graph']['num_nodes']} nodes, {stats['knowledge_graph']['num_edges']} edges")
    print(f"[OK] Agents: {', '.join(stats['agents'])}")
    
    # Test queries
    test_queries = [
        "What causes numbness in diabetic patients?",
        "How does diabetes affect vision and what are the treatments?",
        "What biomarkers are used to diagnose diabetes?"
    ]
    
    print_section("RUNNING TEST QUERIES")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('─'*80)
        
        # Process query
        response = orchestrator.process(query)
        
        # Display results
        print(f"\nAnswer:")
        print(response.answer)
        
        print(f"\nAgents Used: {', '.join(response.agents_used)}")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Processing Time: {response.processing_time:.3f}s")
        
        # Show workflow trace
        print(f"\nWorkflow Trace:")
        for line in response.workflow_trace.split('\n')[:15]:  # First 15 lines
            print(line)
        if response.workflow_trace.count('\n') > 15:
            print("... (trace truncated)")
        
        # Show findings summary
        print(f"\nFindings Summary:")
        for key, value in response.findings.items():
            if isinstance(value, dict):
                print(f"  • {key}: {len(value)} items")
            else:
                print(f"  • {key}: {value}")
    
    # Final summary
    print_section("SYSTEM SUMMARY")
    print(f"\nProcessed {len(test_queries)} queries successfully")
    print(f"Knowledge Graph: {stats['knowledge_graph']['num_nodes']} medical entities")
    print(f"Multi-Agent System: {stats['agent_count']} specialized agents")
    print(f"Tools: Knowledge Graph, Medical NER, PubMed Search, Validation")
    print(f"Workflow: Plan -> Execute -> Reflect -> Validate -> Answer")
    
    print("\nKey Features:")
    print("  - 5 specialized agents with distinct roles")
    print("  - Tool usage: knowledge graph, PubMed, medical NER")
    print("  - Agent reflection and self-correction")
    print("  - Automatic consistency validation")
    print("  - Confidence scoring on all findings")
    print("  - Complete audit trail of reasoning steps")
    
    print("\nAdvanced agentic workflow for medical question answering")
    print("="*80)


if __name__ == "__main__":
    main()
