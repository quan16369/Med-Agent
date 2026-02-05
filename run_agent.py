#!/usr/bin/env python3
"""
Simple Agent Runner - Interactive CLI
Chạy agent đơn giản qua command line, không cần API server
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_agent import SimpleAgent
from medassist.logging_utils import get_logger

logger = get_logger(__name__)


def interactive_mode(agent: SimpleAgent):
    """Interactive mode - chat với agent"""
    print("\n" + "=" * 70)
    print("🤖 Medical Agent - Interactive Mode")
    print("=" * 70)
    print("Commands:")
    print("  /ask <question>          - Hỏi câu hỏi medical")
    print("  /ingest <file.txt>       - Ingest document vào KG")
    print("  /explore <entity>        - Explore entity trong KG")
    print("  /image <path> <question> - Hỏi với medical image")
    print("  /quit                    - Thoát")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/quit":
                print("👋 Goodbye!")
                break
            
            # Parse command
            if user_input.startswith("/ask "):
                question = user_input[5:]
                print(f"\n🤔 Processing: {question}")
                result = agent.ask(question)
                print(f"\n🤖 Agent: {result.get('answer', 'No answer')}")
                
                if 'entities' in result:
                    print(f"\n📊 Entities: {len(result['entities'])}")
                if 'relationships' in result:
                    print(f"🔗 Relationships: {len(result['relationships'])}")
            
            elif user_input.startswith("/ingest "):
                file_path = user_input[8:].strip()
                if Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        text = f.read()
                    print(f"\n📄 Ingesting: {file_path}")
                    result = agent.ingest_document(text, doc_id=file_path)
                    print(f"✅ Ingested: {result.get('entity_count', 0)} entities, {result.get('relationship_count', 0)} relationships")
                else:
                    print(f"❌ File not found: {file_path}")
            
            elif user_input.startswith("/explore "):
                entity = user_input[9:].strip()
                print(f"\n🔍 Exploring: {entity}")
                result = agent.explore_knowledge_graph(entity, max_depth=2)
                
                if result.get('success', True):
                    related = result.get('related_entities', [])
                    print(f"✅ Found {len(related)} related entities:")
                    for ent in related[:10]:  # Show first 10
                        print(f"  - {ent.get('name', 'N/A')} ({ent.get('type', 'N/A')})")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown')}")
            
            elif user_input.startswith("/image "):
                parts = user_input[7:].split(maxsplit=1)
                if len(parts) == 2:
                    image_path, question = parts
                    if Path(image_path).exists():
                        print(f"\n🖼️ Processing image: {image_path}")
                        result = agent.ask_with_image(question, image_path=image_path)
                        print(f"\n🤖 Agent: {result.get('answer', 'No answer')}")
                    else:
                        print(f"❌ Image not found: {image_path}")
                else:
                    print("❌ Usage: /image <path> <question>")
            
            else:
                # Default: treat as question
                print(f"\n🤔 Processing: {user_input}")
                result = agent.ask(user_input)
                print(f"\n🤖 Agent: {result.get('answer', 'No answer')}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}", exc_info=True)


def single_question(agent: SimpleAgent, question: str):
    """Single question mode"""
    print(f"\n💬 Question: {question}")
    result = agent.ask(question)
    print(f"\n🤖 Answer: {result.get('answer', 'No answer')}")
    
    if 'reasoning' in result:
        print(f"\n🧠 Reasoning:\n{result['reasoning']}")
    
    if 'entities' in result:
        print(f"\n📊 Entities found: {len(result['entities'])}")
        for ent in result['entities'][:5]:
            print(f"  - {ent.get('name', 'N/A')} ({ent.get('type', 'N/A')})")
    
    if 'relationships' in result:
        print(f"\n🔗 Relationships found: {len(result['relationships'])}")


def ingest_file(agent: SimpleAgent, file_path: str):
    """Ingest file mode"""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📄 Reading: {file_path}")
    with open(path, 'r') as f:
        text = f.read()
    
    print(f"📊 Content: {len(text)} characters")
    print(f"🔄 Processing...")
    
    result = agent.ingest_document(
        text=text,
        doc_id=str(path.name),
        metadata={"file_path": str(path)}
    )
    
    if result.get('success', True):
        print(f"\n✅ Ingestion complete!")
        print(f"  Entities: {result.get('entity_count', 0)}")
        print(f"  Relationships: {result.get('relationship_count', 0)}")
    else:
        print(f"\n❌ Ingestion failed: {result.get('error', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Medical Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_agent.py
  
  # Single question
  python run_agent.py --ask "What are symptoms of diabetes?"
  
  # Ingest document
  python run_agent.py --ingest medical_paper.txt
  
  # With config
  python run_agent.py --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--ask', '-a',
        type=str,
        help='Single question mode'
    )
    
    parser.add_argument(
        '--ingest', '-i',
        type=str,
        help='Ingest document from file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    print("🚀 Initializing Medical Agent...")
    agent = SimpleAgent(config_path=args.config)
    print("✅ Agent ready!\n")
    
    # Determine mode
    if args.ask:
        single_question(agent, args.ask)
    elif args.ingest:
        ingest_file(agent, args.ingest)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
