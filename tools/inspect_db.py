#!/usr/bin/env python3
"""
ChromaDB Database Inspector

This utility helps you inspect all records in your ChromaDB database.
Use this to debug, explore, and understand what's stored in your vector database.
"""

import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
import os
import sys
from dotenv import load_dotenv

# Get the parent directory (project root)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv(os.path.join(parent_dir, '.env'))

# Set ChromaDB path for tools directory
os.environ.setdefault('CHROMADB_PERSIST_DIRECTORY', os.path.join(parent_dir, 'data', 'chromadb'))

from src.storage.vector_storage import VectorStorage
from config import settings, logger

class ChromaDBInspector:
    """Inspector for ChromaDB records with various viewing options."""
    
    def __init__(self):
        """Initialize the database inspector."""
        try:
            self.storage = VectorStorage()
            print(f"✅ Connected to ChromaDB")
            print(f"📂 Database path: {settings.CHROMADB_PERSIST_DIRECTORY}")
            print(f"📚 Collection: {settings.CHROMADB_COLLECTION_NAME}")
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB: {e}")
            sys.exit(1)
    
    def show_collection_stats(self):
        """Display basic collection statistics."""
        print("\n" + "="*60)
        print("📊 COLLECTION STATISTICS")
        print("="*60)
        
        stats = self.storage.get_collection_stats()
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    def list_all_records(self, limit: Optional[int] = None, show_content: bool = True):
        """
        List all records in the database.
        
        Args:
            limit: Maximum number of records to show (None for all)
            show_content: Whether to show full content or just metadata
        """
        print("\n" + "="*60)
        print("📝 ALL RECORDS")
        print("="*60)
        
        try:
            # Get all records from ChromaDB
            results = self.storage.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not results['ids']:
                print("No records found in the database.")
                return
            
            total_records = len(results['ids'])
            display_count = limit if limit and limit < total_records else total_records
            
            print(f"Total records: {total_records}")
            print(f"Displaying: {display_count}")
            print("-" * 60)
            
            for i in range(display_count):
                record_id = results['ids'][i]
                document = results['documents'][i] if results['documents'] else ""
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                
                print(f"\n🔍 Record {i+1}/{display_count}")
                print(f"ID: {record_id}")
                
                # Show metadata
                if metadata:
                    print(f"Title: {metadata.get('title', 'N/A')}")
                    print(f"URL: {metadata.get('source_url', 'N/A')}")
                    print(f"Summary: {metadata.get('summary', 'N/A')[:100]}...")
                    
                    # Parse and show topics
                    topics_str = metadata.get('topics', '[]')
                    try:
                        topics = json.loads(topics_str)
                        print(f"Topics: {', '.join(topics) if topics else 'None'}")
                    except:
                        print(f"Topics: {topics_str}")
                    
                    print(f"Extracted: {metadata.get('extracted_at', 'N/A')}")
                
                # Show document content if requested
                if show_content and document:
                    print(f"Document Text Preview: {document[:200]}...")
                
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error listing records: {e}")
    
    def search_records(self, query: str, limit: int = 5):
        """
        Search records using semantic search.
        
        Args:
            query: Search query
            limit: Number of results to show
        """
        print(f"\n🔍 SEARCH RESULTS for: '{query}'")
        print("="*60)
        
        try:
            results = self.storage.search_articles(query, limit=limit)
            
            if not results:
                print("No results found.")
                return
            
            for i, result in enumerate(results, 1):
                print(f"\n📰 Result {i}")
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"URL: {result.get('source_url', 'N/A')}")
                print(f"Score: {result.get('similarity_score', 0):.3f}")
                print(f"Summary: {result.get('summary', 'N/A')[:150]}...")
                print(f"Topics: {', '.join(result.get('topics', []))}")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error searching records: {e}")
    
    def show_record_by_id(self, record_id: str):
        """
        Show detailed information for a specific record.
        
        Args:
            record_id: The ID of the record to show
        """
        print(f"\n📋 RECORD DETAILS: {record_id}")
        print("="*60)
        
        try:
            results = self.storage.collection.get(
                ids=[record_id],
                include=['documents', 'metadatas']
            )
            
            if not results['ids']:
                print(f"Record with ID '{record_id}' not found.")
                return
            
            document = results['documents'][0] if results['documents'] else ""
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            # Show all metadata
            print("📊 METADATA:")
            for key, value in metadata.items():
                if key == 'topics':
                    try:
                        topics = json.loads(value)
                        print(f"  {key}: {topics}")
                    except:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
            
            # Show document content
            print(f"\n📄 DOCUMENT TEXT:")
            print(document)
            
        except Exception as e:
            print(f"❌ Error showing record: {e}")
    
    def show_topics_summary(self):
        """Show a summary of all topics in the database."""
        print("\n📊 TOPICS SUMMARY")
        print("="*60)
        
        try:
            results = self.storage.collection.get(
                include=['metadatas']
            )
            
            if not results['metadatas']:
                print("No records found.")
                return
            
            # Count all topics
            topic_counts = {}
            for metadata in results['metadatas']:
                topics_str = metadata.get('topics', '[]')
                try:
                    topics = json.loads(topics_str)
                    for topic in topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                except:
                    continue
            
            # Sort by frequency
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            
            print(f"Total unique topics: {len(sorted_topics)}")
            print("\nTop topics by frequency:")
            for i, (topic, count) in enumerate(sorted_topics[:20], 1):
                print(f"{i:2d}. {topic}: {count}")
                
        except Exception as e:
            print(f"❌ Error analyzing topics: {e}")
    
    def export_to_json(self, filename: str = None):
        """
        Export all records to a JSON file.
        
        Args:
            filename: Output filename (default: db_export_TIMESTAMP.json)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"db_export_{timestamp}.json"
        
        print(f"\n💾 EXPORTING to {filename}")
        print("="*60)
        
        try:
            results = self.storage.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not results['ids']:
                print("No records to export.")
                return
            
            # Prepare export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'collection_name': settings.CHROMADB_COLLECTION_NAME,
                'total_records': len(results['ids']),
                'records': []
            }
            
            for i in range(len(results['ids'])):
                record = {
                    'id': results['ids'][i],
                    'document': results['documents'][i] if results['documents'] else "",
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                }
                export_data['records'].append(record)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported {len(results['ids'])} records to {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting records: {e}")

def main():
    """Main CLI interface for database inspection."""
    print("🔍 ChromaDB Database Inspector")
    print("Type 'help' for available commands")
    
    inspector = ChromaDBInspector()
    
    while True:
        print("\n" + "-"*60)
        command = input("inspector> ").strip().lower()
        
        if command in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        elif command == 'help':
            print("""
Available commands:
  stats          - Show collection statistics
  list [N]       - List all records (optionally limit to N records)
  list-meta [N]  - List records with metadata only (no content)
  search <query> - Search records by query
  show <id>      - Show detailed record by ID
  topics         - Show topics summary
  export [file]  - Export all records to JSON
  help           - Show this help
  quit/exit      - Exit the inspector
            """)
        
        elif command == 'stats':
            inspector.show_collection_stats()
        
        elif command.startswith('list'):
            parts = command.split()
            limit = None
            show_content = 'meta' not in command
            
            if len(parts) > 1 and parts[-1].isdigit():
                limit = int(parts[-1])
            
            inspector.list_all_records(limit=limit, show_content=show_content)
        
        elif command.startswith('search '):
            query = command[7:]  # Remove 'search '
            inspector.search_records(query)
        
        elif command.startswith('show '):
            record_id = command[5:]  # Remove 'show '
            inspector.show_record_by_id(record_id)
        
        elif command == 'topics':
            inspector.show_topics_summary()
        
        elif command.startswith('export'):
            parts = command.split()
            filename = parts[1] if len(parts) > 1 else None
            inspector.export_to_json(filename)
        
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands")

if __name__ == "__main__":
    main()
