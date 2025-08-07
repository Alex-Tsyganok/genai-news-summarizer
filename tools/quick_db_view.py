#!/usr/bin/env python3
"""
Quick ChromaDB viewer - Simple script to see all records
"""

import sys
import json
from pprint import pprint

# Handle Windows console encoding for piped operations
if sys.platform == "win32":
    import codecs
    # Fix encoding when output is piped (findstr, Select-Object, etc.)
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding='utf-8')
    if not sys.stderr.isatty():
        sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
import os
sys.path.append('..')
sys.path.append('../src')

# Set ChromaDB path for tools directory
os.environ.setdefault('CHROMADB_PERSIST_DIRECTORY', '../data/chromadb')

try:
    from src.storage.vector_storage import VectorStorage
    from config import settings
    
    print("🔍 Connecting to ChromaDB...")
    storage = VectorStorage()
    
    print(f"📂 Database: {settings.CHROMADB_PERSIST_DIRECTORY}")
    print(f"📚 Collection: {settings.CHROMADB_COLLECTION_NAME}")
    
    # Get collection stats
    stats = storage.get_collection_stats()
    print(f"📊 Total articles: {stats['total_articles']}")
    
    if stats['total_articles'] == 0:
        print("❌ No records found in database")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("📝 ALL RECORDS")
    print("="*80)
    
    # Get all records
    results = storage.collection.get(
        include=['documents', 'metadatas']
    )
    
    for i, (record_id, document, metadata) in enumerate(
        zip(results['ids'], results['documents'], results['metadatas'])
    ):
        print(f"\n🔍 Record {i+1}")
        print(f"ID: {record_id}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"URL: {metadata.get('source_url', 'N/A')}")
        print(f"Summary: {metadata.get('summary', 'N/A')[:100]}...")
        
        # Parse topics
        topics_str = metadata.get('topics', '[]')
        try:
            topics = json.loads(topics_str)
            print(f"Topics: {', '.join(topics) if topics else 'None'}")
        except:
            print(f"Topics: {topics_str}")
        
        print(f"Extracted: {metadata.get('extracted_at', 'N/A')}")
        print(f"Document preview: {document[:150]}...")
        print("-" * 60)

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you're in the project root directory and ChromaDB is properly set up.")
