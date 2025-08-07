#!/usr/bin/env python3
"""
Direct ChromaDB exploration without custom wrappers
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import json
import os
import sys
from pprint import pprint

# Handle Windows console encoding for subprocess operations
if sys.platform == "win32":
    # Fix encoding when output is captured by subprocess
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding='utf-8')
    if not sys.stderr.isatty():
        sys.stderr.reconfigure(encoding='utf-8')

def explore_chromadb():
    """Directly explore ChromaDB contents."""
    
    # Database path (adjust if needed)
    db_path = "../data/chromadb"
    collection_name = "news_articles"
    
    if not os.path.exists(db_path):
        print(f"❌ Database path not found: {db_path}")
        print("Make sure you're in the project root directory")
        return
    
    try:
        print(f"🔍 Connecting to ChromaDB at: {db_path}")
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # List all collections
        collections = client.list_collections()
        print(f"📚 Available collections: {[c.name for c in collections]}")
        
        if not collections:
            print("❌ No collections found")
            return
        
        # Get the news articles collection (or first available)
        try:
            collection = client.get_collection(collection_name)
        except:
            collection = collections[0]
            collection_name = collection.name
            print(f"⚠️ Using collection: {collection_name}")
        
        # Get collection count
        count = collection.count()
        print(f"📊 Total records in '{collection_name}': {count}")
        
        if count == 0:
            print("❌ No records in collection")
            return
        
        print("\n" + "="*80)
        print("📝 COLLECTION CONTENTS")
        print("="*80)
        
        # Get all records
        results = collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        print(f"Retrieved {len(results['ids'])} records")
        
        # Show sample records
        sample_size = min(5, len(results['ids']))
        print(f"\nShowing first {sample_size} records:")
        
        for i in range(sample_size):
            print(f"\n🔍 Record {i+1}")
            print(f"ID: {results['ids'][i]}")
            
            # Show metadata
            if results['metadatas'] and i < len(results['metadatas']):
                metadata = results['metadatas'][i]
                print("📊 Metadata:")
                for key, value in metadata.items():
                    if key == 'topics' and isinstance(value, str):
                        try:
                            topics = json.loads(value)
                            print(f"  {key}: {topics}")
                        except:
                            print(f"  {key}: {value}")
                    else:
                        # Truncate long values
                        display_value = str(value)
                        if len(display_value) > 100:
                            display_value = display_value[:100] + "..."
                        print(f"  {key}: {display_value}")
            
            # Show document content
            if results['documents'] and i < len(results['documents']):
                doc = results['documents'][i]
                print(f"📄 Document (first 200 chars): {doc[:200]}...")
            
            # Show embedding info
            if results['embeddings'] is not None and i < len(results['embeddings']):
                embedding = results['embeddings'][i]
                print(f"🧮 Embedding dimension: {len(embedding) if embedding is not None else 'N/A'}")
            
            print("-" * 60)
        
        # Show all IDs for reference
        if len(results['ids']) > sample_size:
            print(f"\n📋 All record IDs ({len(results['ids'])} total):")
            for i, record_id in enumerate(results['ids']):
                print(f"  {i+1}. {record_id}")
        
        # Collection metadata
        print(f"\n📚 Collection metadata:")
        try:
            collection_metadata = collection.metadata
            if collection_metadata:
                pprint(collection_metadata)
            else:
                print("  No metadata available")
        except:
            print("  Could not retrieve collection metadata")
            
    except Exception as e:
        print(f"❌ Error exploring database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_chromadb()
