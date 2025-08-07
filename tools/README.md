# 🛠️ Database Inspection Tools

This directory contains utilities for inspecting, debugging, and exploring the ChromaDB vector database used by the AI News Summarizer. These tools help you understand what's stored in your database and troubleshoot issues.

## 📋 Available Tools

### 1. 🔍 Interactive Database Inspector (`inspect_db.py`)

**Most comprehensive tool** - Interactive command-line interface for database exploration.

```bash
cd tools
python inspect_db.py
```

**Features:**
- Interactive CLI with multiple commands
- Semantic search testing
- Export capabilities
- Detailed record inspection
- Topics analysis

**Available Commands:**
```
stats          - Show collection statistics
list [N]       - List all records (optionally limit to N records)
list-meta [N]  - List records with metadata only (no content)
search <query> - Search records by query
show <id>      - Show detailed record by ID
topics         - Show topics summary
export [file]  - Export all records to JSON
help           - Show this help
quit/exit      - Exit the inspector
```

**Example Session:**
```
inspector> stats
📊 COLLECTION STATISTICS
Total Articles: 12
Collection Name: news_articles

inspector> search artificial intelligence
🔍 SEARCH RESULTS for: 'artificial intelligence'
📰 Result 1
Title: The people who think AI might become conscious
Score: 0.892

inspector> list 3
📝 ALL RECORDS
Displaying: 3
🔍 Record 1/3
ID: article_260861
Title: The people who think AI might become conscious
```

### 2. ⚡ Quick Database Viewer (`quick_db_view.py`)

**Fastest overview** - Simple script that immediately shows all records.

```bash
cd tools
python quick_db_view.py
```

**Features:**
- No interaction needed
- Shows all records at once
- Quick statistics
- Minimal output for easy scanning

**Best for:**
- Quick health checks
- Immediate overview
- CI/CD pipelines
- Automated scripts

### 3. 🔧 Direct ChromaDB Explorer (`explore_chromadb.py`)

**Low-level access** - Direct ChromaDB exploration without custom wrappers.

```bash
cd tools
python explore_chromadb.py
```

**Features:**
- Direct ChromaDB client access
- Shows embedding dimensions
- Collection metadata
- Raw database structure
- Independent of project classes

**Best for:**
- Debugging ChromaDB issues
- Understanding raw data structure
- Bypassing custom code layers
- Database troubleshooting

## 🚀 Usage Examples

### Quick Health Check
```bash
# Fast overview of database contents
cd tools
python quick_db_view.py
```

### Deep Exploration
```bash
# Interactive exploration with search capabilities
cd tools
python inspect_db.py
```

### Troubleshooting ChromaDB
```bash
# Low-level database inspection
cd tools
python explore_chromadb.py
```

### Export Database for Backup
```bash
cd tools
python inspect_db.py
# Then in the inspector:
inspector> export backup_2025_08_07.json
```

### Search Testing
```bash
cd tools
python inspect_db.py
# Then test different queries:
inspector> search "artificial intelligence"
inspector> search "technology news"
inspector> search "machine learning"
```

## 📊 Understanding the Output

### Record Structure
Each record in your database contains:

```
🔍 Record ID: article_XXXXXX
📊 Metadata:
  - title: Article headline
  - source_url: Original article URL
  - summary: AI-generated summary
  - topics: JSON array of extracted topics
  - extracted_at: Timestamp of processing
  - meta_extraction_method: Tool used (newspaper3k/beautifulsoup)
  - meta_top_image: Featured image URL

📄 Document Text: Combined text used for embeddings
🧮 Embedding: 384-dimensional vector for semantic search
```

### Similarity Scores
- **0.9-1.0**: Highly relevant matches
- **0.7-0.9**: Good matches
- **0.5-0.7**: Moderate relevance
- **<0.5**: Low relevance

### Topics Analysis
- Empty `topics: []` indicates AI summarization issues
- Check your OpenAI API key configuration
- Topics are extracted using GPT models

## 🔧 Troubleshooting

### Common Issues

**1. Connection Errors**
```
❌ Failed to connect to ChromaDB
```
- Ensure you're running from the project root
- Check if `data/chromadb` directory exists
- Verify ChromaDB installation: `pip install chromadb`

**2. Import Errors**
```
ModuleNotFoundError: No module named 'src'
```
- Run tools from the `tools/` directory
- Ensure project dependencies are installed

**3. Empty Database**
```
📊 Total articles: 0
```
- No articles have been processed yet
- Run the main pipeline to add articles
- Check `demo.py` or `cli.py` for data ingestion

**4. Empty Topics**
```
Topics: []
```
- AI summarization not working
- Check OpenAI API key in environment
- Verify `OPENAI_API_KEY` is set correctly

### Environment Setup
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Set OpenAI API key (if using AI features)
set OPENAI_API_KEY=your_api_key_here

# Verify ChromaDB data exists
dir ..\data\chromadb
```

## 📝 Development Notes

### Adding New Tools
When creating new database tools:

1. Place them in this `tools/` directory
2. Update import paths:
   ```python
   sys.path.append('..')
   sys.path.append('../src')
   ```
3. Update this README with the new tool
4. Follow the existing error handling patterns

### Code Pattern
All tools follow this pattern:
```python
try:
    # Database connection
    # Data processing
    # Output formatting
except Exception as e:
    print(f"❌ Error: {e}")
```

### Import Structure
```python
import sys
sys.path.append('..')
sys.path.append('../src')

from src.storage.vector_storage import VectorStorage
from config import settings, logger
```

## 🎯 Use Cases

### For Developers
- **Debug vector search**: Test different queries and similarity thresholds
- **Validate data ingestion**: Ensure articles are properly stored
- **Monitor data quality**: Check topics extraction and summaries
- **Backup management**: Export data for backups or migration

### For Data Scientists
- **Analyze embedding quality**: Check similarity scores and clustering
- **Topic modeling validation**: Verify extracted topics make sense
- **Data exploration**: Understand content distribution and patterns
- **Performance testing**: Measure search response times

### For DevOps
- **Health monitoring**: Quick database status checks
- **Automated testing**: Validate database state in CI/CD
- **Backup procedures**: Regular data exports
- **Migration support**: Data inspection during moves

## 🔗 Integration

These tools integrate with the main project components:

- **`src/storage/vector_storage.py`**: Uses VectorStorage class
- **`src/search/semantic_searcher.py`**: Tests search functionality
- **`config/settings.py`**: Reads database configuration
- **Main pipeline**: Validates processed data

Run these tools alongside your main application to monitor and debug the vector database effectively.
