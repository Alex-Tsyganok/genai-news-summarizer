# AI News Summarizer and Semantic Search

## 🎯 Test Assignment: News Scraping & GenAI Integration

**Objective**: A comprehensive solution that demonstrates problem-solving skills and GenAI technology integration for news article processing, summarization, and semantic search.

### 📋 Assignment Requirements Fulfilled

✅ **News Extraction**: Advanced web scraping with multiple fallback methods  
✅ **GenAI Summarization**: OpenAI GPT integration for intelligent content analysis  
✅ **Topic Identification**: AI-powered topic extraction and categorization  
✅ **Vector Database**: ChromaDB for efficient storage and retrieval  
✅ **Semantic Search**: Context-aware search with synonym handling  
✅ **Python Implementation**: Clean, documented, production-ready code  
✅ **LangChain Integration**: Modern AI orchestration framework  

## 🚀 Key Features

- 📰 **Multi-Method News Extraction**: Newspaper3k + BeautifulSoup fallback for robust scraping
- 🎯 **AI Confidence Scoring**: GPT-powered authenticity and quality validation of news content
- 🤖 **AI-Powered Analysis**: OpenAI GPT models for summarization and topic identification  
- 🔍 **Advanced Semantic Search**: Vector embeddings with contextual understanding
- 🧠 **Synonym Recognition**: Handles semantically similar search terms
- 💾 **Persistent Vector Storage**: ChromaDB for scalable article management
- 🌐 **Multiple Interfaces**: Web UI, CLI, and Python API
- 📊 **Analytics Dashboard**: Trending topics and collection insights

## 🏗️ Architecture & Technology Stack

### Core Components
```text
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  News Extractor │    │  AI Confidence   │    │   AI Summarizer  │    │ Vector Storage  │
│                 │    │     Scorer       │    │                  │    │                 │
│ • Newspaper3k   │───▶│ • GPT Analysis   │───▶│ • OpenAI GPT     │───▶│ • ChromaDB      │
│ • BeautifulSoup │    │ • Quality Check  │    │ • LangChain      │    │ • Embeddings    │
│ • Fallback Logic│    │ • News Validate  │    │ • Topic Extract  │    │ • Persistence   │
└─────────────────┘    └──────────────────┘    └──────────────────┘    └─────────────────┘
                                                                                 │
┌─────────────────┐    ┌──────────────────┐                                     │
│ Search Interface│    │ Semantic Searcher│◀────────────────────────────────────┘
│                 │    │                  │
│ • Web UI        │───▶│ • Vector Search  │
│ • CLI Tool      │    │ • Synonym Handle │
│ • Python API    │    │ • Context Match  │
└─────────────────┘    └──────────────────┘
```

### Technology Stack

- **🐍 Python 3.8+**: Core programming language (3.12.x recommended for best compatibility)
- **🔗 LangChain**: AI orchestration and prompt management
- **🤖 OpenAI API**: GPT models for analysis, summarization, and embeddings
- **🎯 AI Models**: GPT-3.5/4 for content validation and scoring
- **💾 ChromaDB**: Vector database for semantic storage
- **🌐 Streamlit**: Interactive web interface
- **🕷️ Newspaper3k**: Primary article extraction
- **🍲 BeautifulSoup**: Fallback web scraping
- **📊 Pandas**: Data manipulation and analysis

## 🚀 Quick Start Guide

### 1. Setup & Installation

```bash
# Clone the repository
git clone <repository-url>
cd genai-news-summarizer

# Create and activate virtual environment (RECOMMENDED)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you encounter build errors with Python 3.13, try:
# python -m pip install --upgrade pip setuptools wheel
# pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Run the Application

**Web Interface (Recommended)**

```bash
streamlit run src/ui/Home.py
```

**Command Line Interface**

```bash
# Process articles
python cli.py add https://www.bbc.com/news/articles/c0k3700zljjo https://www.artificialintelligence-news.com/news/alan-turing-institute-humanities-are-key-future-of-ai

# Search articles
python cli.py search "artificial intelligence developments"

# View statistics
python cli.py stats
```

**Python API**

```python
from src.pipeline import NewsPipeline

# Initialize pipeline
pipeline = NewsPipeline()

# Process articles
urls = ["https://example.com/article1", "https://example.com/article2"]
results = pipeline.process_articles(urls)

# Semantic search
search_results = pipeline.search("machine learning breakthroughs")

# Topic-based search
topic_results = pipeline.search_by_topics(["AI", "technology"])
```

## 📋 Assignment Implementation

This project fulfills all requirements of the **News Scraping & GenAI Integration** test assignment:

### ✅ News Extraction
- **Multi-method scraping**: Newspaper3k + BeautifulSoup fallback
- **Robust content capture**: Full text, headlines, and metadata
- **Error handling**: Graceful failures and retry mechanisms

### ✅ GenAI-Driven Summarization & Topic Identification  
- **OpenAI GPT integration**: Advanced language model utilization
- **LangChain orchestration**: Professional AI workflow management
- **Intelligent analysis**: Key point extraction and topic categorization
- **Structured outputs**: Consistent JSON-formatted responses

### ✅ Semantic Search with Vector Database
- **ChromaDB storage**: Persistent vector database for articles
- **OpenAI embeddings**: High-quality text representations
- **Context understanding**: Semantic similarity beyond keywords
- **Synonym handling**: Intelligent matching of related terms

### ✅ Technical Excellence
- **Python implementation**: Clean, documented, modular code
- **Modern libraries**: LangChain, OpenAI API, ChromaDB integration
- **Multiple interfaces**: Web UI, CLI, and Python API
- **Production ready**: Comprehensive testing and error handling

## Configuration

The system supports various configurations in `config/settings.py`:

- **AI Models**: Choose between different OpenAI models
- **Embedding Models**: Supported models: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Vector DB**: Configure ChromaDB settings
- **Extraction**: Customize scraping parameters
- **Search**: Adjust similarity thresholds

## 🛠️ Database Tools

The `tools/` directory contains utilities for database inspection and debugging:

- **`tools/inspect_db.py`** - Interactive CLI for comprehensive database exploration
- **`tools/quick_db_view.py`** - Fast overview of all database records  
- **`tools/explore_chromadb.py`** - Direct ChromaDB access for troubleshooting

```bash
# Interactive database exploration
cd tools
python inspect_db.py

# Quick database overview
python quick_db_view.py

# Direct ChromaDB inspection
python explore_chromadb.py
```

See [`tools/README.md`](tools/README.md) for detailed documentation.

## 📁 Project Structure

```text
genai-news-summarizer/
├── 📄 README.md                  # Main project documentation
├── 📄 requirements.txt           # Python dependencies
├── 📄 setup.py                   # Environment setup script
├── 📄 cli.py                     # Command-line interface
├── 📄 demo.py                    # Interactive demonstration
│
├── 📁 src/                       # Core application code
│   ├── 📄 pipeline.py            # Main orchestration pipeline
│   ├── 📄 models.py              # Data models and structures
│   ├── 📁 extractors/            # News extraction modules
│   ├── 📁 summarizers/           # AI summarization modules
│   ├── 📁 storage/               # Vector database operations
│   ├── 📁 search/                # Semantic search implementation
│   └── 📁 ui/                    # Streamlit web interface
│
├── 📁 config/                    # Configuration and settings
│   ├── 📄 settings.py            # Environment variables and config
│   └── 📄 logging_config.py      # Logging configuration
│
├── 📁 tools/                     # Database inspection utilities
│   ├── 📄 README.md              # Tools documentation
│   ├── 📄 inspect_db.py          # Interactive database explorer
│   ├── 📄 quick_db_view.py       # Fast database overview
│   └── 📄 explore_chromadb.py    # Direct ChromaDB access
│
├── 📁 tests/                     # Test suite
├── 📁 examples/                  # Usage examples
├── 📁 data/                      # Database and sample data
└── 📁 docs/                      # Additional documentation
```

## API Reference

See detailed documentation in each module:

- `src/extractors/` - Article extraction and parsing
- `src/summarizers/` - AI summarization and topic detection
- `src/storage/` - Vector database operations
- `src/search/` - Semantic search implementation

## Contributing

This project is for educational purposes and is not accepting external contributions.

- You are welcome to fork the repository and experiment locally.
- Issues and pull requests may not be reviewed.

## License

MIT License - see LICENSE file for details
