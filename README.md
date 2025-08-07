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
- 🤖 **AI-Powered Analysis**: OpenAI GPT models for summarization and topic identification  
- 🔍 **Advanced Semantic Search**: Vector embeddings with contextual understanding
- 🧠 **Synonym Recognition**: Handles semantically similar search terms
- 💾 **Persistent Vector Storage**: ChromaDB for scalable article management
- 🌐 **Multiple Interfaces**: Web UI, CLI, and Python API
- 📊 **Analytics Dashboard**: Trending topics and collection insights

## 🏗️ Architecture & Technology Stack

### Core Components
```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  News Extractor │    │   AI Summarizer  │    │ Vector Storage  │
│                 │    │                  │    │                 │
│ • Newspaper3k   │───▶│ • OpenAI GPT     │───▶│ • ChromaDB      │
│ • BeautifulSoup │    │ • LangChain      │    │ • Embeddings    │
│ • Fallback Logic│    │ • Topic Extract  │    │ • Persistence   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ Search Interface│    │ Semantic Searcher│◀────────────┘
│                 │    │                  │
│ • Web UI        │───▶│ • Vector Search  │
│ • CLI Tool      │    │ • Synonym Handle │
│ • Python API    │    │ • Context Match  │
└─────────────────┘    └──────────────────┘
```

### Technology Stack
- **🐍 Python 3.8+**: Core programming language (3.12.x recommended for best compatibility)
- **🔗 LangChain**: AI orchestration and prompt management
- **🤖 OpenAI API**: GPT models for summarization and embeddings
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
streamlit run src/ui/app.py
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
- **Vector DB**: Configure ChromaDB settings
- **Extraction**: Customize scraping parameters
- **Search**: Adjust similarity thresholds

## API Reference

See detailed documentation in each module:

- `src/extractors/` - Article extraction and parsing
- `src/summarizers/` - AI summarization and topic detection
- `src/storage/` - Vector database operations
- `src/search/` - Semantic search implementation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
