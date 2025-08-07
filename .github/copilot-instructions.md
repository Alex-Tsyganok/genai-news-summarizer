# AI News Summarizer - Copilot Instructions

## Project Overview
This is a **GenAI-powered news processing pipeline** with multi-stage architecture: extraction → AI summarization → vector storage → semantic search. The system demonstrates sophisticated LLM integration patterns and modular design.

## Architecture & Data Flow
```
URLs → NewsExtractor → AIsummarizer → VectorStorage → SemanticSearcher
        ↓              ↓              ↓             ↓
    BeautifulSoup   OpenAI GPT    ChromaDB      Vector Search
    fallback        + LangChain   embeddings    + filtering
```

**Key principle**: Every component has fallback mechanisms and graceful error handling.

## Core Components

### `src/pipeline.py` - Central Orchestrator
- **NewsPipeline** class coordinates all operations
- Entry point: `pipeline = NewsPipeline()` → auto-initializes all components
- Multi-step processing: extraction → summarization → storage with detailed error tracking
- Pattern: Always return structured results with `success`, `errors`, `processing_time`

### `src/models.py` - Data Contracts
- **Article**: Core data model with `to_dict()`/`from_dict()` serialization
- **ProcessingResult**: Standardizes success/failure responses
- **SearchResult**: Wraps articles with similarity scores and ranks
- All models use dataclasses with post-init validation

### Component Pattern: Interface + Implementation
Each module follows: `Interface → Primary Implementation → Fallback → Error Handling`

**Extractors** (`src/extractors/`):
- Primary: `newspaper3k` for robust parsing
- Fallback: `BeautifulSoup` for challenging sites
- Always validate content length before returning

**Summarizers** (`src/summarizers/`):
- Structured OpenAI prompts with JSON schema enforcement
- Token-aware content truncation (8000 char limit)
- Fallback: First 3 sentences if AI fails
- Pattern: `_prepare_content()` → `_generate_summary_and_topics()` → `_validate_*()` methods

**Storage** (`src/storage/`):
- ChromaDB with persistent collections
- Document prep: `title | summary | topics | body_excerpt`
- Metadata includes original article data + extraction method
- Batch operations for efficiency

**Search** (`src/search/`):
- Vector similarity + semantic filters
- Query enhancement for context
- Multiple search modes: text, topics, similarity-based

## Configuration System

### `config/settings.py` - Environment-Based Config
```python
# All settings load from environment with sensible defaults
OPENAI_MODEL = "gpt-3.5-turbo"  # Configurable model
MAX_SUMMARY_LENGTH = 200        # Token management
SIMILARITY_THRESHOLD = 0.7      # Search quality control
```

**Critical**: API keys validation happens at pipeline init, not import time.

## Development Workflows

### Testing Strategy
```bash
python tests/test_pipeline.py  # Component tests with API key checks
python demo.py                 # Full pipeline demonstration
python setup.py               # Dependency + environment setup
```

**Pattern**: Tests skip AI components if `OPENAI_API_KEY` missing, focus on data processing logic.

### Multi-Interface Pattern
- **Web UI**: `streamlit run src/ui/app.py` (session state management)
- **CLI**: `python cli.py add|search|stats|export` (command routing)
- **API**: Direct pipeline imports for programmatic use

## Critical Patterns

### Error Handling Strategy
1. **Graceful Degradation**: AI failures → fallback summaries, not system crashes
2. **Structured Errors**: Always include `step`, `url`, `error` in failure tracking
3. **Component Health**: `pipeline.health_check()` validates all dependencies

### AI Integration Best Practices
```python
# Structured prompts with explicit output format
system_prompt = """Extract summary and topics as JSON:
{
  "summary": "2-3 sentences capturing key points",
  "topics": ["topic1", "topic2", "topic3"]
}"""

# Token management with content truncation
if len(content) > 8000:
    content = content[:8000] + "..."
```

### Vector Database Patterns
- **ID Generation**: URL-based hashing for deduplication
- **Document Text**: Multi-field concatenation with separators
- **Metadata**: Flatten nested objects, JSON-encode lists
- **Search Results**: Always convert distances to similarity scores (1 - distance)

## File Organization Logic

### Source Structure
- `src/` - Core implementation (no configs/utils mixed in)
- `config/` - Centralized settings + logging
- `examples/` - Working demonstrations, not documentation
- `tests/` - Component-focused with real integration tests

### Interface Files
- `cli.py` - Full CLI with subcommands
- `demo.py` - Interactive pipeline demonstration
- `setup.py` - Automated environment setup

## Integration Points

### External Dependencies
- **OpenAI API**: GPT models + embeddings (configurable models)
- **ChromaDB**: Persistent vector storage (auto-creates collections)
- **Streamlit**: Web interface with session state management

### Cross-Component Communication
- **Pipeline → Components**: Direct instantiation with dependency injection
- **UI → Pipeline**: Session state caching with health checks
- **CLI → Pipeline**: Command-line argument routing

## Development Guidelines

### Commit Message Convention
Follow this structured format for consistent commit history:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New features (extractors, summarizers, search, ui)
- `fix`: Bug fixes and error handling improvements
- `ai`: AI/LLM integration changes (prompts, models, fallbacks)
- `data`: Vector storage, ChromaDB, or data model changes
- `config`: Settings, environment, or configuration updates
- `test`: Test additions or modifications
- `docs`: Documentation updates
- `refactor`: Code restructuring without functionality changes
- `perf`: Performance improvements
- `chore`: Dependencies, setup, or maintenance tasks

**Scopes:**
- `pipeline`: Core NewsPipeline orchestration
- `extractor`: News extraction (newspaper3k, BeautifulSoup)
- `summarizer`: AI summarization and topic identification
- `storage`: Vector database operations (ChromaDB)
- `search`: Semantic search functionality
- `ui`: Streamlit web interface
- `cli`: Command-line interface
- `config`: Configuration and settings
- `models`: Data models and structures

**Examples:**
```bash
feat(extractor): add retry mechanism for failed extractions
fix(ai): improve fallback summary when GPT fails
ai(summarizer): optimize prompts for better topic extraction
data(storage): implement batch article storage
config: add similarity threshold environment variable
test(pipeline): add comprehensive integration tests
```

**Body Guidelines:**
- Explain the "why" behind changes
- Reference issue numbers: `Fixes #123`
- Mention breaking changes: `BREAKING CHANGE: API updated`

### When Modifying Components
1. **Extractors**: Test both newspaper3k and BeautifulSoup paths
2. **Summarizers**: Validate JSON parsing + fallback behavior
3. **Storage**: Check batch operations and ID consistency
4. **Search**: Verify similarity score calculations

### Adding New Features
- Follow the fallback pattern: primary method → backup → graceful failure
- Add configuration to `settings.py` with environment variable support
- Include in `health_check()` if external dependency
- Update all three interfaces (Web, CLI, API) consistently

### Common Pitfalls
- **API Keys**: Validate at runtime, not import time
- **Vector Storage**: URLs as IDs can collide - use hash functions
- **Streamlit**: Session state clearing requires `st.experimental_rerun()`
- **ChromaDB**: Collections must exist before operations (auto-create pattern)

This system prioritizes **robustness over perfection** - every operation can gracefully handle failures while providing detailed diagnostics.
