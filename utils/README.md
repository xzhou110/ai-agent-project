# AI Agent Project Utilities

Common utilities for the AI agent project.

## Available Utilities

### Search Engine (`search_engine.py`)

Performs web searches using DuckDuckGo and returns structured results.

**Features:**
- Real web search via DuckDuckGo HTML interface
- Automatic extraction of titles, URLs, and snippets
- Built-in retry mechanism with exponential backoff
- Fallback strategies for failed searches
- Proper URL extraction from redirect links

**Usage:**
```python
from utils.search_engine import search_internet

# Basic search
results = search_internet("fashion news")

# With parameters
results = search_internet(
    query="quantum computing", 
    max_results=5,
    max_retries=3
)
```

**Demo:**
```bash
# Windows
py demo_search.py "fashion news"

# With options
py demo_search.py "quantum computing" --max-results 3 --max-retries 2
```

**Dependencies:**
- requests
- beautifulsoup4 