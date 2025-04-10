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

### Web Scraper (`web_scraper.py`)

Extracts content from web pages with anti-blocking measures and intelligent content parsing.

**Features:**
- Random delays between requests to avoid rate limiting
- User-agent rotation to avoid detection
- Smart content extraction for various page types
- Retry logic with exponential backoff
- Support for extracting articles, tables, and general content
- Term search with context extraction

**Usage:**
```python
from utils.web_scraper import WebScraper, scrape_webpage, scrape_article

# General webpage scraping
scraper = WebScraper(delay_range=(1, 3), max_retries=3)
content = scraper.extract_content("https://example.com")

# Article extraction
article = scrape_article("https://example.com/blog/article")

# Table extraction 
from utils.web_scraper import scrape_table
table_data = scrape_table("https://example.com/data-table")

# Search for specific terms
from utils.web_scraper import search_webpage
results = search_webpage("https://example.com", ["keyword1", "keyword2"])
```

**Demo:**
```bash
# Test with predefined sites
py demo_web_scraper.py

# Test specific site (0-5)
py demo_web_scraper.py --site 0

# Test custom URL
py demo_web_scraper.py --url https://example.com --type article

# Search for terms
py demo_web_scraper.py --url https://example.com --search keyword1 keyword2

# Save results to file
py demo_web_scraper.py --site 0 --save results.json
```

**Dependencies:**
- requests
- beautifulsoup4 