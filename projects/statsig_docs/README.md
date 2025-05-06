# Statsig Documentation Scraper

This project scrapes and saves the complete documentation from [Statsig's documentation website](https://docs.statsig.com/).

## Features

- Scrapes all pages and subpages from the Statsig documentation
- Implements rate limiting and anti-blocking measures
- Saves documentation in both JSON and Markdown formats
- Handles dynamic content loading using Selenium
- Includes retry logic for failed requests

## Prerequisites

- Python 3.8 or higher
- Chrome browser installed (for Selenium WebDriver)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scraper:
```bash
python src/statsig_scraper.py
```

The script will:
1. Start scraping from the main documentation page
2. Follow all internal links recursively
3. Save the documentation in the `docs` directory:
   - `statsig_docs.json`: Complete documentation in JSON format
   - Individual `.md` files for each page

## Output Structure

The documentation is saved in two formats:

1. JSON format (`docs/statsig_docs.json`):
   - Contains all pages with their titles, content, and HTML
   - Structured by URL for easy reference

2. Markdown format (`docs/*.md`):
   - Individual markdown files for each page
   - Named based on the URL path
   - Includes title and content in markdown format

## Notes

- The scraper includes random delays between requests to avoid rate limiting
- Failed requests are retried up to 3 times with increasing delays
- The script uses a headless Chrome browser for better compatibility with dynamic content

## License

MIT License 