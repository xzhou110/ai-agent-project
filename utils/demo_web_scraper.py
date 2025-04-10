#!/usr/bin/env python3
"""
Demo script to test the web scraper with various websites.
"""

import argparse
import json
import logging
from web_scraper import WebScraper, scrape_webpage, scrape_article, scrape_table, search_webpage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of test websites with different types of content
TEST_SITES = [
    {
        "name": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Web_scraping",
        "type": "article",
        "description": "Wikipedia article about web scraping"
    },
    {
        "name": "BBC News",
        "url": "https://www.bbc.com/news",
        "type": "news",
        "description": "BBC News homepage"
    },
    {
        "name": "GitHub",
        "url": "https://github.com/trending",
        "type": "table",
        "description": "GitHub trending repositories"
    },
    {
        "name": "Python.org",
        "url": "https://www.python.org/",
        "type": "general",
        "description": "Python programming language homepage"
    },
    {
        "name": "Stack Overflow",
        "url": "https://stackoverflow.com/questions/tagged/python",
        "type": "list",
        "description": "Stack Overflow Python questions"
    },
    {
        "name": "MDN Web Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/table",
        "type": "documentation",
        "description": "MDN documentation about HTML tables"
    }
]

def pretty_print_result(result, indent=2):
    """Print the result in a readable format."""
    if not result.get('success', False):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    print(f"✅ Successfully scraped content")
    
    # Print title if available
    if 'title' in result and result['title']:
        print(f"\nTitle: {result['title']}")
    
    # Print author and date for articles
    if 'author' in result and result['author']:
        print(f"Author: {result['author']}")
    if 'date' in result and result['date']:
        print(f"Date: {result['date']}")
    
    # Print summary of paragraphs
    if 'paragraphs' in result and result['paragraphs']:
        print(f"\nFound {len(result['paragraphs'])} paragraphs")
        if len(result['paragraphs']) > 0:
            print(f"\nFirst paragraph: {result['paragraphs'][0][:100]}...")
            if len(result['paragraphs']) > 1:
                print(f"Last paragraph: {result['paragraphs'][-1][:100]}...")
    
    # Print summary of main content
    if 'main_content' in result and result['main_content']:
        content_preview = result['main_content'][:200] + "..." if len(result['main_content']) > 200 else result['main_content']
        print(f"\nMain content preview: {content_preview}")
    
    # Print summary of article text
    if 'text' in result and result['text']:
        text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
        print(f"\nArticle text preview: {text_preview}")
    
    # Print tables
    if 'tables' in result and result['tables']:
        print(f"\nFound {len(result['tables'])} tables")
        for i, table in enumerate(result['tables']):
            print(f"\nTable {i+1} headers: {table['headers']}")
            print(f"Table {i+1} rows: {len(table['rows'])}")
            if table['rows']:
                print(f"First row sample: {table['rows'][0]}")
    
    # Print links count
    if 'links' in result and result['links']:
        print(f"\nFound {len(result['links'])} links")
        if len(result['links']) > 0:
            print(f"First link: {result['links'][0]['text']} -> {result['links'][0]['href']}")
    
    # Print search matches
    if 'matches' in result and result['matches']:
        print(f"\nFound matches for {len(result['matches'])} terms")
        for term_match in result['matches']:
            print(f"Term '{term_match['term']}' found in {len(term_match['occurrences'])} paragraphs")
            if term_match['occurrences']:
                print(f"Sample match: {term_match['occurrences'][0]['paragraph'][:100]}...")

def test_site(site, scrape_type=None, search_terms=None):
    """Test scraping a specific site."""
    print(f"\n{'='*80}")
    print(f"Testing: {site['name']} - {site['description']}")
    print(f"URL: {site['url']}")
    print(f"Type: {site['type']}")
    print(f"{'='*80}")
    
    scraper = WebScraper(delay_range=(2, 4), max_retries=2)
    
    # Determine the scraping method based on site type if not specified
    if scrape_type is None:
        scrape_type = site['type']
    
    result = None
    if scrape_type == 'article':
        print("\nScraping as article...")
        result = scraper.extract_article(site['url'])
    elif scrape_type == 'table':
        print("\nScraping tables...")
        result = scraper.extract_table(site['url'])
    elif search_terms:
        print(f"\nSearching for terms: {', '.join(search_terms)}...")
        result = scraper.search_and_extract(site['url'], search_terms)
    else:
        print("\nGeneral scraping...")
        result = scraper.extract_content(site['url'])
    
    pretty_print_result(result)
    return result

def main():
    parser = argparse.ArgumentParser(description='Test the web scraper with various websites')
    parser.add_argument('--site', type=int, help='Index of the site to test (0-5, default: all)')
    parser.add_argument('--url', type=str, help='Custom URL to scrape')
    parser.add_argument('--type', choices=['article', 'table', 'general', 'search'], 
                        help='Type of scraping to perform')
    parser.add_argument('--search', type=str, nargs='+', 
                        help='Terms to search for in the webpage')
    parser.add_argument('--save', type=str, help='Save the results to a JSON file')
    args = parser.parse_args()
    
    results = {}
    
    # If custom URL is provided
    if args.url:
        site = {
            "name": "Custom URL",
            "url": args.url,
            "type": args.type or "general",
            "description": "Custom URL provided via command line"
        }
        result = test_site(site, args.type, args.search)
        results["custom"] = result
    
    # Otherwise test predefined sites
    elif args.site is not None:
        if 0 <= args.site < len(TEST_SITES):
            site = TEST_SITES[args.site]
            result = test_site(site, args.type, args.search)
            results[site["name"]] = result
        else:
            print(f"Error: Site index {args.site} is out of range. Must be between 0 and {len(TEST_SITES)-1}.")
    else:
        # Test all sites
        for site in TEST_SITES:
            result = test_site(site, args.type, args.search)
            results[site["name"]] = result
    
    # Save results to file if requested
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")

if __name__ == "__main__":
    main() 