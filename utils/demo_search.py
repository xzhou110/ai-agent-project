#!/usr/bin/env python3
"""
Demo script to test the search engine functionality with a real query.
"""

import argparse
import json
from search_engine import search_internet

def main():
    parser = argparse.ArgumentParser(description='Search the internet using the search engine utility')
    parser.add_argument('query', help='The search query')
    parser.add_argument('--max-results', type=int, default=5, help='Maximum number of results to return')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retry attempts')
    args = parser.parse_args()

    print(f"Searching for: {args.query}")
    results = search_internet(args.query, max_results=args.max_results, max_retries=args.max_retries)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet']}")
    else:
        print("\nNo relevant results found for your query. Please try:")
        print(" - Different search terms")
        print(" - More general keywords")
        print(" - Check spelling of specialized terms")
        
    # Also display the raw JSON results
    print("\nRaw JSON results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 