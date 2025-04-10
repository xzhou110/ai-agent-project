import requests
import time
from typing import List, Dict, Any, Optional
import logging
import urllib.parse
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_internet(query: str, max_results: int = 10, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Search the internet using DuckDuckGo and return results with URLs and text snippets.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        List[Dict[str, Any]]: List of search results, each containing 'title', 'url', and 'snippet'
    """
    results = []
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Searching for: {query} (Attempt {retry_count + 1}/{max_retries})")
            
            # Format the query for a DuckDuckGo HTML search
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results[:max_results]:
                # Extract title, URL, and snippet
                title_element = result.find('a', class_='result__a')
                if not title_element:
                    continue
                    
                title = title_element.get_text().strip()
                url = title_element.get('href', '')
                
                # Extract actual URL from DuckDuckGo redirect URL
                if url.startswith('/'):
                    parsed_url = urllib.parse.urlparse(url)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    if 'uddg' in query_params:
                        url = query_params['uddg'][0]
                
                snippet_element = result.find('a', class_='result__snippet')
                snippet = snippet_element.get_text().strip() if snippet_element else ""
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
            
            # If we got any results, return them
            if results:
                logger.info(f"Found {len(results)} results for query: {query}")
                return results
            else:
                logger.warning(f"No results found for query: {query}")
                if retry_count + 1 < max_retries:
                    # Try again if we have retries left
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying with different approach in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return []  # Return empty results after all retries
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                # Exponential backoff: 2^retry_count seconds
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached. Returning empty results.")
                return []
    
    return results


def search_with_fallback(query: str, max_results: int = 10, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Search with the primary method first, and if it fails or returns no results, try a fallback method.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        List[Dict[str, Any]]: List of search results, or empty list if no relevant results found
    """
    # Try primary search
    results = search_internet(query, max_results, max_retries)
    
    # If we got results, return them
    if results:
        return results
        
    # Fallback: try with a simplified query if it contains multiple words
    if ' ' in query:
        simplified_query = ' '.join(query.split()[:2])  # Take only first two words
        if simplified_query != query:  # Only try if simplified query is different
            logger.warning(f"Primary search returned no results. Trying with simplified query: {simplified_query}")
            return search_internet(simplified_query, max_results, max_retries)
    
    # Second fallback: try with "what is" prefix for informational queries
    if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
        modified_query = f"what is {query}"
        logger.warning(f"Simplified search returned no results. Trying with modified query: {modified_query}")
        return search_internet(modified_query, max_results, max_retries)
    
    logger.warning(f"No relevant results found for '{query}' after trying fallbacks.")
    return []  # Return empty list if no relevant results found 