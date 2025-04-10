#!/usr/bin/env python3
"""
Web Scraper Utility

A general-purpose web scraper that can extract content from webpages
with built-in measures to avoid being blocked by websites.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from typing import Dict, List, Optional, Union, Any
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

class WebScraper:
    """A class for scraping content from websites with safeguards against blocking."""
    
    def __init__(self, delay_range: tuple = (1, 3), respect_robots: bool = True, max_retries: int = 3):
        """
        Initialize the WebScraper.
        
        Args:
            delay_range (tuple): Min and max delay between requests in seconds
            respect_robots (bool): Whether to respect robots.txt rules
            max_retries (int): Maximum number of retry attempts for failed requests
        """
        self.delay_range = delay_range
        self.respect_robots = respect_robots
        self.max_retries = max_retries
        self.session = requests.Session()
        self.last_request_time = 0
        self._robots_cache = {}  # Cache for robots.txt rules
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the list of common user agents."""
        return random.choice(USER_AGENTS)
    
    def _apply_delay(self) -> None:
        """Apply a random delay between requests to avoid rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        
        # Only delay if we've made a request before and not enough time has passed
        if self.last_request_time > 0:
            min_delay, max_delay = self.delay_range
            desired_delay = random.uniform(min_delay, max_delay)
            
            if elapsed < desired_delay:
                sleep_time = desired_delay - elapsed
                logger.debug(f"Sleeping for {sleep_time:.2f} seconds to avoid rate limiting")
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt rules.
        Always returns True if respect_robots is False.
        """
        if not self.respect_robots:
            return True
            
        try:
            domain = urlparse(url).netloc
            if domain not in self._robots_cache:
                # TODO: Implement actual robots.txt parsing for production use
                # For now, just cache that we haven't checked this domain
                self._robots_cache[domain] = True
            
            return self._robots_cache[domain]
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True  # Default to allowing in case of error
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None, 
            retry_status_codes: List[int] = None) -> Optional[requests.Response]:
        """
        Make a GET request to the URL with anti-blocking measures.
        
        Args:
            url (str): The URL to request
            headers (dict, optional): Additional headers to include
            retry_status_codes (list, optional): Status codes to retry on
            
        Returns:
            Response object if successful, None otherwise
        """
        if retry_status_codes is None:
            retry_status_codes = [429, 500, 502, 503, 504]
            
        if not self._can_fetch(url):
            logger.warning(f"URL {url} is disallowed by robots.txt")
            return None
            
        # Apply delay to avoid rate limiting
        self._apply_delay()
        
        # Set up headers with a random user agent
        request_headers = {'User-Agent': self._get_random_user_agent()}
        if headers:
            request_headers.update(headers)
            
        retries = 0
        while retries <= self.max_retries:
            try:
                logger.info(f"Requesting {url} (Attempt {retries + 1}/{self.max_retries + 1})")
                response = self.session.get(url, headers=request_headers, timeout=30)
                
                if response.status_code == 200:
                    return response
                    
                if response.status_code in retry_status_codes and retries < self.max_retries:
                    wait_time = 2 ** (retries + 2)  # Exponential backoff
                    logger.warning(f"Got status code {response.status_code}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    # Rotate user agent for retry
                    request_headers['User-Agent'] = self._get_random_user_agent()
                else:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status_code}")
                    return None
                    
            except requests.RequestException as e:
                if retries < self.max_retries:
                    wait_time = 2 ** (retries + 2)
                    logger.warning(f"Request error: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    # Rotate user agent for retry
                    request_headers['User-Agent'] = self._get_random_user_agent()
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries + 1} attempts: {str(e)}")
                    return None
        
        return None
    
    def extract_content(self, url: str, selectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Extract content from a webpage based on CSS selectors.
        
        Args:
            url (str): URL to scrape
            selectors (dict): CSS selectors mapped to field names
                              e.g. {'title': 'h1.title', 'content': 'div.article-content'}
                              
        Returns:
            dict: Extracted content with field names as keys
        """
        result = {'url': url, 'success': False}
        
        if selectors is None:
            # Default selectors for common content
            selectors = {
                'title': 'title',
                'headings': 'h1, h2',
                'paragraphs': 'p',
                'links': 'a'
            }
            
        response = self.get(url)
        if not response:
            return result
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content based on selectors
            for field, selector in selectors.items():
                elements = soup.select(selector)
                if field == 'links':
                    result[field] = [{'text': el.get_text().strip(), 'href': el.get('href')} 
                                    for el in elements if el.get('href')]
                elif field in ['paragraphs', 'headings']:
                    result[field] = [el.get_text().strip() for el in elements]
                else:
                    if elements:
                        result[field] = elements[0].get_text().strip()
                    else:
                        result[field] = None
            
            # If no selectors matched content, try extracting main text content
            if not any(result.get(field) for field in selectors.keys() if field != 'links'):
                # Extract main text as a fallback
                main_content = self._extract_main_content(soup)
                if main_content:
                    result['main_content'] = main_content
            
            result['success'] = True
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            result['error'] = str(e)
            return result
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from a webpage using heuristics."""
        # Try common content container selectors
        content_selectors = [
            'article', 'main', '.content', '.post-content', '.article-content',
            '.entry-content', '#content', '[role="main"]'
        ]
        
        for selector in content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                # Extract text with paragraph breaks
                paragraphs = main_element.find_all('p')
                if paragraphs:
                    return "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        # Fallback: get all paragraphs with reasonable length
        all_paragraphs = soup.find_all('p')
        content_paragraphs = [p.get_text().strip() for p in all_paragraphs 
                             if len(p.get_text().strip()) > 40]  # Filter out short paragraphs
        
        if content_paragraphs:
            return "\n\n".join(content_paragraphs)
        
        # Last resort: get all text from body
        body = soup.find('body')
        if body:
            return body.get_text().strip()
            
        return ""
    
    def extract_article(self, url: str) -> Dict[str, Any]:
        """
        Extract article content with a focus on text content.
        
        Args:
            url (str): URL of the article to scrape
            
        Returns:
            dict: Article content with title, author, date, and text
        """
        selectors = {
            'title': 'h1, .article-title, .entry-title, .post-title',
            'author': '.author, .byline, .meta-author, [rel="author"]',
            'date': '.date, .published, time, .meta-date, [itemprop="datePublished"]',
            'content': 'article, .article-content, .entry-content, .post-content'
        }
        
        result = self.extract_content(url, selectors)
        
        # Process the content field if it exists
        if result['success'] and 'content' in result and result['content']:
            content_html = result['content']
            soup = BeautifulSoup(f"<div>{content_html}</div>", 'html.parser')
            
            # Extract paragraphs from the content
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
            if paragraphs:
                result['text'] = "\n\n".join(paragraphs)
            else:
                # If no paragraphs found, use the raw text
                result['text'] = soup.get_text().strip()
        
        return result
    
    def extract_table(self, url: str, table_selector: str = 'table') -> Dict[str, Any]:
        """
        Extract tabular data from a webpage.
        
        Args:
            url (str): URL to scrape
            table_selector (str): CSS selector for the table
            
        Returns:
            dict: Table data with headers and rows
        """
        result = {'url': url, 'success': False, 'tables': []}
        
        response = self.get(url)
        if not response:
            return result
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.select(table_selector)
            
            for idx, table in enumerate(tables):
                table_data = {'headers': [], 'rows': []}
                
                # Extract headers
                thead = table.find('thead')
                if thead:
                    headers = thead.find_all('th')
                    if not headers:
                        headers = thead.find_all('td')
                    table_data['headers'] = [h.get_text().strip() for h in headers]
                
                # If no thead, try first row as header
                if not table_data['headers']:
                    first_row = table.find('tr')
                    if first_row:
                        headers = first_row.find_all('th')
                        if not headers:
                            headers = first_row.find_all('td')
                        table_data['headers'] = [h.get_text().strip() for h in headers]
                
                # Extract rows
                rows = table.find_all('tr')
                for row in rows:
                    # Skip header row if we already extracted headers
                    if row == table.find('tr') and table_data['headers']:
                        continue
                        
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_data = [cell.get_text().strip() for cell in cells]
                        table_data['rows'].append(row_data)
                
                result['tables'].append(table_data)
            
            result['success'] = len(result['tables']) > 0
            return result
            
        except Exception as e:
            logger.error(f"Error extracting table from {url}: {str(e)}")
            result['error'] = str(e)
            return result
    
    def search_and_extract(self, url: str, search_terms: List[str], 
                         context_size: int = 2) -> Dict[str, Any]:
        """
        Search for specific terms in a webpage and extract surrounding context.
        
        Args:
            url (str): URL to scrape
            search_terms (list): List of terms to search for
            context_size (int): Number of paragraphs before/after to include as context
            
        Returns:
            dict: Matches with context
        """
        result = {'url': url, 'success': False, 'matches': []}
        
        # First extract the content
        content_result = self.extract_content(url)
        if not content_result['success']:
            return result
        
        # Compile all text content
        all_paragraphs = []
        if 'paragraphs' in content_result:
            all_paragraphs.extend(content_result['paragraphs'])
        if 'main_content' in content_result:
            all_paragraphs.extend(content_result['main_content'].split('\n\n'))
        
        # Search for terms
        for term in search_terms:
            term_matches = []
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            for i, paragraph in enumerate(all_paragraphs):
                if pattern.search(paragraph):
                    # Get context paragraphs
                    start = max(0, i - context_size)
                    end = min(len(all_paragraphs), i + context_size + 1)
                    context = all_paragraphs[start:end]
                    
                    term_matches.append({
                        'paragraph': paragraph,
                        'context': '\n\n'.join(context),
                        'position': i
                    })
            
            if term_matches:
                result['matches'].append({
                    'term': term,
                    'occurrences': term_matches
                })
        
        result['success'] = True
        return result

# Helper functions for common scraping tasks
def scrape_webpage(url: str, selectors: Optional[Dict[str, str]] = None, 
                  delay_range: tuple = (1, 3), max_retries: int = 3) -> Dict[str, Any]:
    """
    Scrape a webpage with the given selectors.
    
    Args:
        url (str): URL to scrape
        selectors (dict, optional): CSS selectors mapped to field names
        delay_range (tuple): Min and max delay between requests
        max_retries (int): Maximum number of retries
        
    Returns:
        dict: Extracted content
    """
    scraper = WebScraper(delay_range=delay_range, max_retries=max_retries)
    return scraper.extract_content(url, selectors)

def scrape_article(url: str, delay_range: tuple = (1, 3), max_retries: int = 3) -> Dict[str, Any]:
    """
    Scrape an article with focus on text content.
    
    Args:
        url (str): URL of the article
        delay_range (tuple): Min and max delay between requests
        max_retries (int): Maximum number of retries
        
    Returns:
        dict: Article content
    """
    scraper = WebScraper(delay_range=delay_range, max_retries=max_retries)
    return scraper.extract_article(url)

def scrape_table(url: str, table_selector: str = 'table', 
                delay_range: tuple = (1, 3), max_retries: int = 3) -> Dict[str, Any]:
    """
    Scrape a table from a webpage.
    
    Args:
        url (str): URL to scrape
        table_selector (str): CSS selector for the table
        delay_range (tuple): Min and max delay between requests
        max_retries (int): Maximum number of retries
        
    Returns:
        dict: Table data
    """
    scraper = WebScraper(delay_range=delay_range, max_retries=max_retries)
    return scraper.extract_table(url, table_selector)

def search_webpage(url: str, search_terms: List[str], context_size: int = 2,
                 delay_range: tuple = (1, 3), max_retries: int = 3) -> Dict[str, Any]:
    """
    Search for specific terms in a webpage and extract context.
    
    Args:
        url (str): URL to scrape
        search_terms (list): List of terms to search for
        context_size (int): Number of paragraphs before/after to include
        delay_range (tuple): Min and max delay between requests
        max_retries (int): Maximum number of retries
        
    Returns:
        dict: Matches with context
    """
    scraper = WebScraper(delay_range=delay_range, max_retries=max_retries)
    return scraper.search_and_extract(url, search_terms, context_size) 