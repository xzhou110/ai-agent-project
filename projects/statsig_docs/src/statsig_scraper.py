import os
import time
import random
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import markdown
import atexit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class StatsigScraper:
    def __init__(self, base_url: str = "https://docs.statsig.com"):
        self.base_url = base_url
        self.visited_urls = set()
        self.docs_data = {}
        self.driver = None
        self.setup_driver()
        # Register cleanup handler
        atexit.register(self.close)
        
    def setup_driver(self):
        """Setup undetected-chromedriver."""
        try:
            logging.info("Setting up undetected-chromedriver...")
            options = uc.ChromeOptions()
            options.add_argument("--headless=new")  # Updated headless mode
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--remote-debugging-port=9222")  # Added for better stability
            
            self.driver = uc.Chrome(
                options=options,
                driver_executable_path=None,  # Will be downloaded automatically
                browser_executable_path=None,  # Will use system Chrome
            )
            logging.info("Chrome WebDriver setup successful")
        except Exception as e:
            logging.error(f"Failed to setup Chrome WebDriver: {str(e)}")
            raise
        
    def random_delay(self, min_seconds: float = 2.0, max_seconds: float = 5.0):
        """Add random delay between requests to avoid rate limiting."""
        delay = random.uniform(min_seconds, max_seconds)
        logging.debug(f"Waiting for {delay:.2f} seconds")
        time.sleep(delay)
        
    def get_page_content(self, url: str) -> Optional[str]:
        """Get page content with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logging.info(f"Fetching page: {url} (Attempt {attempt + 1}/{max_retries})")
                self.driver.get(url)
                # Wait for the main content to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "main"))
                )
                self.random_delay()
                return self.driver.page_source
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                self.random_delay(5.0, 10.0)  # Longer delay on failure
                
    def extract_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract all relevant links from the page."""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') and not href.startswith('//'):
                full_url = f"{self.base_url}{href}"
                if full_url not in self.visited_urls:
                    links.append(full_url)
        logging.info(f"Found {len(links)} new links to process")
        return links
        
    def extract_content(self, soup: BeautifulSoup) -> Dict:
        """Extract the main content from the page."""
        main_content = soup.find('main')
        if not main_content:
            logging.warning("No main content found on page")
            return {}
            
        # Extract title
        title = soup.find('h1')
        title_text = title.get_text().strip() if title else "Untitled"
        logging.info(f"Extracted title: {title_text}")
        
        # Extract content
        content = main_content.get_text(separator='\n', strip=True)
        
        return {
            'title': title_text,
            'content': content,
            'html': str(main_content)
        }
        
    def scrape_page(self, url: str):
        """Scrape a single page and its content."""
        if url in self.visited_urls:
            logging.debug(f"Skipping already visited URL: {url}")
            return
            
        logging.info(f"Scraping: {url}")
        self.visited_urls.add(url)
        
        page_content = self.get_page_content(url)
        if not page_content:
            logging.error(f"Failed to get content for {url}")
            return
            
        soup = BeautifulSoup(page_content, 'html.parser')
        content_data = self.extract_content(soup)
        
        if content_data:
            self.docs_data[url] = content_data
            logging.info(f"Successfully scraped content from {url}")
            
        # Get and process subpages
        links = self.extract_links(soup)
        for link in links:
            self.scrape_page(link)
            
    def create_consolidated_docs(self, output_dir: str = "docs") -> str:
        """Create a consolidated documentation file with proper structure and navigation."""
        # Group pages by their top-level section
        sections = {}
        for url, data in self.docs_data.items():
            # Extract section from URL (e.g., /guides/feature-flags -> guides)
            path_parts = url.replace(self.base_url, "").strip("/").split("/")
            section = path_parts[0] if path_parts else "main"
            
            if section not in sections:
                sections[section] = []
            sections[section].append((url, data))
        
        # Create the consolidated content
        consolidated = []
        
        # Add title and introduction
        consolidated.append("# Statsig Documentation\n")
        consolidated.append("This is a consolidated version of the Statsig documentation.\n")
        
        # Add table of contents
        consolidated.append("## Table of Contents\n")
        for section in sorted(sections.keys()):
            consolidated.append(f"- [{section.title()}](#{section.lower()})")
        consolidated.append("\n---\n")
        
        # Add content for each section
        for section, pages in sorted(sections.items()):
            consolidated.append(f"## {section.title()}\n")
            
            # Add section table of contents
            consolidated.append("### Pages in this section:\n")
            for url, data in sorted(pages, key=lambda x: x[1]['title']):
                page_title = data['title']
                page_id = url.replace(self.base_url, "").strip("/").replace("/", "-")
                consolidated.append(f"- [{page_title}](#{page_id})")
            consolidated.append("\n")
            
            # Add content for each page
            for url, data in sorted(pages, key=lambda x: x[1]['title']):
                page_title = data['title']
                page_id = url.replace(self.base_url, "").strip("/").replace("/", "-")
                
                consolidated.append(f"### {page_title} {{#{page_id}}}\n")
                consolidated.append(f"*Source: [{url}]({url})*\n")
                consolidated.append(data['content'])
                consolidated.append("\n---\n")
        
        return "\n".join(consolidated)
            
    def save_docs(self, output_dir: str = "docs"):
        """Save the scraped documentation to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving documentation to {output_path}")
        
        # Save as JSON (keep this for programmatic access)
        json_path = output_path / "statsig_docs.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.docs_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSON documentation to {json_path}")
        
        # Save consolidated markdown
        consolidated_content = self.create_consolidated_docs(output_dir)
        consolidated_path = output_path / "statsig_docs_consolidated.md"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            f.write(consolidated_content)
        logging.info(f"Saved consolidated documentation to {consolidated_path}")
        
        # Also save individual markdown files for reference
        for url, data in self.docs_data.items():
            # Create a filename from the URL
            filename = url.replace(self.base_url, "").replace("/", "_").strip("_")
            if not filename:
                filename = "index"
            filename = f"{filename}.md"
            
            # Convert content to markdown
            md_content = f"# {data['title']}\n\n{data['content']}"
            
            md_path = output_path / filename
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logging.info(f"Saved individual markdown documentation to {md_path}")
                
    def close(self):
        """Clean up resources gracefully."""
        if hasattr(self, 'driver') and self.driver is not None:
            try:
                logging.info("Closing Chrome WebDriver")
                # First try to close all windows
                try:
                    self.driver.quit()
                except Exception as e:
                    logging.warning(f"Error during driver.quit(): {str(e)}")
                
                # Then try to close the service
                try:
                    if hasattr(self.driver, 'service') and self.driver.service:
                        self.driver.service.stop()
                except Exception as e:
                    logging.warning(f"Error during service.stop(): {str(e)}")
                
                self.driver = None
            except Exception as e:
                logging.error(f"Error during WebDriver cleanup: {str(e)}")
            finally:
                # Ensure driver is set to None even if cleanup fails
                self.driver = None

def main():
    logging.info("Starting Statsig documentation scraper")
    scraper = None
    try:
        scraper = StatsigScraper()
        scraper.scrape_page(scraper.base_url)
        scraper.save_docs()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if scraper:
            try:
                scraper.close()
            except Exception as e:
                logging.error(f"Error during final cleanup: {str(e)}")
    logging.info("Scraping completed successfully")
        
if __name__ == "__main__":
    main() 