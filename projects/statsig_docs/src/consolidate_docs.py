import os
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def consolidate_docs(docs_dir: str = "docs"):
    """Consolidate existing documentation files into a single markdown file."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logging.error(f"Docs directory {docs_dir} does not exist!")
        return

    # First try to load the JSON file if it exists
    json_path = docs_path / "statsig_docs.json"
    if json_path.exists():
        logging.info("Loading documentation from JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
    else:
        logging.info("JSON file not found, loading from individual markdown files...")
        docs_data = {}
        for md_file in docs_path.glob("*.md"):
            if md_file.name == "statsig_docs_consolidated.md":
                continue
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract title from first line
                title = content.split('\n')[0].replace('# ', '')
                docs_data[str(md_file)] = {
                    'title': title,
                    'content': content,
                    'url': f"https://docs.statsig.com/{md_file.stem.replace('_', '/')}"
                }

    # Group pages by their top-level section
    sections = {}
    for url, data in docs_data.items():
        # Extract section from URL
        if isinstance(url, str) and url.startswith('http'):
            path_parts = url.replace('https://docs.statsig.com/', '').strip('/').split('/')
        else:
            path_parts = str(url).replace('_', '/').split('/')
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
            page_id = str(url).replace('https://docs.statsig.com/', '').replace('/', '-').strip('-')
            consolidated.append(f"- [{page_title}](#{page_id})")
        consolidated.append("\n")
        
        # Add content for each page
        for url, data in sorted(pages, key=lambda x: x[1]['title']):
            page_title = data['title']
            page_id = str(url).replace('https://docs.statsig.com/', '').replace('/', '-').strip('-')
            
            consolidated.append(f"### {page_title} {{#{page_id}}}\n")
            if isinstance(url, str) and url.startswith('http'):
                consolidated.append(f"*Source: [{url}]({url})*\n")
            consolidated.append(data['content'])
            consolidated.append("\n---\n")
    
    # Save the consolidated file
    output_path = docs_path / "statsig_docs_consolidated.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(consolidated))
    logging.info(f"Saved consolidated documentation to {output_path}")

if __name__ == "__main__":
    consolidate_docs() 