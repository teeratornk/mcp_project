
import os
import json
import arxiv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from fastmcp import FastMCP

# Initialize MCP
mcp = FastMCP("research")

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")

from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

PAPER_DIR = "papers"
Path(PAPER_DIR).mkdir(exist_ok=True)

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """Search arXiv for papers matching a topic and save the results locally."""
    client_arxiv = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = client_arxiv.results(search)

    topic_dir = Path(PAPER_DIR) / topic.lower().replace(" ", "_")
    topic_dir.mkdir(parents=True, exist_ok=True)
    file_path = topic_dir / "papers_info.json"

    try:
        with open(file_path, "r") as f:
            papers_info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in results:
        pid = paper.get_short_id()
        paper_ids.append(pid)
        papers_info[pid] = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date())
        }

    with open(file_path, "w") as f:
        json.dump(papers_info, f, indent=2)

    return paper_ids

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """Extract metadata for a specific arXiv paper by ID from local storage."""
    for subdir in Path(PAPER_DIR).iterdir():
        if subdir.is_dir():
            file_path = subdir / "papers_info.json"
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        info = json.load(f)
                        if paper_id in info:
                            return json.dumps(info[paper_id], indent=2)
                except Exception as e:
                    return f"Error reading {file_path}: {str(e)}"
    return f"No info found for paper ID: {paper_id}"

@mcp.tool()
def summarize_paper(text: str) -> str:
    """Summarize a detailed research paper using Azure OpenAI."""
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant. Summarize this academic paper in plain English."},
            {"role": "user", "content": text}
        ],
        max_completion_tokens=500
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    mcp.run(transport='stdio')

