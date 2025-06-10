
import os
import json
import arxiv
import fitz  # PyMuPDF
import tempfile
import os
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

@mcp.tool()
def get_full_text(paper_id: str) -> str:
    """Download and extract full text from an arXiv paper using its ID."""
    try:
        paper = next(arxiv.Search(id_list=[paper_id]).results())
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, f"{paper_id}.pdf")
            paper.download_pdf(filename=pdf_path)

            doc = fitz.open(pdf_path)
            full_text = "\n".join([page.get_text() for page in doc])
            return full_text[:20000]  # Truncate if needed (LLM context length)
    except Exception as e:
        return f"Failed to fetch or extract text for {paper_id}: {e}"

@mcp.tool()
def list_all_papers() -> dict:
    """List all downloaded paper IDs grouped by topic."""
    topic_papers = {}
    for subdir in Path(PAPER_DIR).iterdir():
        if subdir.is_dir():
            topic = subdir.name.replace("_", " ")
            file_path = subdir / "papers_info.json"
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        info = json.load(f)
                        topic_papers[topic] = list(info.keys())
                except Exception:
                    topic_papers[topic] = ["Error reading metadata"]
    return topic_papers

@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.
    
    This resource provides a simple list of all available topic folders.
    """
    folders = []

    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR, topic_dir)
            if os.path.isdir(topic_path):
                papers_file = os.path.join(topic_path, "papers_info.json")
                if os.path.exists(papers_file):
                    folders.append(topic_dir)

    content = "# 📁 Available Topics\n\n"
    if folders:
        for folder in folders:
            display_name = folder.replace("_", " ").title()
            content += f"- [{display_name}](papers://{folder})\n"
    else:
        content += "_No topics found. Use the `search_papers` tool to create some._\n"

    return content

@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Concise view of papers under a specific topic.
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")

    if not os.path.exists(papers_file):
        return f"# ❌ No papers found for topic: `{topic}`\nTry using `search_papers('{topic}')` to fetch some."

    try:
        with open(papers_file, "r") as f:
            papers_data = json.load(f)

        content = f"# 📚 Topic: {topic.replace('_', ' ').title()}\n"
        content += f"Found {len(papers_data)} paper(s):\n\n"

        for i, (paper_id, paper) in enumerate(papers_data.items(), 1):
            content += f"### {i}. {paper['title']}\n"
            content += f"- 🆔 `{paper_id}` | 🗓 {paper['published']}\n"
            content += f"- 👥 {', '.join(paper['authors'])}\n"
            content += f"- 🔗 [PDF]({paper['pdf_url']})\n"
            content += f"- 📝 Summary: {paper['summary'][:200].strip()}...\n\n"

        return content

    except Exception as e:
        return f"# ⚠️ Error reading topic `{topic}`: {e}"



if __name__ == "__main__":
    mcp.run(transport='stdio')

