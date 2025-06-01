import os
import json
import arxiv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import AzureOpenAI

# === Load environment ===
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")

# === OpenAI client ===
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# === Global paper storage ===
PAPER_DIR = "papers"
Path(PAPER_DIR).mkdir(exist_ok=True)

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search arXiv for papers matching a topic and save the info.
    """
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

def extract_info(paper_id: str) -> str:
    """
    Extract paper info from saved results.
    """
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
                    print(f"Error reading {file_path}: {e}")
    return f"No info found for paper ID: {paper_id}"

# === Tools for function calling ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search arXiv for papers about a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to search for"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Get stored info for a paper by its arXiv ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "The arXiv paper ID"}
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_paper",
            "description": "Summarize a detailed research paper using LLM",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Full paper text or long abstract"}
                },
                "required": ["text"]
            }
        }
    }
]


def execute_tool(tool_name, args):
    if tool_name == "search_papers":
        result = search_papers(args["topic"], args.get("max_results", 5))
        return json.dumps({"paper_ids": result})
    elif tool_name == "extract_info":
        return extract_info(args["paper_id"])
    elif tool_name == "summarize_paper":
        # Use the assistant to summarize
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Summarize this academic paper in plain English."},
                {"role": "user", "content": args["text"]}
            ],
            max_completion_tokens=500
        )
        return response.choices[0].message.content.strip()
    return f"Tool {tool_name} not implemented."


def process_query(query: str):
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=1024
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"Calling tool: {tool_name} with args: {tool_args}")
            tool_result = execute_tool(tool_name, tool_args)

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

            # === Tool Chaining: Automatically call extract_info on each paper ID ===
            if tool_name == "search_papers":
                result_data = json.loads(tool_result)
                paper_ids = result_data.get("paper_ids", [])[:3]  # Limit to 3 summaries for readability

                summary_block = ""
                for pid in paper_ids:
                    paper_json = extract_info(pid)
                    info = json.loads(paper_json)
                    summary_block += f"\n📝 **{info['title']}**\n- Authors: {', '.join(info['authors'])}\n- Published: {info['published']}\n- PDF: {info['pdf_url']}\n- Summary: {info['summary'][:500]}...\n\n"

                print("Top papers on LLM:")
                print(summary_block)
                return  # Skip second chat call, we printed custom output

        # === fallback second model call ===
        followup = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=1024
        )
        print(followup.choices[0].message.content)
    else:
        print(msg.content)


def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            process_query(query)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
