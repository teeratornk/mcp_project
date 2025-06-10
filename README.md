# MCP-Powered Research Assistant 🧠📚

This project showcases how to build a terminal-based research assistant using Large Language Models (LLMs), Azure OpenAI, and the arXiv API. It includes two progressively enhanced implementations:

- **Baseline version** using traditional API calls
- **Advanced version** using the Model Context Protocol (MCP) via FastMCP

---

## 📖 Related Articles

🧰 **[Building a Terminal-Based Research Assistant Using Azure OpenAI and arXiv](https://medium.com/@tkadeethum/building-a-terminal-based-research-assistant-using-azure-openai-and-arxiv-7738a2a215e4)**  
Introduces a basic, step-by-step implementation using arXiv and Azure OpenAI APIs without MCP. Ideal for beginners.

🛠 **[From API Calls to MCP: Developing an LLM-Powered Research Assistant](https://medium.com/@tkadeethum/from-api-calls-to-mcp-developing-an-llm-powered-research-assistant-5bc806585ab7)**  
Extends the previous implementation by integrating FastMCP for a structured and agent-ready architecture. This article explains the benefits of MCP and showcases tool chaining.

---

## 📁 Project Structure

```text
mcp_project/
├── research_server.py       # MCP server with arXiv + Azure OpenAI tools/resources
├── mcp_chatbot.py           # Client interface to interact with the MCP server
├── run_mcp_chatbot.sh       # Shell script to set up and run the chatbot
├── .env                     # Environment variables (not committed)
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Dependency configuration for uv
├── papers/                  # Local paper metadata cache
└── README.md                # This file
```

---


## 📝 About `research_server.py`

`research_server.py` is the core backend that exposes research tools and resources via the Model Context Protocol (MCP) using FastMCP. It enables structured, agent-compatible access to academic research workflows.

### 🔧 Tools

Tools are functions decorated with `@mcp.tool()` that can be called by LLM agents or clients. Each tool encapsulates a specific research action:

- **search_papers(topic, max_results):**  
  Searches arXiv for papers on a topic, saves metadata locally, and returns a list of paper IDs.

- **extract_info(paper_id):**  
  Retrieves metadata for a specific paper from local storage.

- **summarize_paper(text):**  
  Uses Azure OpenAI to summarize a research paper or text in plain English.

- **get_full_text(paper_id):**  
  Downloads and extracts the full text from a paper’s PDF using PyMuPDF.

- **list_all_papers():**  
  Lists all downloaded paper IDs, grouped by topic.

### 📦 Resources

Resources, decorated with `@mcp.resource()`, provide structured, navigable data for agents:

- **papers://folders**  
  Lists all available topic folders in the local `papers/` directory, allowing agents to browse topics.

- **papers://{topic}**  
  Shows a concise, human-readable summary of all papers under a specific topic, including titles, authors, publication dates, and short summaries.

### 💡 Prompts

Prompts, decorated with `@mcp.prompt()`, generate structured instructions for LLM agents:

- **generate_search_prompt(topic, num_papers):**  
  Produces a multi-step prompt guiding the agent to:
  1. Search for papers on a topic.
  2. Extract and summarize key details from each paper.
  3. Synthesize an overview of the research landscape, including trends, gaps, and notable works.
  4. Format the output with clear headings and bullet points for readability.

### 🛠️ How it works

- The server registers all tools, resources, and prompts with FastMCP.
- Clients (like `mcp_chatbot.py`) interact with the server using MCP, invoking tools and navigating resources as needed.
- The design supports tool chaining and structured workflows, making it easy to extend or integrate with other agent frameworks.

---
## 📝 About `mcp_chatbot.py`

`mcp_chatbot.py` is an interactive terminal-based client for the MCP-powered research assistant. It connects to the MCP server (`research_server.py`) and allows users to discover, invoke, and chain research tools, browse resources, and use structured prompts—all from the command line.

---

## 🚀 Features

- **Interactive Chat Loop:**  
  Type natural language queries or special commands to interact with the assistant.

- **Tool Discovery & Invocation:**  
  List all available tools and invoke them directly or as part of a workflow.

- **Resource Browsing:**  
  List and view structured resources (e.g., available topics, paper summaries) exposed by the server.

- **Prompt Generation & Execution:**  
  List available prompts, generate structured research instructions, and execute them with arguments.

- **Tool Chaining:**  
  Automatically chains tools (e.g., search → extract → summarize) for streamlined research tasks.

- **Azure OpenAI Integration:**  
  Uses Azure OpenAI for language understanding and summarization.

---

## 🚀 How to Run (MCP Version)

1. Clone the repository:

   ```bash
   git clone https://github.com/teeratornk/mcp_project.git
   cd mcp_project

2. Create a .env file with your Azure OpenAI credentials:

   AZURE_OPENAI_ENDPOINT=your-endpoint-url
   AZURE_OPENAI_MODEL=your-deployment-name
   AZURE_OPENAI_API_KEY=your-api-key
   
4. Run the chatbot using the provided script:

   bash run_mcp_chatbot.sh

This will:

- Initialize the project with uv
- Set up a virtual environment
- Install required packages if missing
- Launch the chatbot client

📌 Highlights
- Modular Design with FastMCP tools
- Tool Chaining: search_papers → extract_info → summarize_paper
- Local Caching of metadata for efficient repeat queries
- Declarative Tool Use compatible with structured agent frameworks

---

## 💬 Interactive Commands

Within the chatbot, you can use:

- `list tools` – Show all available tools
- `list resources` – Show all available resources
- `list prompts` – Show all available prompts
- `papers://<topic>` – Access a specific topic resource
- `run prompt <prompt_name> [arg1=val1 arg2=val2 ...]` – Generate and execute a prompt with arguments
- `<any query>` – Let the assistant process your question using LLM and available tools
- `help` – Show available commands
- `quit` – Exit the chatbot

---

📬 Contact
Developed by Teeratorn Kadeethum (Meen)

Feel free to open an issue or reach out if you'd like to collaborate!
