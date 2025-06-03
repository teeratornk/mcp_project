# MCP-Powered Research Assistant 🧠📚

This project showcases how to build a terminal-based research assistant using Large Language Models (LLMs), Azure OpenAI, and the arXiv API. It includes two progressively enhanced implementations:

- **Baseline version** using traditional API calls
- **Advanced version** using the [Model Context Protocal (MCP)](https://mcp.readthedocs.io/en/latest/) via [FastMCP](https://github.com/pacman82/fastmcp)

---

## 📖 Related Articles

🧰 **[Building a Terminal-Based Research Assistant Using Azure OpenAI and arXiv](https://medium.com/@tkadeethum/building-a-terminal-based-research-assistant-using-azure-openai-and-arxiv-7738a2a215e4)**  
Introduces a basic, step-by-step implementation using arXiv and Azure OpenAI APIs without MCP. Ideal for beginners.

🛠 **[From API Calls to MCP: Developing an LLM-Powered Research Assistant](https://medium.com/@tkadeethum/from-api-calls-to-mcp-developing-an-llm-powered-research-assistant-5bc806585ab7)**  
Extends the previous implementation by integrating [FastMCP](https://github.com/pacman82/fastmcp) for a structured and agent-ready architecture. This article explains the benefits of MCP and showcases tool chaining.

---

## 📁 Project Structure

```text
mcp_project/
├── research_server.py       # MCP server with arXiv + Azure OpenAI tools
├── mcp_chatbot.py           # Client interface to interact with the MCP server
├── run_mcp_chatbot.sh       # Shell script to set up and run the chatbot
├── .env                     # Environment variables (not committed)
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Dependency configuration for uv
├── papers/                  # Local paper metadata cache
└── README.md                # This file
```

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

📬 Contact
Developed by Teeratorn Kadeethum (Meen)

Feel free to open an issue or reach out if you'd like to collaborate!
