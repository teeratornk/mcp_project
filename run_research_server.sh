#!/bin/bash

# Set script to exit on error
set -e

# Step 1: Navigate to project directory
cd "$(dirname "$0")"

# Step 2: Initialize project (only if pyproject.toml doesn't exist)
if [ ! -f "pyproject.toml" ]; then
  echo "Initializing uv project..."
  uv init
fi

# Step 3: Create and activate virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv
fi

source .venv/bin/activate

# Step 4: Install dependencies (only if not already installed)
if ! uv pip freeze | grep -q fastmcp; then
  echo "Installing dependencies..."
  uv add mcp arxiv
fi

# Step 5: Run the server with Inspector support
echo "Launching FastMCP server with Inspector..."
npx @modelcontextprotocol/inspector uv run research_server.py
