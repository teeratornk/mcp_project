#!/bin/bash

# Set script to exit on error
set -e

# Step 1: Navigate to the mcp_project directory
cd "$(dirname "$0")"

# Step 2: Initialize the uv project if not already initialized
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

# Step 4: Install required dependencies (only installs if missing)
required_packages=("fastmcp" "mcp" "python-dotenv" "nest_asyncio" "openai")

for pkg in "${required_packages[@]}"; do
  if ! uv pip freeze | grep -q "$pkg"; then
    echo "Installing $pkg..."
    uv add "$pkg"
  fi
done

# Step 5: Run the MCP chatbot client
echo "Running MCP chatbot client..."
uv run mcp_chatbot.py
