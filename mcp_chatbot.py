from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os
import json
import argparse

nest_asyncio.apply()
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")


class MCP_ChatBot:
    def __init__(self):
        self.session: ClientSession = None
        self.available_tools: List[dict] = []
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.message_history = [{"role": "system", "content": "You are a helpful assistant."}]

    async def handle_search_and_summarize(self, paper_ids: List[str]):
        for pid in paper_ids[:3]:
            print(f"\n🔎 Extracting info for {pid}")
            try:
                info_result = await self.session.call_tool("extract_info", {"paper_id": pid})
                info_text = info_result.content[0].text
                print(f"📄 Info for {pid}:\n{info_text[:400]}...\n")

                # Try getting full text using get_full_text tool
                full_text = None
                if "get_full_text" in {tool["function"]["name"] for tool in self.available_tools}:
                    print(f"📥 Downloading full text for {pid}")
                    full_text_result = await self.session.call_tool("get_full_text", {"paper_id": pid})
                    full_text = full_text_result.content[0].text
                    print(f"📄 Full text (truncated preview):\n{full_text[:500]}...\n")

                # Choose what to summarize
                text_to_summarize = full_text or info_text  # Fallback to metadata if full text fails

                if "summarize_paper" in {tool["function"]["name"] for tool in self.available_tools}:
                    print(f"🧠 Summarizing {pid}")
                    summary_result = await self.session.call_tool("summarize_paper", {"text": text_to_summarize})
                    summary_text = summary_result.content[0].text
                    print(f"✅ Summary for {pid}:\n{summary_text}\n")
                else:
                    print("❌ summarize_paper tool not available.")
            except Exception as e:
                print(f"⚠️ Error during extract/full_text/summarize chain for {pid}: {e}")

    async def process_query(self, query: str):
        messages = self.message_history + [{"role": "user", "content": query}]
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
            max_completion_tokens=1024
        )

        assistant_message = response.choices[0].message
        available_tool_names = {tool["function"]["name"] for tool in self.available_tools}
        self.message_history.append({"role": "user", "content": query})
        self.message_history.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    print(f"🔧 Raw tool args: {tool_call.function.arguments}")
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"❌ JSONDecodeError while parsing tool arguments: {e}")
                    continue

                print(f"🛠️ Calling tool '{tool_name}' with args: {tool_args}")
                try:
                    result = await self.session.call_tool(tool_name, arguments=tool_args)
                    tool_result_text = (
                        result.content[0].text if hasattr(result.content[0], "text") else str(result.content)
                    )
                    self.message_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text
                    })

                    # Optional print toggle
                    if tool_name not in {"extract_info", "summarize_paper"}:
                        print("🧩 Tool Result:", result.content)

                    # Special handling for tool chaining
                    if tool_name == "search_papers" and {"extract_info", "summarize_paper"}.issubset(available_tool_names):
                        try:
                            paper_ids = json.loads(tool_result_text)
                            await self.handle_search_and_summarize(paper_ids)
                        except Exception as e:
                            print(f"⚠️ Tool chaining error: {e}")
                except Exception as e:
                    print(f"❗ Failed to call tool {tool_name}: {e}")

            # Final model response with full message history
            followup = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=self.message_history,
                max_completion_tokens=1024
            )
            print(followup.choices[0].message.content)
            self.message_history.append(followup.choices[0].message)
        else:
            print(assistant_message.content)

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self, single_query: str = None):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "research_server.py"],
            env=None
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()

                response = await session.list_tools()
                print("\n✅ Connected to server with tools:", [tool.name for tool in response.tools])

                self.available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.inputSchema.get("properties", {}),
                                "required": tool.inputSchema.get("required", [])
                            }
                        }
                    } for tool in response.tools
                ]

                if single_query:
                    await self.process_query(single_query)
                else:
                    await self.chat_loop()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Run a single query in non-interactive mode")
    args = parser.parse_args()

    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run(single_query=args.query)


if __name__ == "__main__":
    asyncio.run(main())
