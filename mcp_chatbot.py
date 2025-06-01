from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os
import json

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

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
            max_completion_tokens=1024
        )

        assistant_message = response.choices[0].message
        available_tool_names = {tool["function"]["name"] for tool in self.available_tools}

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    print(f"🔧 Raw tool args: {tool_call.function.arguments}")
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"❌ JSONDecodeError while parsing tool arguments: {e}")
                    print(f"🚫 Skipping tool {tool_name} due to malformed arguments.\n")
                    continue

                print(f"🛠️ Calling tool '{tool_name}' with args: {tool_args}")
                messages.append(assistant_message)

                try:
                    result = await self.session.call_tool(tool_name, arguments=tool_args)
                    print("🧩 Tool Result:", result.content)

                    tool_result_text = (
                        result.content[0].text if hasattr(result.content[0], "text") else str(result.content)
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text
                    })
                except Exception as e:
                    print(f"❗ Failed to call tool {tool_name}: {e}")
                    continue

                # Tool chaining for search_papers
                if tool_name == "search_papers" and {"extract_info", "summarize_paper"}.issubset(available_tool_names):
                    try:
                        paper_ids = json.loads(tool_result_text)
                        for pid in paper_ids[:3]:  # Limit for brevity
                            print(f"\n🔎 Extracting info for {pid}")
                            info_result = await self.session.call_tool("extract_info", {"paper_id": pid})
                            info_text = info_result.content[0].text
                            print(f"📄 Info for {pid}:\n{info_text[:400]}...\n")

                            print(f"🧠 Summarizing {pid}")
                            summary_result = await self.session.call_tool("summarize_paper", {"text": info_text})
                            summary_text = summary_result.content[0].text
                            print(f"✅ Summary for {pid}:\n{summary_text}\n")
                    except Exception as e:
                        print(f"⚠️ Tool chaining error: {e}")

            # Final response after tool use
            followup = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                max_completion_tokens=1024
            )
            print(followup.choices[0].message.content)
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
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
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
                print("\nConnected to server with tools:", [tool.name for tool in response.tools])

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

                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
