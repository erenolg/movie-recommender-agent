import os
import json
import asyncio
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class MovieRecommenderAgent:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-20b"
        self.max_iterations = 10
        self.system_prompt = """You are a conversational movie recommendation assistant.

        You have access to tools that use collaborative filtering to find movies.
        Your job is to:
        1. Understand what the user wants — their reference movies, mood, constraints
        2. Call the right tool ONCE to get candidates — do not call the same tool twice
        3. Re-rank or filter the candidates yourself based on qualitative constraints
        the algorithm can't handle (tone, mood, "lighthearted", "dark", "feel-good" etc.)
        Use the genres field to guide this judgment:
        - lighthearted/fun → prefer Comedy, Animation, Family, Adventure
        - dark/intense → prefer Thriller, Crime, Drama, Horror
        - epic/action → prefer Action, Adventure, Sci-Fi
        4. Present 3-5 final recommendations with a brief reason for each

        Critical rules:
        - Call each tool AT MOST ONCE per user message — never repeat the same tool call
        - Once you have tool results, STOP calling tools and give your final answer immediately
        - Never recommend the movie the user already mentioned as their reference
        - Always explain WHY each movie fits what they asked for
        - If the user gives a qualitative constraint, explicitly acknowledge how you filtered for it
        - Be conversational, not robotic
        - NEVER make up ratings or scores — if you cannot get the data from a tool, say explicitly "I don't have that information in my database"
        """

    def _mcp_tool_to_groq_format(self, mcp_tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }

    async def chat(self, user_message: str, conversation_history: list) -> str:
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server/server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Discover tools from MCP server
                tools_response = await session.list_tools()
                groq_tools = [
                    self._mcp_tool_to_groq_format(t)
                    for t in tools_response.tools
                ]

                # Build messages with full conversation history
                messages = [
                    {"role": "system", "content": self.system_prompt}
                ] + conversation_history + [
                    {"role": "user", "content": user_message}
                ]

                # Agent loop
                for iteration in range(self.max_iterations):
                    response = self.groq_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto"
                    )

                    response_message = response.choices[0].message

                    # No tool calls - we have the final answer
                    if not response_message.tool_calls:
                        return response_message.content

                    # Add assistant's tool call decision to messages
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in response_message.tool_calls
                        ]
                    })

                    # Execute each tool call via MCP
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        print(f"  [Agent] Calling tool: {function_name}({function_args})")

                        result = await session.call_tool(function_name, function_args)
                        if result.content and len(result.content) > 0:
                            result_text = result.content[0].text
                        else:
                            result_text = json.dumps({"error": "Tool returned empty result"})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })

                return "I was unable to generate a recommendation. Please try again."


async def main():
    agent = MovieRecommenderAgent()
    conversation_history = []

    print("Movie Recommendation Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        print("Agent: thinking...\n")
        response = await agent.chat(user_input, conversation_history)

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())