import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ElicitRequestParams, ElicitResult
from mcp.shared.context import RequestContext

from anthropic import Anthropic
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()  # load environment variables from .env

MODEL_ID = "claude-3-7-sonnet-20250219"
MODEL_ID_OPENAI = "gpt-5-nano"

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.openai = OpenAI()

    async def handle_elicitation(
        self,
        context: RequestContext[ClientSession, any],
        params: ElicitRequestParams
    ) -> ElicitResult:
        """Handle elicitation requests from the server by prompting the human user"""
        print(f"\n🤖 Server Request: {params.message}")

        # Collect user input for each field in the schema
        properties = params.requestedSchema.get('properties', {})
        user_data = {}

        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            prompt = f"{field_name} ({field_type}): "

            user_input = input(prompt).strip()

            # Convert to appropriate type
            if field_type == 'integer':
                user_data[field_name] = int(user_input)
            elif field_type == 'number':
                user_data[field_name] = float(user_input)
            elif field_type == 'boolean':
                user_data[field_name] = user_input.lower() in ('true', 'yes', '1')
            else:
                user_data[field_name] = user_input

        return ElicitResult(action="accept", content=user_data)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_ts = server_script_path.endswith('.ts')
        if not (is_python or is_ts):
            raise ValueError("Server script must be a .py or .ts file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.stdio,
                self.write,
                elicitation_callback=self.handle_elicitation,
                client_info={
                    "name": "mcp-client",
                    "version": "1.0.0",
                }
            )
        )

        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        available_tools = []

        # Add server tools
        tools_response = await self.session.list_tools()
        available_tools.extend(
            {
                "name": tool.name,
                "description": tool.description or "MCP Tool",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        )

        # Add server prompts (same shape)
        prompts_response = await self.session.list_prompts()
        if prompts_response.prompts:
            prompt_list = []
            for p in prompts_response.prompts:
                args_desc = ""
                if p.arguments:
                    args = [f"{a.name} ({'required' if a.required else 'optional'})" for a in p.arguments]
                    args_desc = f" (args: {', '.join(args)})"
                prompt_list.append(f"{p.name}{args_desc}")
            
            available_tools.append({
                "name": "get_prompt",
                "description": f"Get a pre-defined prompt from the MCP server. Available prompts: {'; '.join(prompt_list)}. The prompt will be inserted into the conversation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt_name": {
                            "type": "string",
                            "description": "Name of the prompt to retrieve"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the prompt (if required)",
                            "additionalProperties": True
                        }
                    },
                    "required": ["prompt_name"]
                }
            })

        resource_templates = await self.session.list_resource_templates()

        templates_data = [f"URI: {template.uriTemplate}\n Description: {template.description or 'No description'}\n Example: {template.uriTemplate.replace('{file_path}', 'server.py').replace('{directory}', '.')}" for template in resource_templates.resourceTemplates]
        
        available_tools.append({
            "name": "read_resource",
            "description": f"Read an MCP resource. Available templates: \n" + "\n".join(templates_data),
            "input_schema": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The resource URI. Replace {parameters} with actual values."
                    }
                },
                "required": ["uri"]
            }
        })

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=MODEL_ID,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Agentic loop - handle tool use
        while response.stop_reason == "tool_use":
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Execute all tool calls
            tool_results = []
            for content in response.content:
                if content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    
                    try:
                        if tool_name == "read_resource":
                            # Strip trailing slash defensively
                            uri = tool_args["uri"].rstrip('/')
                            
                            resource = await self.session.read_resource(uri)
                            result_content = []
                            for content_part in resource.contents:
                                if content_part.text:
                                    result_content.append(content_part.text)
                                elif content_part.blob:
                                    result_content.append(f"[Binary: {len(content_part.blob)} bytes]")
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": "\n".join(result_content)
                            })
                        
                        elif tool_name == "get_prompt":
                            prompt_name = tool_args["prompt_name"]
                            prompt_arguments = tool_args.get("arguments", {})
                            
                            prompt_response = await self.session.get_prompt(
                                prompt_name,
                                arguments=prompt_arguments
                            )
                            
                            prompt_content = []
                            for message in prompt_response.messages:
                                text = (message.content.text 
                                       if hasattr(message.content, 'text') 
                                       else str(message.content))
                                prompt_content.append(f"[{message.role}]: {text}")
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": "\n\n".join(prompt_content)
                            })
                        
                        else:
                            result = await self.session.call_tool(tool_name, tool_args)
                            
                            if isinstance(result.content, list):
                                result_text = "\n".join([
                                    c.text if hasattr(c, 'text') else str(c)
                                    for c in result.content
                                ])
                            else:
                                result_text = result.content
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result_text
                            })
                    
                    except Exception as e:
                        print(f"Error: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })

            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get next response
            response = self.anthropic.messages.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=messages,
                tools=available_tools
            )

        # Extract final response
        final_text = []
        for content in response.content:
            if hasattr(content, 'text'):
                final_text.append(content.text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit'/'q' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit' or query.lower() == 'q':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())