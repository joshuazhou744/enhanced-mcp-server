import asyncio
import sys
import json
from typing import Optional, Dict, Any, List, Union
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ElicitRequestParams, ElicitResult
from mcp.shared.context import RequestContext
from mcp.shared.exceptions import McpError

try:
    from fastmcp.client.transports import StreamableHttpTransport
    HAS_HTTP_TRANSPORT = True
except ImportError:
    HAS_HTTP_TRANSPORT = False

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

MODEL_ID = "claude-3-7-sonnet-20250219"

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_ts = server_script_path.endswith('.ts')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_ts or is_js):
            raise ValueError("Server script must be a .py, .js, or .ts file")
            
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

    async def handle_elicitation(
        self,
        context: RequestContext[ClientSession, Any],
        params: ElicitRequestParams
    ) -> ElicitResult:
        """Handle elicitation requests from the server by prompting the human user"""
        print(f"\n🤖 Server Request: {params.message}")

        # Collect user input for each field in the schema
        properties = params.requestedSchema.get('properties', {})
        user_data = {}

        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            description = field_info.get('description', '')
            prompt = f"{field_name} ({field_type})"
            if description:
                prompt += f": {description}"
            prompt += ": "

            user_input = input(prompt).strip()

            # Convert to appropriate type
            try:
                if field_type == 'integer':
                    user_data[field_name] = int(user_input)
                elif field_type == 'number':
                    user_data[field_name] = float(user_input)
                elif field_type == 'boolean':
                    user_data[field_name] = user_input.lower() in ('true', 'yes', '1')
                else:
                    user_data[field_name] = user_input
            except ValueError:
                print(f"Invalid input for {field_name}. Using as string.")
                user_data[field_name] = user_input

        return ElicitResult(action="accept", content=user_data)

    async def _get_tools(self) -> List[Dict[str, Any]]:
        """Retrieve available tools from the server"""
        tools_response = await self.session.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "MCP Tool",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        ]

        return tools
    
    async def _get_prompts(self):
        """Retrieve available prompts from the server"""
        prompts_response = await self.session.list_prompts()
        return prompts_response
    
    async def _get_resources(self):
        """Retrieve available resources from the server"""
        resources_response = await self.session.list_resources()
        return resources_response
    
    async def _get_resource_templates(self):
        """Retrieve available resource templates from the server"""
        resource_templates_response = await self.session.list_resource_templates()
        return resource_templates_response

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools
        
        Args:
            query: The user's query to process
            
        Returns:
            The final text response from Claude
        """
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Add server tools
        available_tools = await self._get_tools()

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
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
                        print(f"Error calling tool {tool_name}: {e}")
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
    
    async def prompt(self, prompt_name: str):
        """Execute a named prompt from the server"""
        try:
            prompts_response = await self._get_prompts()
            prompt_obj = next(
                (p for p in prompts_response.prompts if p.name == prompt_name),
                None
            )

            if not prompt_obj:
                print(f"Prompt '{prompt_name}' not found")
                return
            
            arguments = {}
            if prompt_obj.arguments:
                for arg in prompt_obj.arguments:
                    required = "required" if arg.required else "optional"
                    user_input = input(f"{arg.name} ({required}): ").strip()

                    if not user_input and arg.required:
                        print(f"Error: {arg.name} is required")
                        return

                    if user_input:    
                        arguments[arg.name] = user_input
            
            prompt_result = await self.session.get_prompt(prompt_name, arguments=arguments)

            prompt = prompt_result.messages[0].content.text

            response = await self.process_query(prompt)
            print(response)

        except McpError as e:
            print(f"Server Error: {e}\n")
            return
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")
            return


    async def read_file(self):
        """Read the contents of a file and process it with the agent"""
        try:
            file_name = input("Enter file path: ").strip()
            resource = await self.session.read_resource(f"file:///{file_name}")
            file_content = json.loads(resource.contents[0].text)["file_content"]
            
            print(f"File Content:\n {file_content}")
            return file_content
        except Exception as e:
            print(f"Error reading file: {e}")

    def _print_dir_listing(self, items: list[dict]):
        print("\nDirectory Listing:\n")
        print(f"{'Type':<10} {'Size':>10} {'Modified':<25} {'Name'}")
        print("-" * 70)
        for item in items:
            type_icon = "📁" if item["type"] == "directory" else "📄"
            size = f"{item['size']} B"
            print(f"{type_icon:<2} {item['type']:<8} {size:>10}  {item['modified']:<25} {item['name']}")


    async def read_dir(self):
        """List the contents of a directory and process it with the agent"""
        try:
            resource = await self.session.read_resource(f"dir://.")
            dir_list = json.loads(resource.contents[0].text)["items"]
            self._print_dir_listing(dir_list)
            return
        except Exception as e:
            print(f"Error reading directory: {e}")

    async def converse(self):
        """Start a conversation with the agent"""
        print("\nEntering conversation mode. Type 'quit' or 'q' to exit.")
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ("quit", "q"):
                break  # signal exit
                
            if not query:
                print("Please enter a query")
                continue
                
            try:
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"Error processing query: {e}")

        return

    async def quit_action(self):
        """Exit the client"""
        return "quit"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Select from the menu or 'quit'/'q' to exit.")
        
        menu_actions = {
            "1": lambda: self.prompt("documentation_generator"),
            "2": lambda: self.prompt("code_review"),
            "3": self.read_file,
            "4": self.read_dir,
            "5": self.converse,
            "q": self.quit_action,
            "quit": self.quit_action
        }

        while True:
            choice = input("""
Select from the Menu
1. Generate Documentation
2. Review Code
3. Read File
4. Read Current Directory
5. Converse with Agent
q. Quit
> """).strip()

            action = menu_actions.get(choice)

            if not action:
                print("Invalid choice. Please try again.")
                continue

            result = await action()
            if result == "quit":
                break
    
    async def cleanup(self):
        """Clean up resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_path>")
        sys.exit(1)

    client = MCPClient()
    try:
        server_path = sys.argv[1]
        print(f"Connecting to server: {server_path}")

        await client.connect_to_server(server_path)

        await client.chat_loop()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())