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

load_dotenv()  # Load environment variables from .env file

# Claude model identifier for API calls
MODEL_ID = "claude-3-7-sonnet-20250219"

class MCPClient:
    """MCP (Model Context Protocol) client for interacting with MCP servers and Claude.

    This client manages connections to MCP servers, handles tool execution,
    and provides an interactive interface for querying Claude with MCP tools.
    """

    def __init__(self):
        """Initialize the MCP client with session management and Anthropic API client.

        Sets up:
        - ClientSession for MCP server communication
        - AsyncExitStack for managing async context managers
        - Anthropic client for Claude API interactions
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server via stdio transport.

        Establishes a connection to an MCP server by launching the server script
        as a subprocess and communicating via stdin/stdout.

        Args:
            server_script_path: Path to the server script (.py, .js, or .ts file)

        Raises:
            ValueError: If server_script_path is not a .py, .js, or .ts file
        """
        # Determine script type based on file extension
        is_python = server_script_path.endswith('.py')
        is_ts = server_script_path.endswith('.ts')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_ts or is_js):
            raise ValueError("Server script must be a .py, .js, or .ts file")

        # Select appropriate runtime command
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # Establish stdio transport connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        # Create client session with elicitation callback for server requests
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

        # Initialize the session with the server
        await self.session.initialize()

    async def handle_elicitation(
        self,
        context: RequestContext[ClientSession, Any],
        params: ElicitRequestParams
    ) -> ElicitResult:
        """Handle elicitation requests from the server by prompting the user for input.

        When the MCP server needs additional information, it can elicit data from
        the user through this callback. The method prompts for each field defined
        in the schema and performs type conversion.

        Args:
            context: Request context from the MCP session
            params: Elicitation parameters including message and schema

        Returns:
            ElicitResult with action="accept" and collected user data
        """
        print(f"\n🤖 Server Request: {params.message}")

        # Collect user input for each field in the schema
        properties = params.requestedSchema.get('properties', {})
        user_data = {}

        # Prompt for each field defined in the schema
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            description = field_info.get('description', '')
            prompt = f"{field_name} ({field_type})"
            if description:
                prompt += f": {description}"
            prompt += ": "

            user_input = input(prompt).strip()

            # Convert to appropriate type based on schema
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
        """Retrieve available tools from the MCP server.

        Fetches the list of tools exposed by the server and formats them
        for use with the Claude API.

        Returns:
            List of tool definitions with name, description, and input schema
        """
        tools_response = await self.session.list_tools()
        # Format tools for Claude API compatibility
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
        """Retrieve available prompts from the MCP server.

        Returns:
            PromptsResponse containing available prompt templates
        """
        prompts_response = await self.session.list_prompts()
        return prompts_response

    async def _get_resources(self):
        """Retrieve available resources from the MCP server.

        Returns:
            ResourcesResponse containing available resources
        """
        resources_response = await self.session.list_resources()
        return resources_response

    async def _get_resource_templates(self):
        """Retrieve available resource templates from the MCP server.

        Returns:
            ResourceTemplatesResponse containing available resource templates
        """
        resource_templates_response = await self.session.list_resource_templates()
        return resource_templates_response

    async def process_query(self, query: str) -> str:
        """Process a query using Claude with access to MCP server tools.

        Implements an agentic loop where Claude can use MCP tools to answer
        the query. The loop continues until Claude provides a final response
        without requesting further tool use.

        Args:
            query: The user's query to process

        Returns:
            The final text response from Claude
        """
        # Initialize conversation with user query
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Fetch available tools from MCP server
        available_tools = await self._get_tools()

        # Initial Claude API call with tools
        response = self.anthropic.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            messages=messages,
            tools=available_tools
        )

        # Agentic loop - continue while Claude requests tool use
        while response.stop_reason == "tool_use":
            # Add Claude's response (including tool use requests) to conversation
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Execute all requested tool calls
            tool_results = []
            for content in response.content:
                if content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input

                    try:
                        # Call the tool via MCP session
                        result = await self.session.call_tool(tool_name, tool_args)

                        # Format result content
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
                        # Handle tool execution errors
                        print(f"Error calling tool {tool_name}: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })

            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get next response from Claude
            response = self.anthropic.messages.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=messages,
                tools=available_tools
            )

        # Extract final text response from Claude
        final_text = []
        for content in response.content:
            if hasattr(content, 'text'):
                final_text.append(content.text)

        return "\n".join(final_text)
    
    async def prompt(self, prompt_name: str):
        """Execute a named prompt template from the MCP server.

        Retrieves a prompt template from the server, collects required arguments
        from the user, generates the prompt, and processes it with Claude.

        Args:
            prompt_name: Name of the prompt template to execute
        """
        try:
            # Fetch available prompts from server
            prompts_response = await self._get_prompts()
            prompt_obj = next(
                (p for p in prompts_response.prompts if p.name == prompt_name),
                None
            )

            if not prompt_obj:
                print(f"Prompt '{prompt_name}' not found")
                return

            # Collect arguments for the prompt template
            arguments = {}
            if prompt_obj.arguments:
                for arg in prompt_obj.arguments:
                    required = "required" if arg.required else "optional"
                    user_input = input(f"{arg.name} ({required}): ").strip()

                    # Validate required arguments
                    if not user_input and arg.required:
                        print(f"Error: {arg.name} is required")
                        return

                    if user_input:
                        arguments[arg.name] = user_input

            # Generate the prompt with provided arguments
            prompt_result = await self.session.get_prompt(prompt_name, arguments=arguments)

            prompt = prompt_result.messages[0].content.text

            # Process the generated prompt with Claude
            response = await self.process_query(prompt)
            print(response)

        except McpError as e:
            print(f"Server Error: {e}\n")
            return
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")
            return


    async def read_file(self):
        """Read the contents of a file via MCP resource.

        Prompts the user for a file path and retrieves the file content
        through the MCP server's file resource.
        """
        try:
            file_name = input("Enter file path: ").strip()
            # Access file resource using file:/// URI scheme
            resource = await self.session.read_resource(f"file:///{file_name}")
            file_content = json.loads(resource.contents[0].text)["file_content"]

            print(f"File Content:\n {file_content}")
            return file_content
        except Exception as e:
            print(f"Error reading file: {e}")

    def _print_dir_listing(self, items: list[dict]):
        """Format and print a directory listing.

        Args:
            items: List of directory items with metadata (type, size, modified, name)
        """
        print("\nDirectory Listing:\n")
        print(f"{'Type':<10} {'Size':>10} {'Modified':<25} {'Name'}")
        print("-" * 70)
        for item in items:
            # Add icon based on item type
            type_icon = "📁" if item["type"] == "directory" else "📄"
            size = f"{item['size']} B"
            print(f"{type_icon:<2} {item['type']:<8} {size:>10}  {item['modified']:<25} {item['name']}")


    async def read_dir(self):
        """List the contents of the current directory via MCP resource.

        Retrieves and displays directory contents through the MCP server's
        directory resource.
        """
        try:
            # Access directory resource using dir:// URI scheme
            resource = await self.session.read_resource(f"dir://.")
            dir_list = json.loads(resource.contents[0].text)["items"]
            self._print_dir_listing(dir_list)
            return
        except Exception as e:
            print(f"Error reading directory: {e}")

    async def converse(self):
        """Start an interactive conversation mode with Claude.

        Allows the user to have a multi-turn conversation with Claude,
        where each query can trigger tool use. Exits when user types 'quit' or 'q'.
        """
        print("\nEntering conversation mode. Type 'quit' or 'q' to exit.")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() in ("quit", "q"):
                break  # Signal exit

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
        """Signal to exit the client.

        Returns:
            String "quit" to signal exit from chat loop
        """
        print("Exiting client...")
        return "quit"

    async def chat_loop(self):
        """Run the main interactive chat loop with menu-driven interface.

        Presents a menu of options including prompt execution, file operations,
        and conversation mode. Continues until user selects quit.
        """
        print("\nMCP Client Started!")
        print("Select from the menu or 'quit'/'q' to exit.")

        # Map menu choices to async functions
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
        """Clean up resources and close connections.

        Closes the async exit stack which manages all open connections
        and resources.
        """
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