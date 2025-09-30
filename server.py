#!/usr/bin/env python3

import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel
from datetime import datetime

# Get project root directory (current working directory)
BASE_DIR = Path.cwd()

# Setup module-level logger
logger = logging.getLogger("FileOpServer")

# Documentation file name schema
class Docname(BaseModel):
    name: str

# Create FastMCP server with lifespan management
mcp = FastMCP("File Operations MCP Server")

# ============================================================================
# Helper Functions
# ============================================================================

def get_path(relative_path: str) -> Path:
    """Return an absolute Path inside the project base directory."""
    return BASE_DIR / relative_path

# ============================================================================
# TOOLS - File Operations
# ============================================================================
    

@mcp.tool()
async def write_file(file_path: str, content: str, ctx: Context) -> str:
    """Create a new file with specified content
    Args:
        file_path: Relative path where the file should be created
        content: Content to write to the file
    Returns:
        Success or error message
    """
    try:
        path = get_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

        await ctx.info(f"File written successfully to: {file_path}")
        return f"File written successfully to: {file_path}"
    except Exception as e:
        await ctx.error(f"Error creating file: {str(e)}")
        raise


@mcp.tool()
async def delete_file(file_path: str, ctx: Context) -> str:
    """Delete a file from the project directory

    Args:
        file_path: Relative path to the file to delete
    Returns:
        Success or error message
    """
    try:
        path = get_path(file_path)
        if path.is_file():
            path.unlink()
            await ctx.info(f"Successfully deleted file {file_path}")
            return f"Successfully deleted file {file_path}"
        elif path.is_dir():
            await ctx.warning(f"Error: {file_path} is a directory, not a file")
            return f"Error: {file_path} is a directory, not a file"
        else:
            await ctx.warning(f"File not found: {file_path}")
            return f"File not found: {file_path}"
    except Exception as e:
        await ctx.error(f"Error deleting file: {str(e)}")
        return f"Error deleting file: {str(e)}"


# ============================================================================
# RESOURCES - File Reading
# ============================================================================

@mcp.resource("file:///{file_path}")
async def read_file_resource(file_path: str) -> dict:
    """Read the content of a file as a resource
    Args:
        file_path: Relative path to the file
    Returns:
        File content as string
    """
    try:
        path = get_path(file_path)

        if not path.exists() or not path.is_file():
            return {"error": f"Error: {file_path} is not a valid file"}
        
        return {"file_content": path.read_text(encoding='utf-8')}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}


@mcp.resource("dir://{directory}")
async def list_files_resource(directory: str) -> dict:
    """List files in a directory
    Args:
        directory: Relative path to directory
    Returns:
        List of files and directories as newline-separated text
    """
    try:
        path = get_path(directory)
        if not path.exists() or not path.is_dir():
            return {"error": f"{path} is not a valid directory"}

        items = []
        for item in path.iterdir():
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })

        return {
            "directory": directory,
            "items": items
        }
    except Exception as e:
        return {"error": f"Error listing files: {e}"}

@mcp.resource("mcp://resource-templates")
def list_resource_templates() -> dict:
    """List available resource URI templates"""
    templates = {
        "file:///{file_path}": {
            "description": "Read any file by path",
            "parameters": {
                "file_path": "Relative path to the file (e.g., 'server.py', 'src/main.py')"
            },
            "examples": ["file:///server.py", "file:///src/main.py"]
        },
        "dir://{directory}": {
            "description": "List directory contents",
            "parameters": {
                "directory": "Relative path to directory (e.g., '.', 'src')"
            },
            "examples": ["dir://.", "dir://src"]
        }
    }
    return templates

# ============================================================================
# PROMPTS - Code Editing and Documentation
# ============================================================================

@mcp.prompt()
async def code_editor(file_path: str, instruction: str, ctx: Context) -> str:
    """Generate a prompt for code editing and refactoring
    Args:
        file_path: Path to the code file to edit
        instruction: Specific instruction for code modification
    Returns:
        Code editing prompt
    """
    try:
        path = get_path(file_path)

        if not path.exists() or not path.is_file():
            await ctx.warning(f"Error: {file_path} is not a valid file to code review")
            return f"Error: {file_path} is not a valid file to code review"
        
        current_code = path.read_text(encoding='utf-8').strip()
        language = path.suffix.lower()

        prompt = f"""You are an expert code editor. Please help me modify the following code file.

File: {file_path}
Language (file suffix): {language or "unknown"}
Instruction: {instruction}

Current code:
```
{current_code}
```

Please provide the updated code that follows the instruction. Make sure to:
1. Follow best practices for the programming language
2. Add appropriate comments if needed
3. Maintain code readability and structure
4. Handle edge cases appropriately

Return only the updated code without additional explanations unless specifically requested.""".strip()
        
        await ctx.info("Successfully returned prompt")
        return prompt

    except Exception as e:
        await ctx.error(f"Error preparing code review prompt: {e}")
        return f"Error preparing code review prompt: {e}"


@mcp.prompt()
async def documentation_generator(file_path: str, ctx: Context) -> str:
    """Generate a prompt for creating documentation
    Args:
        file_path: Path to the code file to document
    Returns:
        Documentation generation prompt
    """
    try:
        path = get_path(file_path)

        if not path.exists() or not path.is_file():
            await ctx.warning(f"Error: {file_path} is not a valid file to code review")
            return f"Error: {file_path} is not a valid file to code review"

        code = path.read_text(encoding='utf-8').strip()
        language = path.suffix.lower()

        result = await ctx.elicit(
            "Please provide the documentation file name",
            schema=Docname
        )
        doc_name = result.data

        prompt = f"""You are an expert technical writer and documentation specialist. Please help me create documentation for the following code file.

File: {file_path}
Language (file suffix): {language or "unknown"}

Current code:
```
{code}
```

Please provide:
1. Well-structured documentation following best practices
2. Clear explanations of functionality
3. Usage examples where appropriate
4. Documentation formatted in markdown  

Use MCP tools available to you to create the separate documentation file:
- **Name that separate document EXACTLY: {doc_name}**
- Add the .md suffix yourself if the name doesn't include it already""".strip()

        await ctx.info("Successfully returned prompt")
        return prompt

    except Exception as e:
        await ctx.error(f"Error generating code documentation prompt: {e}")
        return f"Error generating code documentation prompt: {e}"


# ============================================================================
# MAIN SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting File Operations Server...")
    mcp.run()