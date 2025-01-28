import aiofiles
from typing import Callable

async def read_tool(filename: str) -> str:
    """
    Asynchronously reads content from a file.
    
    Args:
        filename (str): Path to the file to read
        
    Returns:
        str: Content of the file or error message
    """
    path = f"/{filename}"
    try:
        async with aiofiles.open(path, mode="r") as file:
            content = await file.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{path}' not found."

async def write_tool(filename: str, content: str) -> str:
    """
    Asynchronously writes content to a file.
    
    Args:
        filename (str): Path to the file to write to
        content (str): Content to write to the file
        
    Returns:
        str: Success message or error message
    """
    path = f"/Users/gabrielryanturner/Documents/Source/archive/biasbouncer-auto-gpt/backup/{filename}"
    try:
        async with aiofiles.open(path, mode="a") as file:
            await file.write(content)
        return f"Successfully wrote to '{path}'."
    except Exception as e:
        return f"Error writing to '{path}': {str(e)}"

async def process_tool_invocation(tool_data: dict, read_tool: Callable, write_tool: Callable) -> str:
    """
    Processes tool invocation requests based on the parsed JSON data.
    
    Args:
        tool_data (dict): Dictionary containing tool invocation details
        read_tool (Callable): Function to handle reading from files
        write_tool (Callable): Function to handle writing to files
        
    Returns:
        str: Result message or error string
    """
    try:
        tool = tool_data.get("tool")
        filename = tool_data.get("filename")
        
        if tool == "read":
            return await read_tool(filename)
        elif tool == "write":
            content = tool_data.get("content", "")
            return await write_tool(filename, content)
        else:
            return f"Error: Unknown tool '{tool}' specified."
    except Exception as e:
        return f"Error processing tool invocation: {str(e)}"