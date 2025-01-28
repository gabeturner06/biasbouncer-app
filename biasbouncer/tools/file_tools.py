import tempfile
import aiofiles
import os
import json

async def write_tool(filename: str, content: str) -> str:
    """
    Writes content to a named temporary file or appends to an existing one in the temp directory.
    """
    try:
        # Ensure the file is stored in the system's temp directory
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, filename)

        # Open the file in append mode or create it if it doesn't exist
        mode = 'a' if os.path.exists(temp_file_path) else 'w'
        async with aiofiles.open(temp_file_path, mode=mode) as file:
            await file.write(content + '\n')  # Append content with a newline
        
        return f"Successfully wrote to temporary file '{temp_file_path}'."
    except Exception as e:
        return f"Error writing to temporary file: {str(e)}"

async def read_tool(filename: str) -> str:
    """
    Reads content from a named temporary file in the temp directory.
    """
    try:
        # Ensure the file is in the system's temp directory
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, filename)

        if os.path.exists(temp_file_path):
            async with aiofiles.open(temp_file_path, mode='r') as file:
                content = await file.read()
            return content
        else:
            return f"Error: Temporary file '{temp_file_path}' does not exist."
    except Exception as e:
        return f"Error reading from temporary file: {str(e)}"


import json

async def process_tool_invocation_temp(tool_json: str) -> str:
    """
    Processes the tool invocation JSON to call the appropriate temp file read/write tool.
    """
    try:
        # Parse the JSON input
        tool_data = json.loads(tool_json)

        tool_name = tool_data.get("tool")
        filename = tool_data.get("filename")
        content = tool_data.get("content", "")

        if tool_name == "write_temp_file":
            return await write_tool(filename, content)
        elif tool_name == "read_temp_file":
            return await read_tool(filename)
        else:
            return f"Error: Unknown tool '{tool_name}'."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for tool invocation."
    except Exception as e:
        return f"Error processing tool invocation: {str(e)}"
