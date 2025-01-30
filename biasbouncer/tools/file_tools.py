import tempfile
import aiofiles
import os
import json
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st


search_tool = DuckDuckGoSearchRun()

# Ensure session state has a temp directory
def ensure_temp_dir():
    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="biasbouncer_")
    return st.session_state["temp_dir"]

# Function to list files in the session's temp directory
def list_files():
    temp_dir = ensure_temp_dir()  # Ensure temp directory is initialized
    return os.listdir(temp_dir)

async def research_tool(query: str) -> str:
    """
    Calls the DuckDuckGo search API and returns summarized results.
    """
    try:
        results = search_tool.invoke(query)
        return "\n".join([result["content"] for result in results])
    except Exception as e:
        return f"Error fetching search results: {str(e)}"

# Function to write a file and trigger UI update
async def write_tool(filename: str, content: str):
    try:
        temp_dir = ensure_temp_dir()  # Ensure temp directory is initialized
        temp_file_path = os.path.join(temp_dir, filename)
        
        mode = 'a' if os.path.exists(temp_file_path) else 'w'
        async with aiofiles.open(temp_file_path, mode=mode) as file:
            await file.write(content + '\n')

        st.session_state["file_updated"] = True  # Trigger UI refresh

        return f"Successfully wrote to '{temp_file_path}'."
    except Exception as e:
        return f"Error writing to temporary file: {str(e)}"

# Function to read a file
async def read_tool(filename: str):
    try:
        temp_dir = ensure_temp_dir()
        temp_file_path = os.path.join(temp_dir, filename)

        if os.path.exists(temp_file_path):
            async with aiofiles.open(temp_file_path, mode='r') as file:
                content = await file.read()
            return content
        else:
            return f"Error: Temporary file '{filename}' does not exist."
    except Exception as e:
        return f"Error reading from temporary file: {str(e)}"

# Function to process tool invocation from JSON
async def process_tool_invocation_temp(tool_json: str) -> str:
    try:
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