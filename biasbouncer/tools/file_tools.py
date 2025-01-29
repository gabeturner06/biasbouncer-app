import tempfile
import aiofiles
import os
import json
import uuid
import streamlit as st

def delete_all_files():
    temp_dir = tempfile.gettempdir()
    files = list_files()
    for file in files:
        file_path = os.path.join(temp_dir, file)
        os.remove(file_path)

def get_app_temp_dir():
    temp_dir = tempfile.gettempdir()
    app_temp_dir = os.path.join(temp_dir, "biasbouncer")
    os.makedirs(app_temp_dir, exist_ok=True)  # Ensure the directory exists
    return app_temp_dir

# Ensure each session gets a unique temp directory
if "temp_dir" not in st.session_state:
    st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="biasbouncer_")

# Function to list files in the session's temp directory
def list_files():
    return [f for f in os.listdir(st.session_state["temp_dir"])]

# Function to write a file and trigger UI update
async def write_tool(filename: str, content: str):
    try:
        temp_dir = get_app_temp_dir()
        temp_file_path = os.path.join(temp_dir, filename)
        
        mode = 'a' if os.path.exists(temp_file_path) else 'w'
        async with aiofiles.open(temp_file_path, mode=mode) as file:
            await file.write(content + '\n')

        # Trigger UI refresh by setting session state
        st.session_state["file_updated"] = True

        return f"Successfully wrote to '{temp_file_path}'."
    except Exception as e:
        return f"Error writing to temporary file: {str(e)}"

# Function to read a file
async def read_tool(filename: str):
    try:
        temp_dir = get_app_temp_dir()
        temp_file_path = os.path.join(temp_dir, filename)

        if os.path.exists(temp_file_path):
            async with aiofiles.open(temp_file_path, mode='r') as file:
                content = await file.read()
            return content
        else:
            return f"Error: Temporary file '{temp_file_path}' does not exist."
    except Exception as e:
        return f"Error reading from temporary file: {str(e)}"

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