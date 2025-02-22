import tempfile
import aiofiles
import os
import json
import streamlit as st
import fitz  # PyMuPDF

# Ensure session state has a temp directory
def ensure_temp_dir():
    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="biasbouncer_")
    return st.session_state["temp_dir"]

# Function to list files in the session's temp directory
def list_files():
    temp_dir = ensure_temp_dir()  # Ensure temp directory is initialized
    return os.listdir(temp_dir)

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


async def read_tool(filename: str):
    try:
        # First, check if file content is stored in session_state
        if filename == st.session_state.get("uploaded_filename") and "uploaded_file_content" in st.session_state:
            file_content = st.session_state["uploaded_file_content"]
            
            # Determine file extension
            file_ext = filename.lower().split('.')[-1]

            if file_ext == "txt":
                return file_content.decode("utf-8")  # Decode binary content into string
            elif file_ext == "pdf":
                return await read_pdf_from_bytes(file_content)  # Extract text from PDF bytes
            else:
                return f"Error: Unsupported file type '{file_ext}'. Only TXT and PDF are supported."

        # Fallback to reading from temporary storage
        temp_dir = ensure_temp_dir()
        temp_file_path = os.path.join(temp_dir, filename)

        if not os.path.exists(temp_file_path):
            return f"Error: Temporary file '{filename}' does not exist."

        file_ext = filename.lower().split('.')[-1]

        if file_ext == "txt":
            async with aiofiles.open(temp_file_path, mode='r', encoding='utf-8') as file:
                return await file.read()
        elif file_ext == "pdf":
            return await read_pdf(temp_file_path)
        else:
            return f"Error: Unsupported file type '{file_ext}'. Only TXT and PDF are supported."

    except Exception as e:
        return f"Error reading from file: {str(e)}"

async def read_pdf_from_bytes(file_bytes: bytes):
    """Extracts text from a PDF file stored as bytes."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")  # Load from bytes
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text else "Sorry, no text found in the PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

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