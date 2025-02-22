import tempfile
import aiofiles
import os
import streamlit as st
import fitz

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

# Function to read a file
async def read_tool(filename: str):
    try:
        temp_dir = ensure_temp_dir()
        temp_file_path = os.path.join(temp_dir, filename)

        if not os.path.exists(temp_file_path):
            return f"Error: Temporary file '{filename}' does not exist."

        # Check file extension
        file_ext = filename.lower().split('.')[-1]

        if file_ext == "txt":
            async with aiofiles.open(temp_file_path, mode='r', encoding='utf-8') as file:
                content = await file.read()
        elif file_ext == "pdf":
            content = await read_pdf(temp_file_path)
        else:
            return f"Error: Unsupported file type '{file_ext}'. Only TXT and PDF are supported."

        return content

    except Exception as e:
        return f"Error reading from file: {str(e)}"

async def read_pdf(pdf_path: str):
    """Extracts text from a PDF file asynchronously."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text else "Warning: No text found in the PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"