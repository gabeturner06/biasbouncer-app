import streamlit as st

import asyncio
import re
import os
import json
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from biasbouncer.tools.file_tools import read_tool, write_tool, list_files, ensure_temp_dir
from biasbouncer.tools.research_tools import research_tool, scrape_webpage_tool

# ------------------------------------------------------------------------------
# 1. Configure page layout
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------
# 2. Access API Key(s)
# ------------------------------------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not found in Streamlit secrets.")
    st.stop()

api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------------------------------------------------------------
# 2. Session State initialization
# ------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    # We'll store conversation messages here in the format:
    # {"role": "user" OR "<company_name>", "content": "..."}opp
    st.session_state["chat_history"] = []

if "companies" not in st.session_state:
    # Dynamically generated list of 'perspectives'
    st.session_state["companies"] = []

llm = ChatOpenAI(temperature=0)  # Base LLM (not used directly below but you can adapt)
    
# ------------------------------------------------------------------------------
# 5. Multi-Agent Creation System
# ------------------------------------------------------------------------------

async def determine_companies(message: str, agent_number: int) -> List[str]:
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    template = f"""
    Identify a list of up to {agent_number} of perspectives or advocates that could respond to the user's 
    problem or question with different solutions. If the user lists different perspectives or sides of an 
    argument, only use their suggestions. If they do not, create them in a way that will foster a conversation 
    between diverse perspectives. Return them as comma-separated values.

    User query: {message}
    """
    prompt = PromptTemplate(input_variables=["message", "agent_number"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(chain.run, message=message, agent_number=agent_number)
    companies = [item.strip() for item in response.split(",") if item.strip()]
    return companies[:agent_number]

async def handle_tool_request(tool_data, chain, company, user_message, conversation_so_far, all_perspectives):
    updated_conversation = conversation_so_far
    if tool_data["tool"] == "read" and read_tool:
        with st.spinner("Reading Files"):
            filename = tool_data["filename"]
            read_data = await read_tool(filename)
            updated_conversation += f"\n\n[File '{filename}' content:]\n{read_data}"
    elif tool_data["tool"] == "write" and write_tool:
        with st.spinner("Writing to File"):
            filename = tool_data["filename"]
            content = tool_data["content"]
            write_result = await write_tool(filename, content)
            return f"{write_result}"
    elif tool_data["tool"] == "research" and research_tool:
        with st.spinner("Searching the Web"):
            query = tool_data["query"]
            search_results = await research_tool(query)
            updated_conversation += f"\n\n[Research on '{query}':]\n{search_results}"
    elif tool_data["tool"] == "scrape_webpage" and scrape_webpage_tool:
        with st.spinner("Reading Web Pages"):
            url = tool_data["url"]
            scrape_results = await scrape_webpage_tool(url)
            updated_conversation += f"\n\n[Webpage '{url}' info:]\n{scrape_results.get('content', 'No content.')}"
    else:
        return None
    informed_response = await asyncio.to_thread(
        chain.run,
        company=company,
        user_message=user_message,
        conversation_so_far=updated_conversation,
        all_perspectives=", ".join(all_perspectives)
    )
    return informed_response

async def generate_response(company: str, user_message: str, conversation_so_far: str, all_perspectives: List[str]) -> str:
    llm_instance = ChatOpenAI(temperature=0.7, model="gpt-4")
    template = """
    You're in a casual group brainstorming chat trying to accurately and helpfully respond to a user query {user_message}. 
    You're going to answer from the perspective of a {company}, so you MUST role-play from this perspective to accurately
    respond to the user's query.

    Here is the chat history: {conversation_so_far}

    Here are all of the perspectives in this conversation with the user: {all_perspectives}. Remember, you're only representing 
    {company}; other agents will represent the others.

    Please reply briefly and informally, as if you're a professional brainstorming with friends in a group 
    chat. It is meant to be a quick, collaborative brainstorm session with the user, where you discuss and evaluate ideas 
    created by the user, and briefly explain your reasoning. In other words, your response shouldn't be much longer than the
    question asked by the user. Take note of the other perspectives present, so you can try to differentiate your ideas from theirs. 
    If you're instructed to do nothing, then just reply sure thing and do nothing.

    If you need to read, write, or research something online, include a JSON block in your response in the following format:

    
    ```json
    {{
        "tool": "read", "write", "research" or "scrape_webpage",
        "filename": "filename" (only for read/write, do NOT include any other filepaths or folders),
        "content": "(Your agent name): content-to-write" (only for 'write'),
        "query": "search query here" (only for 'research'),
        "url": "full url of the website you want to scrape" (only for 'scrape_webpage')
    }}

    If no tool is needed, do not include the JSON block. You can create .pdf (preferred if appropriate and file type not mentioned), 
    .txt, .docx, .csv, .xlsx, .html, .css, and .json files, but ONLY create them when told to. You can ONLY use one tool per response, 
    so do NOT include a JSON block in your second response if you have one. ALWAYS include as much direct information, figures, or quotes 
    from your web research as you can. List your sources in bullet points in the format: "title," author/organization, website URL (name 
    the link 'Source' always). ALWAYS ask the user before scraping any webpages.
    """
    prompt = PromptTemplate(
        input_variables=["company", "user_message", "conversation_so_far", "all_perspectives"],
        template=template
    )
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(
        chain.run,
        company=company,
        user_message=user_message,
        conversation_so_far=conversation_so_far,
        all_perspectives=", ".join(all_perspectives)
    )
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        try:
            tool_data = json.loads(json_match.group(1))
            tool_response = await handle_tool_request(tool_data, chain, company, user_message, conversation_so_far, all_perspectives)
            if tool_response:
                return tool_response
        except (json.JSONDecodeError, KeyError):
            return f"Error parsing tool invocation:\n{response}"
    return response.strip()

async def run_agents(companies: List[str], user_message: str, conversation: List[Dict[str, str]]) -> Dict[str, str]:
    max_length = 5000  # Character limit
    conversation_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)
    if len(conversation_text) > max_length:
        conversation_text = conversation_text[-max_length:]
    tasks = [generate_response(company, user_message, conversation_text, companies) for company in companies]
    results = await asyncio.gather(*tasks)
    return dict(zip(companies, results))

# ------------------------------------------------------------------------------
# 6. Main Page Layout
# ------------------------------------------------------------------------------

col1, col2, col3 = st.columns([0.15, 0.75, 0.1])

with col1:
    @st.dialog("How BiasBouncer Works")
    def explain():
        st.divider()
        st.write("Every idea begins with a Brainstorming session. Click the sidebar arrow at the top left, then ask a question or pitch your idea in the Brainstorm Chat. BiasBouncer explores different perspectives to address your idea’s complications, with each agent responding casually and outlining their reasoning. You can also create files in the chat to store your ideas. Once your plan is ready, orchestrate your work!")
        st.write("Agents perform tasks like research, coding, or reviewing. These tasks populate in the 'To Do' column and gradually move to 'Done' for review. Chat with agents for feedback, and when all tasks are complete, your finished project will be ready to download!")
        st.caption("As of 2/22/2025, Brainstorm Chat (with file-creation and web research) is operational. Team WorkBench functionality is coming soon.")
    if st.button("How it Works", type="primary"):
        explain()

with col2:
    pass

with col3:
    @st.dialog("Donate to BiasBouncer")
    def explain():
        st.divider()
        st.write("Your support goes into developing BiasBouncer and keeping it free to try for our very first users like you. Thank you!")
        st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <a href="https://donate.stripe.com/bIY2bbeYC1I2208002?locale=en" target="_blank">
                <button style="background-color:#FF4B4B; color:white; padding:10px 20px; font-size:16px; border:none; border-radius:5px; cursor:pointer;">
                    Donate with Stripe
                </button>
            </a>
        </div>
        <br>
        """,
        unsafe_allow_html=True
        )
    if st.button("Donate", type="secondary"):
        explain()

LOGO_URL_LARGE = "biasbouncer/images/biasbouncer-logo.png"
st.logo(image=LOGO_URL_LARGE, link="https://biasbouncer.com", size="large")
st.markdown("<h1 style='text-align: center;'><span style='color: red;'>BiasBouncer</span></h1>", unsafe_allow_html=True)

st.divider()

st.subheader("Team WorkBench")


def create_task_dialog(task_name: str):
    @st.dialog(task_name)
    def view():
        st.html(
            "<ul>"
            "<li><h3>Agents: --</h3></li>"
            "<li><h3>Tools: --</h3></li>"
            "<li><h3>Description: --</h3></li>"
            "<li><h3>Status: To Do</h3></li>"
            "</ul>"
        )
        st.markdown("##")
        st.write("Once your agents create a plan in the Chat, the Tasks they'll work on will populate here. If an agent has a question regarding their Task, they will ask you in the Chat.")
    return view

# Render Task Columns
cols = st.columns(4, border=True, gap="small")
task_names = ["Task One", "Task Two", "Task Three", "Task Four"]
with cols[0]:
    st.subheader("To Do")
    for task in task_names:
        task_view = create_task_dialog(task)
        if st.button(task, use_container_width=True, type="primary"):
            task_view()
with cols[1]:
    st.subheader("In Progress")
    st.markdown("##")
with cols[2]:
    st.subheader("In Review")
    st.markdown("##")
with cols[3]:
    st.subheader("Done")
    st.markdown("##")

st.divider()

st.subheader("Agent Files & Uploaded Docs")

col1, col2 = st.columns([0.4, 0.6])

with col1:
    if st.button("Refresh Files", use_container_width=True):
        st.rerun()

    files = list_files()

    if files:
        selected_file = st.selectbox("Select a file:", files)

        # Define the modal for viewing file content
        # Define the modal for viewing and downloading file content
        @st.dialog(f"{selected_file}")
        def view_file():
            with st.spinner("Reading file..."):

                file_content = asyncio.run(read_tool(selected_file))
            
                st.text_area("File Content", file_content, height=400)

                temp_dir = ensure_temp_dir()
                temp_file_path = os.path.join(temp_dir, selected_file)

                if not os.path.exists(temp_file_path):
                    st.error("Error: File does not exist.")
                    return

                file_ext = selected_file.lower().split('.')[-1]

                if file_ext in ["txt", "md", "py"]:
                    with open(temp_file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    mime_types = {
                        "txt": "text/plain",
                        "md": "text/markdown",
                        "py": "text/x-python",
                    }
                    mime_type = mime_types[file_ext]

                    st.text_area("File Content", file_content, height=400)

                elif file_ext in ["pdf", "docx", "xlsx"]:
                    with open(temp_file_path, "rb") as f:
                        file_content = f.read()
                    mime_types = {
                        "pdf": "application/pdf",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    }
                    mime_type = mime_types[file_ext]

                else:
                    st.error(f"Unsupported file type: {file_ext}")
                    return

                # Download button with correct MIME type
                st.download_button(
                    label="Download File",
                    data=file_content,
                    file_name=selected_file,
                    mime=mime_type,
                )


        if st.button("View File", use_container_width=True, type="secondary"):
            view_file()

    else:
        st.write("No files created.")

# ------------------------------------------------------------------------------
# 7. Sidebar Chat
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("Brainstorm Chat")

    user_input = st.chat_input("Work with the Agents")

    col1, col2 = st.columns([0.5,0.5])

    with col1:
        @st.dialog("Upload Files")
        def upload_files():
            uploaded_files = st.file_uploader(
                "Upload a File to BiasBouncer. Remember to give the agents the full name of the file (including file type) to read in your question.",
                accept_multiple_files=True
            )

            if uploaded_files:
                temp_dir = ensure_temp_dir()  # Ensure temp directory exists
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)

                    # Avoid overwriting existing files by appending a counter if needed
                    base_name, ext = os.path.splitext(uploaded_file.name)
                    counter = 1
                    while os.path.exists(file_path):
                        file_path = os.path.join(temp_dir, f"{base_name}_{counter}{ext}")
                        counter += 1

                    # Save file in chunks (handles large files better)
                    with open(file_path, "wb") as f:
                        for chunk in uploaded_file.chunks() if hasattr(uploaded_file, "chunks") else [uploaded_file.read()]:
                            f.write(chunk)

                    st.write(f"✅ Uploaded: {os.path.basename(file_path)}")

                # Trigger UI update so files appear in the dropdown
                st.session_state["file_updated"] = True  
                st.rerun()  # Refresh UI

        if st.button("Upload", type="primary", use_container_width=True):
            upload_files()

    with col2:
        @st.dialog("Agent Settings")
        def agent_settings():
            st.divider()
            st.write("### Number of Agents")
            st.session_state["agent_number"] = st.slider("", 2, 6, st.session_state["agent_number"])
            
            if st.session_state["companies"]:
                st.divider()
                st.write("### Select Agents to Respond")
                
                # Check if all agents are selected by default
                all_selected = len(st.session_state["selected_agents"]) == len(st.session_state["companies"])
                all_agents_checkbox = st.checkbox("All Agents", value=all_selected)
                
                if all_agents_checkbox:
                    st.session_state["selected_agents"] = st.session_state["companies"]
                else:
                    st.session_state["selected_agents"] = [
                        company for company in st.session_state["companies"]
                        if st.checkbox(company, value=company in st.session_state["selected_agents"])
                    ]

        if st.button("Settings", use_container_width=True, type="secondary"):
            agent_settings()
    
    messages_container = st.container(height=700)

    # Display past conversation in the side bar
    with messages_container:
        for msg in st.session_state["chat_history"]:
            role = msg["role"]
            content = msg["content"]

            # If role is "user", show user bubble
            if role == "user":
                st.chat_message("user").write(content)
            else:
                # If role is one of the agent names, we show it as "assistant"
                # but label it with the role name
                st.chat_message("assistant").write(f"**{role}**: {content}")


        if "agent_number" not in st.session_state:
            st.session_state["agent_number"] = 4  # Default value

        if "selected_agents" not in st.session_state:
            st.session_state["selected_agents"] = []  # Agents selected by the user

        if user_input:
            # Add user's message to the chat
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with messages_container:
                st.chat_message("user").write(user_input)

            # If we haven't determined perspectives yet, do so now
            with st.spinner("Preparing Perspectives..."):
                if not st.session_state["companies"]:
                    st.session_state["companies"] = asyncio.run(determine_companies(user_input, st.session_state["agent_number"]))
                    st.session_state["selected_agents"] = st.session_state["companies"]  # Default to all agents for the first response

            # Run only selected agents
            with st.spinner("Preparing Responses..."):
                selected_companies = st.session_state["selected_agents"]
                responses = asyncio.run(run_agents(selected_companies, user_input, st.session_state["chat_history"]))

            # Append and display each selected agent's response
            for company, text in responses.items():
                st.session_state["chat_history"].append({"role": company, "content": text})
                with messages_container:
                    st.chat_message("assistant").write(f"**{company}**: {text}")