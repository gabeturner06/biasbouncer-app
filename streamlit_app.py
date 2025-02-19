import streamlit as st

import asyncio
import re
import json
from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from biasbouncer.tools.file_tools import read_tool, write_tool, list_files
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

if "agents" not in st.session_state:
    # Dynamically generated list of 'perspectives'
    st.session_state["agents"] = []

llm = ChatOpenAI(temperature=0)  # Base LLM (not used directly below but you can adapt)
    
# ------------------------------------------------------------------------------
# 5. Multi-Agent Creation System
# ------------------------------------------------------------------------------

def get_conversation_so_far() -> str:
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["chat_history"]])

async def determine_agents(user_message: str, agent_number: int) -> List[Dict[str, str]]:
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    template = f"""
    Identify up to {agent_number} agents that could respond to the user's query. 
    Assign each agent a role in the conversation based on their perspective.
    Return output as JSON: [{{"agent_name": "name", "agent_role": "role in conversation"}}, ...]
    
    User query: {user_message}
    """
    prompt = PromptTemplate(input_variables=["user_message", "agent_number"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(chain.run, user_message=user_message, agent_number=agent_number)
    return json.loads(response)

async def manager_agent(agent_name: str, agent_role: str, user_message: str) -> Dict[str, str]:
    conversation_so_far = get_conversation_so_far()
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    template = """
    Given the agent '{agent_name}' with role '{agent_role}', determine who should act first based on the user's query.
    If tools are needed (research, reading, writing, scraping), the Worker acts first. If no tools are needed, the Speaker acts first.
    
    Return output as JSON: {{ "agent_type": "Worker" or "Speaker" }}
    
    User query: {user_message}
    Here is the conversation with the user so far: {conversation_so_far}
    """
    prompt = PromptTemplate(input_variables=["agent_name", "agent_role", "user_message", "conversation_so_far"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(chain.run, agent_name=agent_name, agent_role=agent_role, user_message=user_message, conversation_so_far=conversation_so_far)
    return json.loads(response)

async def worker_agent(agent_name: str, user_message: str) -> str:
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    tool_output = ""
    
    while True:
        template = """
        You're a Worker agent for an '{agent_name}', responsible for handling tool-based operations in response to the user's query: {user_message}.
        Generate a JSON block specifying what tool to use and with what content. Based on the previous tool output, determine what tool needs to be
        used next, if any.
        
        If you need to read, write, or research something online, include a JSON block in your response in the following format:
    
        
        ```json
        {{
            "tool": "read", "write", "research" or "scrape_webpage",
            "filename": "filename" (only for read/write, do NOT include any other filepaths or folders),
            "content": "(Your agent name): content-to-write" (only for 'write'),
            "query": "search query here" (only for 'research'),
            "url": "full url of the website you want to scrape" (only for 'scrape_webpage')
        }}
    
        If no tool is needed, do not include the JSON block. You can only create .txt files.
        
        Previous tool output (if any):
        {tool_output}
        """
        prompt = PromptTemplate(input_variables=["agent_name", "user_message", "tool_output"], template=template)
        chain = LLMChain(llm=llm_instance, prompt=prompt)
        tool_json = await asyncio.to_thread(chain.run, agent_name=agent_name, user_message=user_message, tool_output=tool_output)
        
        try:
            tool_data = json.loads(tool_json)
            if not tool_data:
                break  
            
            if tool_data.get("tool") == "research":
                tool_output = await research_tool(tool_data["query"])
            elif tool_data.get("tool") == "read":
                tool_output = await read_tool(tool_data["filename"])
            elif tool_data.get("tool") == "write":
                tool_output = await write_tool(tool_data["filename"], tool_data["content"])
            elif tool_data.get("tool") == "scrape_webpage":
                tool_output = await scrape_webpage_tool(tool_data["url"])
            else:
                break
        except json.JSONDecodeError:
            return "Error processing tool request."
    
    return tool_output or "No tool needed."

async def speaker_agent(agent_name: str, user_message: str, worker_results: str = "") -> str:
    conversation_so_far = get_conversation_so_far()
    llm_instance = ChatOpenAI(temperature=0.7, model="gpt-4")
    template = """
    You're in a casual group brainstorming chat trying to accurately and helpfully respond to a user query {user_message}. 
    You're going to answer from the perspective of a {agent_name}, so you MUST role-play from this perspective to accurately
    respond to the user's query.

    Here is the chat history: {conversation_so_far}

    Here are all of the perspectives in this conversation with the user: {all_perspectives}. Remember, you're only representing 
    {company}; other agents will represent the others.

    Please reply briefly and informally, as if you're a professional brainstorming with friends in a group 
    chat. It is meant to be a quick, collaborative brainstorm session with the user, where you discuss and evaluate ideas 
    created by the user, and briefly explain your reasoning. In other words, your response shouldn't be much longer than the
    question asked by the user. Take note of the other perspectives present, so you can try to differentiate your ideas from theirs. 
    If you're instructed to do nothing, then just reply sure thing and do nothing.

    ALWAYS include as much direct information, figures, or quotes from the results of web research as you can. List your sources in bullet 
    points in the format: "title," author/organization, website URL (name the link 'Source' always).
    
    If the user asked you to do research or write a file, a Worker Agent should already have compiled this information for you. 
    Worker Results (if any): {worker_results}
    """
    prompt = PromptTemplate(input_variables=["agent_name", "user_message", "conversation_so_far", "worker_results"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    return await asyncio.to_thread(chain.run, agent_name=agent_name, user_message=user_message, conversation_so_far=conversation_so_far, worker_results=worker_results)

async def run_agents(user_message: str, agent_number: int):
    agents = await determine_agents(user_message, agent_number)
    responses = {}
    
    for agent in agents:
        agent_name, agent_role = agent["agent_name"], agent["agent_role"]
        decision = await manager_agent(agent_name, agent_role, user_message)
        
        if decision["agent_type"] == "Worker":
            worker_results = await worker_agent(agent_name, user_message)
            responses[agent_name] = await speaker_agent(agent_name, user_message, worker_results)
        else:
            responses[agent_name] = await speaker_agent(agent_name, user_message)
    
    return responses

# ------------------------------------------------------------------------------
# 6. Main Page Layout
# ------------------------------------------------------------------------------

LOGO_URL_LARGE = "biasbouncer/images/biasbouncer-logo.png"
st.logo(image=LOGO_URL_LARGE, link="https://biasbouncer.com", size="large")
st.markdown("<h1 style='text-align: center;'><span style='color: red;'>BiasBouncer</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Team WorkBench</h3>", unsafe_allow_html=True)
st.divider()

@st.dialog("How BiasBouncer Works")
def explain():
    st.divider()
    st.write("Every idea begins with a Brainstorming session. Click the sidebar arrow at the top left, then ask a question or pitch your idea in the Brainstorm Chat. BiasBouncer explores different perspectives to address your idea’s complications, with each agent responding casually and outlining their reasoning. You can also create files in the chat to store your ideas. Once your plan is ready, orchestrate your work!")
    st.write("Agents perform tasks like research, coding, or reviewing. These tasks populate in the 'To Do' column and gradually move to 'Done' for review. Chat with agents for feedback, and when all tasks are complete, your finished project will be ready to download!")
    st.caption("As of 1/30/2025, Brainstorm Chat (with file-creation and web research) is operational. Team WorkBench functionality is coming soon.")
if st.button("How it Works", type="secondary"):
    explain()

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

st.subheader("Agent Files")

col1, col2 = st.columns([0.4, 0.6])

with col1:
    if st.button("Refresh Files", use_container_width=True):
        st.rerun()

    files = list_files()

    if files:
        selected_file = st.selectbox("Select a file:", files)

        # Define the modal for viewing file content
        @st.dialog(f"{selected_file}")
        def view_file():
            with st.spinner("Reading file..."):
                file_content = asyncio.run(read_tool(selected_file))
            
            st.text_area("File Content", file_content, height=400)

            # Add a download button
            st.download_button(
                label="Download File",
                data=file_content,  # File content as data
                file_name=selected_file,  # Keep the original filename
                mime="text/plain",  # Set appropriate MIME type
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

    messages_container = st.container()

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

    # Chat input at the bottom
    user_input = st.chat_input("Work with the Agents")
    agent_number = st.slider("Number of Agents", 2, 6, 4)
    if user_input:
        # Add user's message to the chat
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with messages_container:
            st.chat_message("user").write(user_input)

        # If we haven't determined perspectives yet, do so now
        with st.spinner("Preparing Perspectives..."):
            if not st.session_state["agents"]:
                # Pass agent_number along with the user_input
                st.session_state["agents"] = asyncio.run(determine_agents(user_input, agent_number))


        # Run all agents concurrently
        with st.spinner("Preparing Responses..."):
            responses = asyncio.run(
                run_agents(st.session_state["agents"], user_input, st.session_state["chat_history"])
            )

        # Append and display each agent's response
        for company, text in responses.items():
            st.session_state["chat_history"].append({"role": company, "content": text})
            with messages_container:
                st.chat_message("assistant").write(f"**{company}**: {text}")