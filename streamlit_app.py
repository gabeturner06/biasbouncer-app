import streamlit as st

import asyncio
import re
import json
from typing import List, Dict

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


async def agent_manager(message: str) -> str:
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    template = f"""
    Identify, based on the user query, what type of agent needs to work first: a Worker agent or a Speaker Agent.
    If the user query asks the agent to use tools like reading/writing files, researching, or scraping web pages,
    your response should be a Worker. If the user query doesn't ask for any of these things, your response should
    be a Speaker. Output ONLY either [Worker] or [Speaker] as your final response.

    User query: {message}
    """
    prompt = PromptTemplate(input_variables=["message"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(chain.run, message=message)
    return response

async def speaker_agent(company: str, user_message: str, conversation_so_far: str, worker_results: str, all_perspectives: List[str]) -> str:
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

        ALWAYS include as much direct information, figures, or quotes from the results of web research as you can. List your sources in bullet 
        points in the format: "title," author/organization, website URL (name the link 'Source' always).
        
        {worker_results}
    """

    # Remove worker_results if empty
    if not worker_results or worker_results == "No additional data was gathered by the Worker Agent.":
        template = template.replace("{worker_results}", "")  # Remove worker_results placeholder entirely

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
    
    return response.strip() if response else "Error: No response generated."

async def handle_tool_request(tool_data, chain, company, user_message):
    tool_results = ""
    if tool_data["tool"] == "read" and read_tool:
        with st.spinner("Reading Files"):
            filename = tool_data["filename"]
            read_data = await read_tool(filename)
            tool_results += f"\n\n[File '{filename}' content:]\n{read_data}"
    elif tool_data["tool"] == "write" and write_tool:
        with st.spinner("Writing to File"):
            filename = tool_data["filename"]
            content = tool_data["content"]
            tool_results += await write_tool(filename, content)
    elif tool_data["tool"] == "research" and research_tool:
        with st.spinner("Searching the Web"):
            query = tool_data["query"]
            search_results = await research_tool(query)
            tool_results += f"\n\n[Research on '{query}':]\n{search_results}"
    elif tool_data["tool"] == "scrape_webpage" and scrape_webpage_tool:
        with st.spinner("Reading Web Pages"):
            url = tool_data["url"]
            scrape_results = await scrape_webpage_tool(url)
            tool_results += f"\n\n[Webpage '{url}' info:]\n{scrape_results.get('content', 'No content.')}"
    else:
        return None
    informed_response = await asyncio.to_thread(
        chain.run,
        company=company,
        user_message=user_message,
        tool_results=tool_results
    )
    return informed_response

async def worker_agent(company: str, user_message: str, tool_results: List[str]) -> str:
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")
    template = """
        You're a Worker agent for an '{company}', responsible for handling tool-based operations in response to the user's query: {user_message}.
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
        
        Previous tool outputs (if any):
        {tool_results}
    """
    prompt = PromptTemplate(
        input_variables=["company", "user_message", "tool_results"],
        template=template
    )
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(
        chain.run,
        company=company,
        user_message=user_message,
        tool_results=", ".join(tool_results),
    )

    max_iterations = 5
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if not json_match:
            break  # No more tools needed

        try:
            tool_data_list = json.loads(json_match.group(1))
            if not isinstance(tool_data_list, list):
                tool_data_list = [tool_data_list]  # Convert to list if single dict

            for tool_data in tool_data_list:
                tool_response = await handle_tool_request(tool_data, chain, company, user_message)
                if tool_response and tool_response not in tool_results:
                    tool_results.append(tool_response)  # ✅ Correctly append to tool_results list

            # Rerun the agent with updated tool_results
            response = await asyncio.to_thread(
                chain.run,
                company=company,
                user_message=user_message,
                tool_results=", ".join(tool_results),
            )

        except (json.JSONDecodeError, KeyError):
            return f"Error parsing tool invocation:\n{response}"

    return response.strip()

async def run_worker_agents(companies: List[str], user_message: str) -> Dict[str, List[str]]:
    tool_results = {company: [] for company in companies}  # Separate tool results for each agent
    async def worker_wrapper(company):
        return company, await worker_agent(company, user_message, tool_results[company])
    results = await asyncio.gather(*(worker_wrapper(company) for company in companies))
    return dict(results)  # Returns {"VC Investor": [...], "Tech Founder": [...], "Startup Lawyer": [...]} 

async def run_speaker_agents(companies: List[str], user_message: str, conversation: List[Dict[str, str]], tool_results: Dict[str, List[str]]) -> Dict[str, str]:
    conversation_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)

    async def speaker_wrapper(company):
        # Ensure worker_results exists
        worker_results = tool_results.get(company, [])
        if not worker_results:  # If None or empty, provide a default value
            worker_results = "No additional data was gathered by the Worker Agent."
        elif isinstance(worker_results, list):
            worker_results = "\n".join(worker_results)  # Convert list to string

        print(f"DEBUG: Calling speaker_agent for {company} with worker_results: {worker_results}")  # Debugging

        return company, await speaker_agent(company, user_message, conversation_text, worker_results, companies)

    results = await asyncio.gather(*(speaker_wrapper(company) for company in companies))
    return dict(zip(companies, results))

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
            if not st.session_state["companies"]:
                # Pass agent_number along with the user_input
                st.session_state["companies"] = asyncio.run(determine_companies(user_input, agent_number))
                
        # Agent Managers decide how to delegate work
        with st.spinner("Delegating Agents..."):
            decisions = asyncio.run(agent_manager(user_input))  # ✅ Use asyncio.run() here

        if decisions == "[Worker]":
            tool_results = asyncio.run(run_worker_agents(st.session_state["companies"], user_input))  # ✅ Use asyncio.run() here
            responses = asyncio.run(run_speaker_agents(st.session_state["companies"], user_input, st.session_state["chat_history"], tool_results))
        elif decisions == "[Speaker]":
            responses = asyncio.run(run_speaker_agents(st.session_state["companies"], user_input, st.session_state["chat_history"], {company: [] for company in st.session_state["companies"]}))

        if not responses or not isinstance(responses, dict):
            responses = {}

        # Iterate over responses safely
        for company, text in responses.items():
            if isinstance(text, tuple):  # Ensure we only extract the response
                text = text[1]

            st.session_state["chat_history"].append({"role": company, "content": text})
            with messages_container:
                st.chat_message("assistant").write(f"**{company}**: {text}")
