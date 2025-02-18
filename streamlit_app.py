import streamlit as st

import asyncio
import re
import json
from typing import List, Dict, Callable

from langchain_openai import ChatOpenAI, OpenAI
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

# Check for DeepSeek API key
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("DeepSeek API key not found in Streamlit secrets.")
    st.stop()

# Retrieve API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]

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
    """
    Uses an LLM to analyze the user query and determine up to {agent_number} relevant perspectives.
    """
    llm_instance =  OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

    template = f"""
    Identify a list of up to {agent_number} of perspectives or advocates that could respond to the user's 
    problem or question with different solutions. If the user lists different perspectives or sides of an 
    argument, only use their suggestions. If they do not, create them in a way that will foster a conversation 
    between diverse perspectives. Return them as comma-separated values.

    User query: {message}
    """
    prompt = PromptTemplate(input_variables=["message", "agent_number"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt, model="deepseek-reasoner")

    response = await asyncio.to_thread(chain.run, message=message, agent_number=agent_number)
    companies = [item.strip() for item in response.split(",") if item.strip()]
    print(companies)
    return companies[:agent_number]

async def generate_response(
    company: str,
    user_message: str,
    conversation_so_far: str,
    all_perspectives: List[str],
    read_tool: Callable = None,
    write_tool: Callable = None,
    research_tool: Callable = None,
    scrape_webpage_tool: Callable = None
) -> str:
    """
    Generates a short, informal, brainstorming-style response from the perspective of `company`.
    """
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

    If no tool is needed, do not include the JSON block. You can only create .txt files, and ONLY create them when told to.
    You can ONLY use one tool per response, so do NOT include a JSON block in your second response if you have one. ALWAYS include
    as much direct information, figures, or quotes from your web research as you can. List your sources in bullet points in the format:
    "title," author/organization, website URL (name the link 'Source' always). ALWAYS ask the user before scraping any webpages.
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

    # Check for JSON tool requests
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        try:
            tool_data = json.loads(json_match.group(1))

            # Handle file reading
            if tool_data["tool"] == "read" and read_tool:
                with st.spinner("Reading Files"):
                    filename = tool_data["filename"]
                    read_data = await read_tool(filename)

                    # Modify conversation history to include file content
                    updated_conversation = conversation_so_far + f"\n\n[File '{filename}' was read and contained:]\n{read_data}"

                    # Rerun the agent with the new context
                    second_response = await asyncio.to_thread(
                        chain.run,
                        company=company,
                        user_message=user_message,
                        conversation_so_far=updated_conversation,
                        all_perspectives=", ".join(all_perspectives)
                    )

                    return second_response

            # Handle file writing
            elif tool_data["tool"] == "write" and write_tool:
                with st.spinner("Writing to File"):
                    filename = tool_data["filename"]
                    content = tool_data["content"]
                    write_result = await write_tool(filename, content)
                    return f"{response}\n\n[Tool Output]: {write_result}"

            # Handle web research
            elif tool_data["tool"] == "research" and research_tool:
                with st.spinner("Searching the Web"):
                    query = tool_data["query"]
                    search_results = await research_tool(query)

                    # Modify conversation history to include research results
                    updated_conversation = conversation_so_far + f"\n\n[Research on '{query}':]\n{search_results} | Remember, ALWAYS include as much direct information, figures, or quotes from your web research as you can. List your sources in bullet points with the title of the source and the author of the source."

                    # Rerun the agent with new knowledge
                    second_response = await asyncio.to_thread(
                        chain.run,
                        company=company,
                        user_message=user_message,
                        conversation_so_far=updated_conversation,
                        all_perspectives=", ".join(all_perspectives)
                    )

                    return second_response
                
            elif tool_data["tool"] == "scrape_webpage" and scrape_webpage_tool:
                with st.spinner("Reading Web Pages"):
                    url = tool_data["url"]
                    webscrape_results = await scrape_webpage_tool(url)

                    # Modify conversation history to include web scrape results
                    updated_conversation = conversation_so_far + f"\n\n[Information from website '{url}':]\n{webscrape_results.get('content', 'No content extracted.')} \n Remember, ALWAYS include as much direct information, figures, or quotes from the website as you can. DO NOT call up the tool again in this response."

                    # Rerun the agent with new knowledge
                    second_response = await asyncio.to_thread(
                        chain.run,
                        company=company,
                        user_message=user_message,
                        conversation_so_far=updated_conversation,
                        all_perspectives=", ".join(all_perspectives)
                    )

                    return second_response

        except (json.JSONDecodeError, KeyError):
            return f"Error parsing JSON tool invocation in the response:\n{response}"

    return response.strip()


async def run_agents(
    companies: List[str],
    user_message: str,
    conversation: List[Dict[str, str]],
    read_tool: Callable = read_tool,
    write_tool: Callable = write_tool,
    research_tool: Callable = research_tool,
    scrape_webpage_tool: Callable = scrape_webpage_tool
) -> Dict[str, str]:
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in conversation
    )

    tasks = [
        generate_response(
            company=company,
            user_message=user_message,
            conversation_so_far=conversation_text,
            all_perspectives=companies,
            read_tool=read_tool,
            write_tool=write_tool,
            research_tool=research_tool,
            scrape_webpage_tool=scrape_webpage_tool
        )
        for company in companies
    ]
    results = await asyncio.gather(*tasks)
    return dict(zip(companies, results))


# ------------------------------------------------------------------------------
# 6. Main Page Layout
# ------------------------------------------------------------------------------

LOGO_URL_LARGE = "biasbouncer/images/biasbouncer-logo.png"

st.logo(
    image=LOGO_URL_LARGE,
    link="https://biasbouncer.com",
    size="large"
)

st.markdown("<h1 style='text-align: center;'><span style='color: red;'>BiasBouncer</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Team WorkBench</h3>", unsafe_allow_html=True)

st.divider()

@st.dialog("How BiasBouncer Works")
def explain():
    st.divider()
    st.write("Every idea begins to take shape with a Brainstorming session. Click the sidebar arrow at the top left-hand corner of the page. Ask a question or pitch your idea to BiasBouncer in the Brainstorm Chat. From there, BiasBouncer will explore the different perspectives needed to accurately address the complications of your idea. Each agent will respond to you casually, outlining their informed thoughts or spitballing new ideas. Keep brainstorming with your team the same way you would with your friends or work partners. You can also create files in the chat to store your ideas. When you have a plan put together, you can begin orchestrating your work.")
    st.write("BiasBouncer will instruct agents to simultaneously perform different tasks like research, coding, reviewing, and more. Those tasks will populate in the 'To Do' column and gradually move to the right, so you can monitor their progress and review the task details. You can always chat with the agents if they have questions about their work or you want to add feedback. Once all the tasks have ended up in the 'Done' column, your finished project will be ready to download!")
    st.caption("As of the last update on 1/30/2025, the Brainstorm Chat with file-creation and web research capabilities are currently operational and available. Team WorkBench functionality coming soon.")
if st.button("How it Works", type="secondary"):
    explain()

col1, col2, col3, col4 = st.columns(4, border=True, gap="small")


with col1:
    st.subheader("To Do")
    @st.dialog("Task One")
    def view():
        st.html("<ul><li><h3>Agents: --</h3></li><li><h3>Tools: --</h3></li><li><h3>Description: --</h3></li><li><h3>Status: To Do</h3></li></ul>")
        st.markdown("##")
        st.write("Once your agents have created a plan in the Chat, the Tasks they'll work to complete will populate here. This is where you will be able to see the details of each task. If an agent has a question regarding their Task, they will ask you in the Chat.")
    if st.button("Task One", use_container_width=True, type="primary"):
        view()
        
    @st.dialog("Task Two")
    def view():
        st.html("<ul><li><h3>Agents: --</h3></li><li><h3>Tools: --</h3></li><li><h3>Description: --</h3></li><li><h3>Status: To Do</h3></li></ul>")
        st.markdown("##")
        st.write("Once your agents have created a plan in the Chat, the Tasks they'll work to complete will populate here. This is where you will be able to see the details of each task. If an agent has a question regarding their Task, they will ask you in the Chat.")
    if st.button("Task Two", use_container_width=True, type="primary"):
        view()

    @st.dialog("Task Three")
    def view():
        st.html("<ul><li><h3>Agents: --</h3></li><li><h3>Tools: --</h3></li><li><h3>Description: --</h3></li><li><h3>Status: To Do</h3></li></ul>")
        st.markdown("##")
        st.write("Once your agents have created a plan in the Chat, the Tasks they'll work to complete will populate here. This is where you will be able to see the details of each task. If an agent has a question regarding their Task, they will ask you in the Chat.")
    if st.button("Task Three", use_container_width=True, type="primary"):
        view()
        
    @st.dialog("Task Four")
    def view():
        st.html("<ul><li><h3>Agents: --</h3></li><li><h3>Tools: --</h3></li><li><h3>Description: --</h3></li><li><h3>Status: To Do</h3></li></ul>")
        st.markdown("##")
        st.write("Once your agents have created a plan in the Chat, the Tasks they'll work to complete will populate here. This is where you will be able to see the details of each task. If an agent has a question regarding their Task, they will ask you in the Chat.")
    if st.button("Task Four", use_container_width=True, type="primary"):
        view()

with col2:
    st.subheader("In Progress")
    st.markdown("##")

with col3:
    st.subheader("In Review")
    st.markdown("##")

with col4:
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


        # Run all agents concurrently
        with st.spinner("Preparing Responses..."):
            responses = asyncio.run(
                run_agents(st.session_state["companies"], user_input, st.session_state["chat_history"])
            )

        # Append and display each agent's response
        for company, text in responses.items():
            st.session_state["chat_history"].append({"role": company, "content": text})
            with messages_container:
                st.chat_message("assistant").write(f"**{company}**: {text}")