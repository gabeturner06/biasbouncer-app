import streamlit as st
import asyncio
import re
import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from biasbouncer.tools import file_tools, research_tools

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
st.set_page_config(layout="wide")
LOGO_URL = "biasbouncer/images/biasbouncer-logo.png"
TASK_NAMES = ["Task One", "Task Two", "Task Three", "Task Four"]
PROMPT_TEMPLATES = {
    "determine_companies": """Identify a list of up to {agent_number} of perspectives or advocates that could respond to the user's 
    problem or question with different solutions. If the user lists different perspectives or sides of an 
    argument, only use their suggestions. If they do not, create them in a way that will foster a conversation 
    between diverse perspectives. Return them as comma-separated values.

    User query: {message}""",
    
    "generate_response": """You're in a casual group brainstorming chat trying to accurately and helpfully respond to a user query {user_message}. 
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
    "title," author/organization, website URL (name the link 'Source' always). ALWAYS ask the user before scraping any webpages."""
}

# ==============================================================================
# 2. Session State Initialization
# ==============================================================================
DEFAULT_STATE = {
    "chat_history": [],
    "companies": [],
    "tasks": {task: {"status": "To Do"} for task in TASK_NAMES}
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)

# ==============================================================================
# 3. Core LLM Functions
# ==============================================================================
def create_llm(temp=0, model="gpt-4"):
    return ChatOpenAI(temperature=temp, model=model, api_key=st.secrets["OPENAI_API_KEY"])

async def determine_companies(message: str, agent_number: int) -> List[str]:
    llm = create_llm()
    prompt = PromptTemplate(
        input_variables=["message", "agent_number"],
        template=PROMPT_TEMPLATES["determine_companies"]
    )
    response = await asyncio.to_thread(LLMChain(llm=llm, prompt=prompt).run,
                                      message=message, agent_number=agent_number)
    return [c.strip() for c in response.split(",")][:agent_number]

async def generate_response(company: str, user_message: str, conversation_so_far: str, 
                           all_perspectives: List[str]) -> str:
    llm = create_llm(0.7)
    prompt = PromptTemplate(
        input_variables=["company", "user_message", "conversation_so_far", "all_perspectives"],
        template=PROMPT_TEMPLATES["generate_response"]
    )
    response = await asyncio.to_thread(LLMChain(llm=llm, prompt=prompt).run,
                                      company=company, user_message=user_message,
                                      conversation_so_far=conversation_so_far,
                                      all_perspectives=", ".join(all_perspectives))
    
    if json_match := re.search(r"```json\n(.*?)\n```", response, re.DOTALL):
        return await handle_tools(json_match.group(1), company, user_message, 
                                conversation_so_far, all_perspectives)
    return response.strip()

async def handle_tools(json_str: str, *args) -> str:
    try:
        tool_data = json.loads(json_str)
        handler = {
            "read": handle_read,
            "write": handle_write,
            "research": handle_research,
            "scrape_webpage": handle_scrape
        }[tool_data["tool"]]
        
        with st.spinner(f"Executing {tool_data['tool']} tool..."):
            return await handler(tool_data, *args)
            
    except (json.JSONDecodeError, KeyError) as e:
        return f"Tool error: {str(e)} in:\n{json_str}"

# ==============================================================================
# 4. Tool Handlers (Refactored)
# ==============================================================================
async def handle_read(data, company, user_message, conv, perspectives):
    content = await file_tools.read_tool(data["filename"])
    updated_conv = f"{conv}\n[File content]:\n{content}"
    return await regenerate_response(company, user_message, updated_conv, perspectives)

async def handle_write(data, *args):
    await file_tools.write_tool(data["filename"], data["content"])
    return f"File {data['filename']} updated"

async def handle_research(data, company, user_message, conv, perspectives):
    results = await research_tools.research_tool(data["query"])
    updated_conv = f"{conv}\n[Research]:\n{results}"
    return await regenerate_response(company, user_message, updated_conv, perspectives)

async def handle_scrape(data, *args):
    results = await research_tools.scrape_webpage_tool(data["url"])
    return f"Scraped content: {results.get('content', 'No data')}"

async def regenerate_response(company, user_message, conv, perspectives):
    return await generate_response(company, user_message, conv, perspectives)

# ==============================================================================
# 5. UI Components
# ==============================================================================
def main_layout():
    st.logo(LOGO_URL, link="https://biasbouncer.com")
    st.markdown("<h1 style='text-align: center;'><span style='color: red;'>BiasBouncer</span></h1>", 
               unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Team WorkBench</h3>", unsafe_allow_html=True)
    st.divider()
    
    cols = st.columns(4, gap="small")
    statuses = ["To Do", "In Progress", "In Review", "Done"]
    for col, status in zip(cols, statuses):
        with col:
            st.subheader(status)
            for task, details in st.session_state.tasks.items():
                if details["status"] == status:
                    render_task_button(task)

def render_task_button(task):
    @st.dialog(task)
    def task_dialog():
        st.write(f"Details for {task}")
        # Add actual task details here
    if st.button(task, use_container_width=True, type="primary"):
        task_dialog()

# ==============================================================================
# 6. Chat System
# ==============================================================================
def chat_sidebar():
    with st.sidebar:
        st.title("Brainstorm Chat")
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            st.chat_message(role).write(f"{msg['role']}: {msg['content']}")
        
        if user_input := st.chat_input("Work with the Agents"):
            process_user_input(user_input)

async def process_user_input(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    if not st.session_state.companies:
        st.session_state.companies = await determine_companies(
            user_input, st.sidebar.slider("Number of Agents", 2, 6, 4)
        )
    
    conversation = "\n".join(f"{m['role']}: {m['content']}" 
                            for m in st.session_state.chat_history)
    responses = await asyncio.gather(*[
        generate_response(c, user_input, conversation, st.session_state.companies) 
        for c in st.session_state.companies
    ])
    
    for company, response in zip(st.session_state.companies, responses):
        st.session_state.chat_history.append({"role": company, "content": response})

# ==============================================================================
# 7. Main Execution
# ==============================================================================
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OpenAI API key")
        st.stop()
        
    main_layout()
    chat_sidebar()