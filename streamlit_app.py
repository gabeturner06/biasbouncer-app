import asyncio
import streamlit as st
from typing import List, Dict

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ------------------------------------------------------------------------------
# 1. Configure page layout
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------
# 2. Access API Key from Streamlit secrets
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
    # {"role": "user" OR "<company_name>", "content": "..."}
    st.session_state["chat_history"] = []

if "companies" not in st.session_state:
    # Dynamically generated list of 'perspectives'
    st.session_state["companies"] = []

# ------------------------------------------------------------------------------
# 3. Vector store (if needed) and base LLM
# ------------------------------------------------------------------------------
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db_oai"
)

llm = ChatOpenAI(temperature=0)  # Base LLM (not used directly below but you can adapt)

# ------------------------------------------------------------------------------
# 4. Functions from the multi-agent approach
# ------------------------------------------------------------------------------

async def determine_companies(message: str, agent_number: int) -> List[str]:
    """
    Uses an LLM to analyze the user query and determine up to {agent_number} relevant perspectives.
    """
    llm_instance = ChatOpenAI(temperature=0, model="gpt-4")

    template = f"""
    Based on this user query, identify a list of up to {agent_number} diverse perspectives or roles
    that could respond to the problem with different solutions. Or, if they ask the to argue over 
    specified perspectives or items (e.g., if they ask you to argue over what item is best over a set 
    of items, each agent should advocate for one of the items), use their specified items as the 
    perspectives (e.g., if the user asks for a debate over which "x" is best, each different "x" should 
    be a perspective). Return them as comma-separated values.

    User query: {message}
    """
    prompt = PromptTemplate(input_variables=["message", "agent_number"], template=template)
    chain = LLMChain(llm=llm_instance, prompt=prompt)

    response = await asyncio.to_thread(chain.run, message=message, agent_number=agent_number)
    companies = [item.strip() for item in response.split(",") if item.strip()]
    print(companies)
    return companies[:agent_number]

async def generate_response(
    company: str,
    user_message: str,
    conversation_so_far: str,
    other_agents_responses: str
) -> str:
    """
    Generates a short, informal, brainstorming-style response from the perspective of `company`.
    """
    llm_instance = ChatOpenAI(temperature=0.7, model="gpt-4")
    template = """
    You're in a casual group brainstorming chat representing or in support of the perspective: {company}.
    The user just said: {user_message}
    Entire conversation so far:
    {conversation_so_far}

    Other participants' latest responses:
    {other_agents_responses}

    Please reply briefly and informally, as if you're a professional brainstorming with friends in a group 
    chat. It is meant to be  quick, collaborative brainstorm session with the user, where you create a single 
    idea for a feature or solution and briefly explain it as if it just "popped into your head." In other words, 
    your response shouldn't be much longer than the question asked by the user.
    """
    prompt = PromptTemplate(
        input_variables=["company", "user_message", "conversation_so_far", "other_agents_responses"],
        template=template
    )

    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = await asyncio.to_thread(
        chain.run,
        company=company,
        user_message=user_message,
        conversation_so_far=conversation_so_far,
        other_agents_responses=other_agents_responses
    )
    return response.strip()

async def run_agents(
    companies: List[str],
    user_message: str,
    conversation: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    Launch all perspectives concurrently. Each agent sees the entire conversation
    plus the most recent user message.
    """
    # Convert the entire conversation into a string
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in conversation
    )

    # For simplicity, let's just feed the full conversation as "other_agents_responses" too.
    # You could refine this if you want only the last round of messages.
    other_agents_text = conversation_text

    tasks = []
    for company in companies:
        tasks.append(generate_response(
            company,
            user_message,
            conversation_text,
            other_agents_text
        ))
    results = await asyncio.gather(*tasks)
    return dict(zip(companies, results))

# ------------------------------------------------------------------------------
# 5. Main Page Layout (Preserve old style)
# ------------------------------------------------------------------------------
LOGO_URL_LARGE = "/Users/gabrielryanturner/Documents/Source/archive/biasbouncer-auto-gpt/backup/biasbouncer-logo.png"

st.logo(
    image=LOGO_URL_LARGE,
    link="https://biasbouncer.com",
    size="large"
)

st.markdown("<h1 style='text-align: center;'><span style='color: white;'>Bias</span><span style='color: red;'>Bouncer</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Team WorkBench</h3>", unsafe_allow_html=True)

st.divider()

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

# ------------------------------------------------------------------------------
# 6. Sidebar Chat
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