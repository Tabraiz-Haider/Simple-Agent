# main.py

import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os

# ----------- Load Environment Variables -----------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

# ----------- Initialize Agent ----------------------
@st.cache_resource
def init_agent():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        openai_client=external_client,
        model="gemini-2.0-flash"
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    agent = Agent(
        name="Simple Agent",
        instructions="A simple agent that can answer questions."
    )

    return agent, config

agent, config = init_agent()

# ----------- Streamlit UI --------------------------
st.set_page_config(page_title="Gemini Agent", page_icon="🤖", layout="centered")
st.title("🤖 Gemini-Powered AI Agent")
st.markdown("Ask any question and get a response from **Gemini 2.0** via Agentic AI SDK.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", placeholder="Type your message and press Enter")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        result = Runner.run_sync(agent, user_input, run_config=config)
        agent_response = result.final_output

    # Save interaction
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Agent", agent_response))

# Chat history display
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Agent:** {message}")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# ----------- Branding Footer -----------------------
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style="text-align: center; font-size: 0.9rem; color: gray;">
        Made with ❤️ by <strong style="color: #4CAF50;">Tabraiz Haider</strong>
    </div>
    """,
    unsafe_allow_html=True
)
