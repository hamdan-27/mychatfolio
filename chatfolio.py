from langchain_openai.chat_models import ChatOpenAI  # , AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

import uuid
import time
import random
from datetime import datetime

import streamlit as st
import pandas as pd

import agent


# Set page launch configurations
try:
    st.set_page_config(
        page_title="MyChatfolio.AI | Talk to CV", page_icon="📄",
        initial_sidebar_state='collapsed')

except Exception as e:
    st.toast(str(e))
    st.toast("Psst. Try refreshing the page.", icon="👀")


# __import__('pysqlite3')
# import sys

# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Rename msg type names for consistency
AIMessage.type = 'assistant'
HumanMessage.type = 'user'

# Add session state variables
if "prompt_ids" not in st.session_state:
    st.session_state["prompt_ids"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False


# VARIABLES
TEMPERATURE = 0.1
model = 'gpt-4'

llm = ChatOpenAI(temperature=TEMPERATURE,
                 model_name=model,
                 openai_api_key=st.secrets['api_key'])

# llm = AzureChatOpenAI(
#     model=model,
#     verbose=True,
#     temperature=TEMPERATURE,
#     openai_api_key = st.secrets["azure_key"],
#     openai_api_base="https://viewit-ai.openai.azure.com/",
#     deployment_name="Hamdan_16K",
#     openai_api_type="azure",
#     openai_api_version="2023-07-01-preview",
# )

spinner_texts = [
    '🧠 Thinking...',
    '📈 Performing Analysis...',
    '👾 Contacting the hivemind...',
    '🏠 Asking my neighbor...',
    '🍳 Preparing your answer...',
    '🏢 Getting nervous...',
    '👨 Pretending to be human...',
    '👽 Becoming sentient...',
    '🔍 Checking my notes...'
]

# API keys
# if type(llm) == ChatOpenAI:
#     openai.api_type = "open_ai"
#     openai.api_base = "https://api.openai.com/v1"
#     openai.api_key = st.secrets["api_key"]
#     openai.organization = st.secrets["org"]
#     openai.api_version = None

# if type(llm) == AzureChatOpenAI:
#     openai.api_type = "azure"
#     openai.api_base = "https://viewit-ai.openai.azure.com/"
#     openai.api_key = st.secrets["azure_key"]
#     openai.api_version = "2023-07-01-preview"



# APP INTERFACE START #

# Add logo image to the center of page
# col1, col2, col3 = st.columns(3)
# with col2:
#     st.image("https://i.postimg.cc/TwC7cjnL/19dd517c-f09d-48ae-9083-e10f5225e6d2.jpg", width=150)


# App Title

st.text('Have an interview with the CV, not the person!')


# AGENT CREATION HAPPENS HERE
agent = agent.create_rag_agent("data/resume.txt")


# App Sidebar
with st.sidebar:
    # st.write(st.session_state['button_question'])
    # st.write("session state msgs: ", st.session_state.messages)
    # st.write("StreamlitChatMessageHistory: ", msgs.messages)

    st.caption('''By using this chatbot, you agree that the chatbot is provided on 
            an "as is" basis and that we do not assume any liability for any 
            errors, omissions or other issues that may arise from your use of 
            the chatbot.''')


# Suggested questions
# questions = [
#     'Where do you see yourself in 5 years?',
#     'Introduce the interviewee.',
#     'Tell me about your skills.'
# ]


def send_button_ques(question):
    """Feeds the button question to the agent for execution.
    Args:
    - question: The text of the button
    Returns: None
    """
    st.session_state.disabled = True
    st.session_state['button_question'] = question


# Welcome message
welcome_msg = "Welcome to MyChatfolio, ask away!"
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": welcome_msg}]

feedback = None
# Render current messages from StreamlitChatMessageHistory
for n, msg in enumerate(st.session_state.messages):
    if msg["role"] == 'assistant':
        st.chat_message(msg["role"]).write(
            msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

    # # Render suggested question buttons
    # buttons = st.container(border=True)
    # if n == 0:
    #     for q in questions:
    #         button_ques = buttons.button(
    #             label=q, on_click=send_button_ques, args=[q],
    #             disabled=st.session_state.disabled
    #         )
    # else:
    #     st.session_state.disabled = True

    user_query = ""
    if msg["role"] == 'user':
        user_query = msg["content"]


# If user inputs a new prompt or clicks button, generate and draw a new response
if user_input := st.chat_input('Ask away'):# or st.session_state['button_question']:

    # Write user input
    st.session_state.messages.append(
        {"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Log user input to terminal
    user_log = f"\nUser [{datetime.now().strftime('%H:%M:%S')}]: " + \
        user_input
    print('='*90)
    print(user_log)

    # Note: new messages are saved to history automatically by Langchain during run
    with st.spinner(random.choice(spinner_texts)):
        # st.session_state.disabled = True

        try:
            response = agent.invoke({"input": user_input})["output"]

        # Handle the parsing error by omitting error from response
        except Exception as e:
            response = str(e)
            if response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix(
                    "Could not parse LLM output: `").removesuffix("`")
            st.toast(str(e), icon='⚠️')
            print(str(e))

    # Clear button question session state to prevent answer regeneration on rerun
    st.session_state['button_question'] = ""

    # Write AI response
    with st.chat_message("assistant"):
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split(' '):
            full_response += chunk + " "
            time.sleep(0.02)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Log AI response to terminal
    response_log = f"Bot [{datetime.now().strftime('%H:%M:%S')}]: " + \
        response
    print(response_log)
    # st.rerun()
