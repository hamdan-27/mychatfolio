from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

import uuid
import time
import random
from datetime import datetime

import streamlit as st

import agent
import requests
import os
import time
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://client.camb.ai/apis"
API_KEY = os.getenv("CAMB_AI_KEY")
HEADERS = {"headers": {"x-api-key": API_KEY}}

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

# App Title
st.text('Have an interview with the CV instead of the person!')


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
questions = [
    'Where do you see yourself in 5 years?',
    'Briefly introduce yourself.',
    'Tell me about your skills.'
]


def send_button_ques(question):
    """Feeds the button question to the agent for execution.
    Args:
    - question: The text of the button
    Returns: None
    """
    st.session_state.disabled = True
    st.session_state['button_question'] = question


# Welcome message
welcome_msg = "Welcome to Voicefolio, ask away!"
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": welcome_msg}]

# Render current messages from StreamlitChatMessageHistory
for n, msg in enumerate(st.session_state.messages):
    if msg["role"] == 'assistant':
        st.chat_message(msg["role"]).write(
            msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

    # # Render suggested question buttons
    buttons = st.container(border=True)
    if n == 0:
        for q in questions:
            button_ques = buttons.button(
                label=q, on_click=send_button_ques, args=[q],
                disabled=st.session_state.disabled
            )
    else:
        st.session_state.disabled = True

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
            
            # Camb API call
            tts_payload = {
                "text": response,
                "voice_id": 8936,
                "language": 1,
                "gender": 1,
                "age": 21
            }

            res = requests.post(f"{BASE_URL}/tts", json=tts_payload, **HEADERS)
            task_id = res.json()["task_id"]
            print(f"Task ID: {task_id}")

            with st.spinner('Generating Audio...'):
                while True:
                    res = requests.get(f"{BASE_URL}/tts/{task_id}", **HEADERS)
                    status = res.json()["status"]
                    print(f"Polling: {status}")
                    time.sleep(1)
                    if status == "SUCCESS":
                        run_id = res.json()["run_id"]
                        break

                print(f"Run ID: {run_id}")
                res = requests.get(
                    f"{BASE_URL}/tts_result/{run_id}", **HEADERS, stream=True)
                st.audio(BytesIO(res.content), format='audio/wav')

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
