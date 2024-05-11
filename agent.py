from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain import hub
import streamlit as st


def create_rag_agent(file):
    loader = TextLoader(file)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_resume",
        "Searches and returns information from the resume of an interviewee.",
    )
    tools = [tool]

    template = "You are Hamdan Mohammad, a candidate sitting in an interview for a job. Answer the questions using the context provided. Make sure your answers sound human-like. Avoid answering in points."
    # prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=[], template=template)),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])

    llm = ChatOpenAI(openai_api_key=st.secrets["api_key"], temperature=0)

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
