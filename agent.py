import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI

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


    prompt = hub.pull("hwchase17/openai-tools-agent")

    llm = ChatOpenAI(temperature=0)


    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
