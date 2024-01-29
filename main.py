import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
from datetime import datetime
import json


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def load_chunks(filename):
    with open(filename, 'r') as file:
        return json.load(file)



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with CBAHI :books:")
    user_question = st.text_input("Ask a question about your CBAHI Standards:")
    if user_question:
        handle_userinput(user_question)

    print("####### loaded_chunks started", datetime.now())
    loaded_chunks = load_chunks('chunks.json')
    print("####### loaded_chunks Finished", datetime.now())
    vectorstore = get_vectorstore(loaded_chunks)
    print("######### vectore chuncks finished ", datetime.now())
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = os.listdir('pdfs/')
        for file in pdf_docs:
            if file.endswith('.pdf'):
                st.write(f" * :page_facing_up: {file}")


if __name__ == '__main__':
    main()