import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from PyPDF2 import PdfReader
from huggingface_hub import login
import cassio
import os
import time
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field, BaseModel

class GeminiLLM(LLM, BaseModel):
    model_name: str= Field(default="gemini-1.5-flash", description='the name of the Gemini model')
    model: Optional[Any]= Field(None, description="The GenerativeModel instance")

    def __init__(self, model_name=str, **data):
        super().__init__(self, model_name=model_name, **data)
        self.model=genai.GenerativeModel(model_name=self.model_name)

    def _call(self, prompt:str, stop: Optional[list[str]]= None) -> str:
        system_prompt=(
                "You are a document-based Q&A assistant. Only answer questions based on the uploaded document. "
                "If the answer is not found in the document, respond with: 'The requested information is not available in the uploaded document. "
                 "Please ask something related to the document's content.' Do not use external knowledge or speculate."
         )
        full_prompt=f"{system_prompt}\n\nUser Query, {prompt}\n\nAnswer:"
        response=self.model.generate_content(full_prompt)
        return response.text
        
        @property
        def _llm_type(self)->str:
            return "gemini"
        
        @property
        def _identifying_params(self)->dict[str: Any]:
            return {"model_name": self.model_name}

load_dotenv(Path('.env'))

st.set_page_config(page_title="Ask to AARI-)", layout="wide")
st.title("Ask to AARI-)")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "faiss_vector_index" not in st.session_state:
    st.session_state.faiss_vector_index = None

def process_uploaded_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_reader= PdfReader(uploaded_file)
        raw_text= ''
        for page in pdf_reader.pages:
            content=page.extract_text()
            if content:
                raw_text += content
        genai.configure(api_key=st.secrets('GOOGLE_API_KEY'))

        embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        faiss_vector_store=FAISS.from_texts([raw_text], embedding_function)

        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size= 800,
            chunk_overlap= 200,
    )
        texts=text_splitter.split_text(raw_text)
        faiss_vector_store.add_texts(texts[:50])

        st.session_state.faiss_vector_index= VectorStoreIndexWrapper(vectorstore=faiss_vector_store)
        st.session_state.pdf_processed= True
        st.success("PDF processed and database intialized!")
    else:
        st.error("Failed to process the uploaded file.")

st.sidebar.markdown('## **Upload a Document**')
uploaded_file=st.sidebar.file_uploader("Upload a PDF document", type=['pdf'])
if uploaded_file:
    process_uploaded_pdf(uploaded_file)

st.sidebar.markdown("## **Welcome to AARI the Document mini chatbot**")
st.sidebar.markdown("##### **This chatbot is especially build for ASHU, for his learning practice")
st.sidebar.markdown(' If anything goes wrong do hard refresh by using **Shift** + **F5** key')

def typing_animation(text, speed):
    for char in text:
        yield char
        time.sleep(speed)

if 'intro_displayed' not in st.session_state:
    st.session_state.intro_displayed= True
    intro="Hi am AARI, ASHU Assistant"
    intro2="Ask to AARI"
    st.write_stream(typing_animation(intro, 0.02))
    st.write_stream(typing_animation(intro2, 0.02))

if 'messages' not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
     with st.chat_message(message['role']):
        st.markdown(message['content'])

if 'prePrompt_selected' not in st.session_state:
    st.session_state.prePrompt_selected= False

if 'btn_selected' not in st.session_state:
    st.session_state.btn_selected= True

def btn_callback():
    st.session_state.prePrompt_selected= False
    st.session_state.btn_selected= False

prePrompt= None

prompt= st.chat_input("Ask to AARI...")

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role":'user',"content": prompt})

    query_text=prompt.strip()
    gemini_llm=GeminiLLM(model_name='gemini-1.5-flash')
    if st.session_state.faiss_vector_index is not None:

        answer= st.session_state.faiss_vector_index.query(query_text, llm=gemini_llm).strip()

        typing_speed=0.02
        if 'context' in answer or 'no' in answer:
            with st.chat_message('assistant'):
                st.write_stream(typing_animation(answer, typing_speed))

        else:
            with st.chat_message('assistant'):
                st.write_stream(typing_animation(answer, typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")


    









    
