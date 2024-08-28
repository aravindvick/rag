import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
login(token=hugging_face_api_key)

# Set the title of the app
st.title("Chatgroq with Llama3 Demo")

# Define the LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Directory to save uploaded PDF files
save_dir = './docs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# File uploader for PDFs
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to the specified directory
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")

    # Perform vector embedding if a new file is uploaded
    if st.button("Document embeddings"):
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(save_dir)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        st.write("Vector DB is ready")

else:
    st.info("Please upload a PDF file.")

# Input field for user question
prompt1 = st.text_input("Enter your Question from Documents")

# Process the question if vector embeddings are ready
if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please initialize the vector embeddings first by clicking on 'Document embeddings' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, docs in enumerate(response['context']):
                st.write(docs.page_content)
                st.write("---------------------------------------")
