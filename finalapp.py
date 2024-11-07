import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Function to create embeddings and process documents
def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize embeddings and document loading processes
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        print(f"Number of documents loaded: {len(st.session_state.docs)}")  # Debugging line

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Limit to first 30 documents

        # Create the vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Store
        print(f"Vector Store Created with {len(st.session_state.vectors)} vectors.")  # Debugging line

# Streamlit UI setup
st.title("Nvidia NIM PDF Reader")

# Initialize the NVIDIA model
llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
""")

# Input field for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to start embedding documents
if st.button("Documents Embedding"):
    with st.spinner('Embedding documents...'):
        vector_embedding()
    st.write("Vector Store DB Is Ready")

# Handle user input and process the response
if prompt1:
    print("Starting retrieval process...")  # Debugging line

    # Set up document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Start timer for response time
    start = time.process_time()

    try:
        # Invoke the retrieval chain with user input
        response = retrieval_chain.invoke({'input': prompt1})
        print(f"Response: {response}")  # Debugging line

        # Display response time and answer
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Display relevant document content
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")

    except Exception as e:
        st.error(f"Error processing the query: {e}")
        print(f"Error: {e}")
