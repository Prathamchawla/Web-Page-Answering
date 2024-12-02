import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit app interface
st.title('Webpage Q&A App')
st.write("Enter the URL of the webpage you'd like to process:")

# Input for webpage URL
url = st.text_input('Webpage URL')

# Create the Langchain objects
llm = Ollama(model="llama3.2")

# When a URL is provided
if url:
    # Display loading message
    st.write('Loading webpage...')
    loader = WebBaseLoader(url)

    # Fetch the webpage content
    documents = loader.load()
    st.write('Document loaded! Now splitting text...')

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(documents)
    st.write('Text split into chunks! Now generating embeddings...')

    # Generate embeddings
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstoredb = FAISS.from_documents(chunk_documents, embeddings)
    st.write('Embeddings generated and saved to vector store!')

    # Create prompt and document chain for querying
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant trained to provide detailed answers based on the provided context."),
        ("user", "{context}")
    ])
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vectorstoredb.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    # Allow user to ask questions
    st.write("Now, you can ask questions related to the content of the webpage!")
    question = st.text_input('Ask a question')

    if question:
        response = retriever_chain.invoke({"input": question})
        st.write("Answer: ")
        st.write(response['answer'])

