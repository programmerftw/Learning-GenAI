from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

import os
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.environ['API_KEY']


def create_and_store_vector_embedding(filename):
    
    file_path = filename
    print("file:", file_path)

    """ Step 1. Load PDF and divide into pages"""
    loader = PyPDFLoader(file_path)
    print("Loading...")

    docs = loader.load() # Divide pdf into pages


    """ Step 2. Create Chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(documents=docs)
    
    print("Chunks created...")

    """ Step 3. Create vector embedding of splitted text """

    embedder = GoogleGenerativeAIEmbeddings(
        google_api_key=API_KEY,
        model="models/text-embedding-004"
        )

    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url="http://localhost:6333",
        collection_name="python_programming_book",
        embedding=embedder
    )
    print("Embedding done...")

    vector_store.add_documents(documents=split_docs)
    print("Documents added")