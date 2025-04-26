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


file_path = './filename.pdf'

""" Step 1. Load PDF and divide into pages"""
loader = PyPDFLoader(file_path)
# print(loader)

docs = loader.load() # Divide pdf into pages

# print(docs[5])


""" Step 2. Create Chunks"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents=docs)
print("Splitted docs: ",len(split_docs))

""" Step 3. Create vector embedding of splitted text """

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=API_KEY,
    model="models/text-embedding-004"
    )

# print(embedder)
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="pdf_rag_agent",
#     embedding=embedder
# )

# vector_store.add_documents(documents=split_docs)

# print("Injection done")

""" Step 4. Take input/query from user """
query = input('what would you like to know> ')

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="pdf_rag_agent",
    embedding=embedder
)

search_result = retriver.similarity_search(
    query=query
)

print("Relevant Chunks", search_result)


system_prompt = f"""
You are an AI assistant which help to answer the question based on given context.


Refere below context to give answers:
{search_result}
"""

messages = [
    { "role": "system", "content": system_prompt }
]


client = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


messages.append({ "role": "user", "content": query })

response = client.chat.completions.create(
            model="models/gemini-1.5-flash-001",
            response_format={"type": "json_object"},
            messages=messages
        )
parsed_output = json.loads(response.choices[0].message.content)

print("LLM Thinking...")
print("🧠", parsed_output)