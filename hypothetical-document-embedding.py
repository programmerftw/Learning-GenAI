from openai import OpenAI
import requests
from dotenv import load_dotenv
import json
import os

from create_and_store_vector_embeddings import create_and_store_vector_embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

""" Step 0 - Create and store vector embeddings of pdf document"""
# create_and_store_vector_embedding(os.getcwd()+"/python_programming_book.pdf")
# print("Injection done..")


""" Step 1 - User gives prompt """
user_query = input("Enter your prompt> ")


""" Step 2 - Create System prompt, and generate relevant response"""
system_prompt_generating_relevant_response = """
You are a helpful AI assistant which provide concise answer for user prompts.

Rules-
1. Follow the strict JSON output as per output schema.
2. Always perform one step at at time and time and wait for next input.
3. Carefully analyze the user query.

Output Format-
{{
    "output": "string", 
}},

"""

client_for_generating_relevant_response = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

messages = [
    { "role": "system", "content": system_prompt_generating_relevant_response }
]

messages.append({ "role": "user", "content": user_query })

response = client_for_generating_relevant_response.chat.completions.create(
    model="models/gemini-1.5-flash-001",
    response_format={"type": "json_object"},
    messages=messages
)

llm_generated_response = json.loads(response.choices[0].message.content)

print("🧠 LLM Thinking...")
print(llm_generated_response)



""" Step 3. Create vector embeddings of relevant response and perform similarity search with vector database."""
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=os.environ['API_KEY'],
    model="models/text-embedding-004"
)

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="python_programming_book",
    embedding=embedder
)


relevant_chunks = retriver.similarity_search(
    query=llm_generated_response['output']
)



""" Step 4. Using relevant chunks we can find out page references """
for chunk in relevant_chunks:
    print("-------------------------------------")
    print("On page no.", chunk.metadata['page'])
    print("Page content:", chunk.page_content)
    print("-------------------------------------")