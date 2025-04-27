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


""" Step 2 - Create System prompt, and generate multiple user prompts"""
system_prompt_generating_multiple_prompts = """
You are a helpful AI assistant which help to create multiple user prompt based on given user prompt.

Rules-
1. Follow the strict JSON output as per output schema.
2. Always perform one step at at time and time and wait for next input.
3. Carefully analyze the user query.
4. Based on user query create maximum 3 to 5 prompts.

Output Format-
[
    {{
    "prompt": "string", 
    }},
    {{
    "prompt": "string", 
    }},
]

Example -
What is python programming?
Output : [
    {{"prompt": "what is python?"}},
    {{"prompt": "what is promgramming?"}},
    {{"prompt": "what is use of programming?}}
]

"""

client_for_generating_multiple_prompts = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

messages = [
    { "role": "system", "content": system_prompt_generating_multiple_prompts }
]

messages.append({ "role": "user", "content": user_query })

response = client_for_generating_multiple_prompts.chat.completions.create(
    model="models/gemini-1.5-flash-001",
    response_format={"type": "json_object"},
    messages=messages
)

llm_generated_prompts = json.loads(response.choices[0].message.content)

print("🧠 LLM Thinking...")
print("LLM created these prompts for user query")
print(llm_generated_prompts)



""" Step 3. Create vector embeddings of each query and perform similarity search with vector database."""
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=os.environ['API_KEY'],
    model="models/text-embedding-004"
)

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="python_programming_book",
    embedding=embedder
)

chunks_list = list()

print("Finding relevant chunks...")
total_chunks_counter = 0 # only for tracking total chunks
for prompt in llm_generated_prompts:
    
    relevant_chunks = retriver.similarity_search(
        query=prompt['prompt']
    )
    
    chunks_list.append(relevant_chunks)


""" Step 4. Ranking Relevant Chunks """
doc_scores = {}
k = 60 # k: A constant used for smoothing the reciprocal rank values

for ranking in chunks_list:
    for idx, doc in enumerate(ranking):
        doc_id = doc.metadata["_id"]
        rr = 1 / (idx + 1 + k)
        if doc_id in doc_scores:
            doc_scores[doc_id] += rr
        else:
            doc_scores[doc_id] = rr

fused_ranking = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

print("Length of fused ranking:", len(fused_ranking))


sorted_relevant_chunk_list = list()
for chunk_id, rank_score in doc_scores.items():
    
    for chunks in chunks_list:
        for chunk in chunks:
            if chunk_id in chunk.metadata["_id"]:
                sorted_relevant_chunk_list.append(chunk)
                


""" Step 5. Use relevant chunks to generate response for original user query """

print("Selecting only top 3 chunks...")
top_relevant_chunks = sorted_relevant_chunk_list[:3]

system_prompt_for_original_user_query = f"""
You are an AI assistant which help to answer the question based on given context.

Refere below context to give answers:
{top_relevant_chunks}

"""

client_for_generating_response_for_original_query = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response_for_original_user_query = client_for_generating_response_for_original_query.chat.completions.create(
    model="models/gemini-1.5-flash-001",
    messages=[
        {"role": "system", "content": system_prompt_for_original_user_query},
        {"role": "user", "content": user_query}    
    ]
)

print("🧠 LLM Thinking...")
print(response_for_original_user_query.choices[0].message.content)