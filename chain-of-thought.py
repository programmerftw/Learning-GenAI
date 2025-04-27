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



""" Step 3. Chaining of prompts and responses"""
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=os.environ['API_KEY'],
    model="models/text-embedding-004"
)

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="python_programming_book",
    embedding=embedder
)

all_subquery_responses = ['']
last_prompt = ''

for subquery in llm_generated_prompts:
    chunks = retriver.similarity_search(
        query=subquery['prompt']
    )
    
    client_for_generating_response_for_subqueries = OpenAI(
        api_key=os.environ['API_KEY'],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    
    # This below system prompt will have relevant chunks + last query and its response.
    system_prompt_for_subqueries = f"""
    You are an AI assistant which help to answer the question based on given context.

    Refere below relevant chunks and query with its response  to give answers:
    Chunks -
    {chunks}
    
    Last Query and response -
    {last_prompt}
    {all_subquery_responses[-1]}

    """

    response_for_subquery = client_for_generating_response_for_subqueries.chat.completions.create(
        model="models/gemini-1.5-flash-001",
        messages=[
            {"role": "system", "content": system_prompt_for_subqueries},
            {"role": "user", "content": subquery['prompt']}    
        ]
    )
    
    all_subquery_responses.append(response_for_subquery.choices[0].message.content)
    last_prompt = subquery['prompt']
    print("\n---------------------------------------------------------------------")
    print("Subquery:", subquery['prompt'])
    print("Response:", response_for_subquery.choices[0].message.content)
    print("---------------------------------------------------------------------\n")
    


""" Step 4. Use relevant subquery responses to generate response for original user query """


system_prompt_for_original_user_query = f"""
You are an AI assistant which help to answer the question based on given context.

Refere below context to give answers:
{all_subquery_responses}

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