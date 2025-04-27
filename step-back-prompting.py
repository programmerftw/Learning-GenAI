from openai import OpenAI
import requests
from dotenv import load_dotenv
import json
import os

from create_and_store_vector_embeddings import create_and_store_vector_embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


""" Step 1 - User gives prompt """
user_query = input("Enter your prompt> ")


""" Step 2 - Create System prompt and generate step back prompt"""
system_prompt_generating_step_back_prompt = """
You are a helpful AI assistant which is master in creating step back prompt.

Rules-
1. Follow the strict JSON output as per output schema.
2. Abstract the key concepts and principles relevant to question.
3. Use the abstraction to reason through the question

Output Format-
{{
    "prompt": "string", 
}}

 

Example -
User prompt - Which is best framework to create REST apis in python?
Step back prompt - What is a framework, and which frameworks are available in Python?

User prompt - How to create REST apis in fastapi?
Step back prompt - What are REST APIs, and what steps are involved in creating REST APIs?

User prompt - How to do performance testing in locust?
Step back prompt - What is performance testing, and what are the requirements and steps involved in performance testing?

"""

client_generating_step_back_prompt = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

messages = [
    { "role": "system", "content": system_prompt_generating_step_back_prompt }
]

messages.append({ "role": "user", "content": user_query })

response = client_generating_step_back_prompt.chat.completions.create(
    model="models/gemini-1.5-flash-001",
    response_format={"type": "json_object"},
    messages=messages
)

llm_generated_step_back_prompt = json.loads(response.choices[0].message.content)

print("🧠 LLM Thinking...")
print("LLM created this step back prompt")
print(llm_generated_step_back_prompt)



""" Step 3. Create context using step back prompt."""
system_prompt_for_generating_context_using_step_back_prompt = """
You are an AI assistant which help to answer the question.
"""

client_for_generating_context_using_step_back_prompt = OpenAI(
    api_key=os.environ['API_KEY'],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response_context_generated_using_step_back_prompt = client_for_generating_context_using_step_back_prompt.chat.completions.create(
    model="models/gemini-1.5-flash-001",
    messages=[
        {"role": "system", "content": system_prompt_for_generating_context_using_step_back_prompt},
        {"role": "user", "content": llm_generated_step_back_prompt['prompt']}    
    ]
)


""" Step 4. Use resposne from step back prompt as a context for user query """

system_prompt_for_original_user_query = f"""
You are an AI assistant which help to answer the question based on given context.

Refer below context to give answers:
{response_context_generated_using_step_back_prompt.choices[0].message.content}

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

print("LLM Thinking...")
print("🧠: ", response_for_original_user_query.choices[0].message.content)