from pymongo import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st
import asyncio


load_dotenv()


uri = os.getenv("uri")
print(uri)

try:
    client = MongoClient(uri)
    client.admin.command('ping')  
    mongo_connected = True
    st.success("Connected to MongoDB!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    mongo_connected = False


print("Databases:", client.list_database_names())

from pymongo.mongo_client import MongoClient
import openai
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd
from dotenv import load_dotenv
import os
from textwrap import dedent
from agno.tools.thinking import ThinkingTools
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.pandas import PandasTools
from agno.models.groq import Groq
from tools.tools import MongoDBUtility
import requests
from tools.rag import RagToolkit
import json
from decimal import Decimal, getcontext
import google.generativeai as genai


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


import json
from decimal import Decimal, getcontext

from pydantic import BaseModel, ValidationError
from typing import Dict, Any

class MongoQuery(BaseModel):
    match: Dict[str, Any] 


def process_question_with_rag_and_gemini(question: str):
    try:
        # Step 1: Send the question to the RAG tool
        rag_tool = RagToolkit()
        rag_response = rag_tool.ragcot(question)
        print("RAG Response:", type(rag_response))
        print("RAG Response:", (rag_response))

        # Step 2: Extract the chain of thought and relevant details from the Pydantic object
        identified_schema = rag_response.identified_schema  # Assuming ragcot returns a QueryResponse object
        chain_of_thought = identified_schema.chain_of_thought
        relevant_collection = identified_schema.relevant_collection
        relevant_fields = identified_schema.relevant_fields
        print("chain_of_thought:", chain_of_thought)
        print("relevant_collection:", relevant_collection)
        print("relevant_fields:", relevant_fields)
 
        # Step 3: Use Gemini LLM to generate the query
        genai.configure(api_key=GOOGLE_API_KEY)

        llm_model = genai.GenerativeModel("gemini-1.5-flash")

        # gemini_model = Groq(id="gemini-1.5-flash", temperature=0)
        query_prompt = f"""
        Generate a MongoDB query based on the following chain of thought:
        {chain_of_thought}

        Return only the MongoDB query in JSON format, without any explanations or additional text.
        """
        gemini_response = llm_model.generate_content(query_prompt)
        print("Gemini Response Content:", gemini_response)

        # Extract the content from the first candidate
        if gemini_response.candidates:
            raw_query_text = gemini_response.candidates[0].content.parts[0].text
        else:
            print("No candidates returned by Gemini LLM.")
            return

        print("Raw Query Text:", raw_query_text)
        # Step 4: Clean the raw query text
        cleaned_query_text = raw_query_text.strip("``[json").strip("](http://_vscodecontentref_/1)``").strip()
        print("Cleaned Query Text:", cleaned_query_text)
        # Step 4: Parse the generated query as JSON
        try:
            mongo_query = json.loads(cleaned_query_text)  # Parse the raw query text as JSON
            if not isinstance(mongo_query, dict):
                raise ValueError("The generated query is not a valid MongoDB query.")
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response as JSON: {e}")
            print(f"Raw Gemini Response: {raw_query_text}")
            return

        print(f"Validated MongoDB Query: {mongo_query}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error: {e}")

prompt = "List all failed calls in the last 7 days."
process_question_with_rag_and_gemini(prompt)