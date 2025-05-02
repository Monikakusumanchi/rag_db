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

# prompt = "List all failed calls in the last 7 days."
# process_question_with_rag_and_gemini(prompt)
from agno.agent import Agent
from agno.tools import Toolkit
from agno.models.google import Gemini
from pymongo import MongoClient
import os
import time
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB connection
uri = os.getenv("uri")
client = MongoClient(uri)

# Define Tools
class RagToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="rag_toolkit")

    def function_declarations(self):
        return [
            {
                "name": "ragcot",
                "parameters": {
                    "properties": [
                        {"name": "query", "type": "string"}
                    ]
                }
            }
        ]

    def ragcot(self, query: str):
        """Simulate RAG Node: Retrieve schema details."""

        rag_toolkit = RagToolkit()
        rag_response = rag_toolkit.ragcot(query)
        identified_schema = rag_response.identified_schema
        response= {
            "chain_of_thought": identified_schema.chain_of_thought,
            "relevant_collection": identified_schema.relevant_collection,
            "relevant_fields": identified_schema.relevant_fields,
        }
        cost_metrics["RAG Node"]["time"] = time.time() - start_time
        cost_metrics["RAG Node"]["api_calls"] += 1
        cost_metrics["RAG Node"]["cost"] += 0.01
        return response

class SchemaToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="schema_toolkit")

    def function_declarations(self):
        return [
            {
                "name": "get_collection_schema",
                "parameters": {
                    "properties": [
                        {"name": "collection_name", "type": "string"}
                    ]
                }
            }
        ]

    def get_collection_schema(self, collection_name: str):
        """Simulate Schema Node: Retrieve collection schema."""
        start_time = time.time()
        db = client["Demo"]
        collection = db[collection_name]
        sample_document = collection.find_one()
        schema = {field: type(value).__name__ for field, value in sample_document.items()} if sample_document else {}
        cost_metrics["Schema Node"]["time"] = time.time() - start_time
        cost_metrics["Schema Node"]["db_queries"] += 1
        cost_metrics["Schema Node"]["cost"] += 0.000
        return schema

class LLMToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="llm_toolkit")

    def function_declarations(self):
        return [
            {
                "name": "generate_query",
                "parameters": {
                    "properties": [
                        {"name": "chain_of_thought", "type": "string"},
                        {"name": "collection_schema", "type": "object"},
                        {"name": "query_type", "type": "string"}
                    ]
                }
            }
        ]

    def generate_query(self, chain_of_thought: str, collection_schema: dict, query_type: str):
        """Simulate LLM Node: Generate MongoDB query."""
        start_time = time.time()
        # Simulate LLM response
        query = {
            "collection": "calls",
            "pipeline": [
                {"$group": {"_id": None, "average_duration": {"$avg": "$duration_minutes"}}}
            ]
        }
        cost_metrics["LLM Node"]["time"] = time.time() - start_time
        cost_metrics["LLM Node"]["api_calls"] += 1
        cost_metrics["LLM Node"]["input_cost"] += 0.00015
        cost_metrics["LLM Node"]["output_cost"] += 0.00060
        cost_metrics["LLM Node"]["total_cost"] += 0.00075
        return query

class ExecutionToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="execution_toolkit")

    def function_declarations(self):
        return [
            {
                "name": "execute_query",
                "parameters": {
                    "properties": [
                        {"name": "collection", "type": "string"},
                        {"name": "pipeline", "type": "array"}
                    ]
                }
            }
        ]

    def execute_query(self, collection: str, pipeline: list):
        """Simulate Execution Node: Execute the MongoDB query."""
        start_time = time.time()
        db = client["Demo"]
        collection = db[collection]
        results = list(collection.aggregate(pipeline))
        cost_metrics["Execution Node"]["time"] = time.time() - start_time
        cost_metrics["Execution Node"]["db_queries"] += 1
        cost_metrics["Execution Node"]["cost"] += 0.000
        return results

# Define Cost Metrics
cost_metrics = {
    "RAG Node": {"time": 0, "api_calls": 0, "cost": 0},
    "Schema Node": {"time": 0, "db_queries": 0, "cost": 0},
    "LLM Node": {"time": 0, "api_calls": 0, "input_cost": 0, "output_cost": 0, "total_cost": 0},
    "Execution Node": {"time": 0, "db_queries": 0, "cost": 0},
}

# Create Agent
agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[
        RagToolkit(),
        SchemaToolkit(),
        LLMToolkit(),
        ExecutionToolkit()
    ],
    instructions="""
    You are an intelligent MongoDB assistant. Follow these steps:
    1. Use the RAG Toolkit to retrieve schema details.
    2. Use the Schema Toolkit to get the collection schema.
    3. Use the LLM Toolkit to generate a MongoDB query.
    4. Use the Execution Toolkit to execute the query and return results.
    """,
    show_tool_calls=True,
    markdown=True,
)

# Run the Agent
response = agent.run("What is the average call duration of the calls made by the agents?")
print("Agent Response:", response)

# Print Cost Metrics
print("\nCost Metrics:")
for node, metrics in cost_metrics.items():
    print(f"{node}: {metrics}")

# Calculate Total Cost
total_cost = sum(metrics.get("cost", 0) + metrics.get("total_cost", 0) for metrics in cost_metrics.values())
print(f"\nTotal Cost of Execution: ${total_cost:.6f}")