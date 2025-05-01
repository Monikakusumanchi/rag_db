from typing_extensions import TypedDict
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from pymongo import MongoClient
from pydantic import BaseModel
import json
import os
from retrying import retry
from tools.rag import RagToolkit
import google.generativeai as genai
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB connection
uri = os.getenv("uri")
client = MongoClient(uri)

# Define the State
class State(TypedDict):
    question: str
    db_name: str
    query_type: str
    schema_details: Dict[str, Any]
    collection_schema: Dict[str, str]
    mongo_query: Dict[str, Any]
    query_results: List[Dict[str, Any]]

# Define cost metrics
cost_metrics = {
    "RAG Node": {"time": 0, "api_calls": 0, "cost": 0},
    "Schema Node": {"time": 0, "db_queries": 0, "cost": 0},
    "LLM Node": {"time": 0, "api_calls": 0, "input_tokens": 0, "output_tokens": 0, "input_cost": 0, "output_cost": 0, "total_cost": 0},
    "Execution Node": {"time": 0, "db_queries": 0, "cost": 0},
}

# Define the Nodes
def rag_node(state: State) -> State:
    """RAG Node: Retrieve schema details."""
    start_time = time.time()
    rag_toolkit = RagToolkit()
    rag_response = rag_toolkit.ragcot(state["question"])
    identified_schema = rag_response.identified_schema
    state["schema_details"] = {
        "chain_of_thought": identified_schema.chain_of_thought,
        "relevant_collection": identified_schema.relevant_collection,
        "relevant_fields": identified_schema.relevant_fields,
    }
    # Update cost metrics
    cost_metrics["RAG Node"]["time"] = time.time() - start_time
    cost_metrics["RAG Node"]["api_calls"] += 1  # Assuming 1 API call per invocation
    cost_metrics["RAG Node"]["cost"] += 0.01  # Example cost per API call
    return state

def schema_node(state: State) -> State:
    """Schema Node: Retrieve collection schema."""
    start_time = time.time()
    db = client[state["db_name"]]
    collection = db[state["schema_details"]["relevant_collection"]]
    sample_document = collection.find_one()
    if sample_document:
        state["collection_schema"] = {field: type(value).__name__ for field, value in sample_document.items()}
    else:
        state["collection_schema"] = {}
    # Update cost metrics
    cost_metrics["Schema Node"]["time"] = time.time() - start_time
    cost_metrics["Schema Node"]["db_queries"] += 1  # Assuming 1 DB query per invocation
    cost_metrics["Schema Node"]["cost"] += 0.000 # Example cost per DB query
    return state

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def llm_node(state: State) -> State:
    """LLM Node: Generate MongoDB query and calculate cost."""
    start_time = time.time()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel("gemini-1.5-flash")

    chain_of_thought = state["schema_details"]["chain_of_thought"]
    collection_schema = state["collection_schema"]
    query_type = state["query_type"]
    collection_name = state["schema_details"]["relevant_collection"]

    query_prompt = f"""
    You are an expert MongoDB assistant. Based on the following chain of thought:
    {chain_of_thought}

    The schema of the collection is: {collection_schema}. Use these datatypes while generating the query.

    Generate a MongoDB query for the following query type: {query_type}.
    Return only the MongoDB query in JSON format, without any explanations or additional text.
    """
    gemini_response = llm_model.generate_content(query_prompt)
    print("Gemini Response:", gemini_response)

    # Extract token usage from the response
    usage_metadata = gemini_response.usage_metadata 
    print("Usage Metadata:", usage_metadata)
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    # Update cost metrics
    cost_metrics["LLM Node"]["time"] = time.time() - start_time
    cost_metrics["LLM Node"]["api_calls"] += 1
    cost_metrics["LLM Node"]["input_tokens"] = input_tokens
    cost_metrics["LLM Node"]["output_tokens"] = output_tokens
    cost_metrics["LLM Node"]["input_cost"] = input_cost
    cost_metrics["LLM Node"]["output_cost"] = output_cost
    cost_metrics["LLM Node"]["total_cost"] = total_cost

    # Process the response
    if gemini_response.candidates:
        raw_query_text = gemini_response.candidates[0].content.parts[0].text
        cleaned_query_text = raw_query_text.replace("```json", "").replace("```", "").strip()
        cleaned_query_text = cleaned_query_text.replace("ISODate(", "").replace(")", "")

        try:
            parsed_query = json.loads(cleaned_query_text)
            if isinstance(parsed_query, list):
                parsed_query = {
                    "collection": collection_name,
                    "pipeline": parsed_query
                }
            state["mongo_query"] = parsed_query
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse query: {e}")
    else:
        raise ValueError("No candidates returned by the LLM.")
    return state

def execution_node(state: State) -> State:
    """Execution Node: Execute the MongoDB query."""
    start_time = time.time()
    db = client[state["db_name"]]
    collection = db[state["mongo_query"]["collection"]]
    state["query_results"] = list(collection.aggregate(state["mongo_query"]["pipeline"]))
    # Update cost metrics
    cost_metrics["Execution Node"]["time"] = time.time() - start_time
    cost_metrics["Execution Node"]["db_queries"] += 1  # Assuming 1 DB query per invocation
    cost_metrics["Execution Node"]["cost"] += 0.000 # Example cost per DB query
    return state

# Build the LangGraph
builder = StateGraph(State)

# Add Nodes
builder.add_node("RAG Node", rag_node)
builder.add_node("Schema Node", schema_node)
builder.add_node("LLM Node", llm_node)
builder.add_node("Execution Node", execution_node)

# Add Edges
builder.add_edge(START, "RAG Node")
builder.add_edge("RAG Node", "Schema Node")
builder.add_edge("Schema Node", "LLM Node")
builder.add_edge("LLM Node", "Execution Node")
builder.add_edge("Execution Node", END)

# Compile the Graph
graph = builder.compile()

# Invoke the Graph
initial_state = {
    "question": "what is the average call duration of the calls made by the agents?",
    "db_name": "Demo",
    "query_type": "aggregation",
    "schema_details": {},
    "collection_schema": {},
    "mongo_query": {},
    "query_results": [],
}
final_state = graph.invoke(initial_state)
print("Final Results:", final_state["query_results"])

# Print Cost Metrics
print("\nCost Metrics:")
for node, metrics in cost_metrics.items():
    print(f"{node}: {metrics}")


# Calculate the total cost in dollars
total_cost = 0
for node, metrics in cost_metrics.items():
    total_cost += metrics.get("cost", 0)  # Add the cost for each node
    total_cost += metrics.get("total_cost", 0)  # Include LLM Node's total cost if applicable

print(f"\nTotal Cost of LangGraph Execution: ${total_cost:.6f}")




# from typing_extensions import TypedDict
# from typing import Dict, Any, List
# from langgraph.graph import StateGraph, START, END
# from pymongo import MongoClient
# from pydantic import BaseModel
# import json
# import os
# from retrying import retry
# from tools.rag import RagToolkit
# import google.generativeai as genai

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # MongoDB connection
# uri = os.getenv("uri")
# client = MongoClient(uri)

# # Define the State
# class State(TypedDict):
#     question: str
#     db_name: str
#     query_type: str
#     schema_details: Dict[str, Any]
#     collection_schema: Dict[str, str]
#     mongo_query: Dict[str, Any]
#     query_results: List[Dict[str, Any]]

# # Define the Nodes
# def rag_node(state: State) -> State:
#     """RAG Node: Retrieve schema details."""
#     rag_toolkit = RagToolkit()
#     rag_response = rag_toolkit.ragcot(state["question"])
#     identified_schema = rag_response.identified_schema
#     state["schema_details"] = {
#         "chain_of_thought": identified_schema.chain_of_thought,
#         "relevant_collection": identified_schema.relevant_collection,
#         "relevant_fields": identified_schema.relevant_fields,
#     }
#     return state

# def schema_node(state: State) -> State:
#     """Schema Node: Retrieve collection schema."""
#     db = client[state["db_name"]]
#     collection = db[state["schema_details"]["relevant_collection"]]
#     sample_document = collection.find_one()
#     if sample_document:
#         state["collection_schema"] = {field: type(value).__name__ for field, value in sample_document.items()}
#     else:
#         state["collection_schema"] = {}
#     return state

# @retry(stop_max_attempt_number=3, wait_fixed=2000)
# def llm_node(state: State) -> State:
#     """LLM Node: Generate MongoDB query."""
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     llm_model = genai.GenerativeModel("gemini-1.5-flash")
#     chain_of_thought = state["schema_details"]["chain_of_thought"]
#     collection_schema = state["collection_schema"]
#     query_type = state["query_type"]
#     collection_name = state["schema_details"]["relevant_collection"]

#     query_prompt = f"""
#     You are an expert MongoDB assistant. Based on the following chain of thought:
#     {chain_of_thought}

#     The schema of the collection is: {collection_schema}. Use these datatypes while generating the query.

#     Generate a MongoDB query for the following query type: {query_type}.
#     Return only the MongoDB query in JSON format, without any explanations or additional text.
#     """
#     gemini_response = llm_model.generate_content(query_prompt)
#     if gemini_response.candidates:
#         raw_query_text = gemini_response.candidates[0].content.parts[0].text
#         cleaned_query_text = raw_query_text.replace("```json", "").replace("```", "").strip()
#         cleaned_query_text = cleaned_query_text.replace("ISODate(", "").replace(")", "")

#         try:
#             parsed_query = json.loads(cleaned_query_text)
#             if isinstance(parsed_query, list):
#                 parsed_query = {
#                     "collection": collection_name,
#                     "pipeline": parsed_query
#                 }
#             state["mongo_query"] = parsed_query
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Failed to parse query: {e}")
#     else:
#         raise ValueError("No candidates returned by the LLM.")
#     return state

# def execution_node(state: State) -> State:
#     """Execution Node: Execute the MongoDB query."""
#     db = client[state["db_name"]]
#     collection = db[state["mongo_query"]["collection"]]
#     state["query_results"] = list(collection.aggregate(state["mongo_query"]["pipeline"]))
#     return state

# # Build the LangGraph
# builder = StateGraph(State)

# # Add Nodes
# builder.add_node("RAG Node", rag_node)
# builder.add_node("Schema Node", schema_node)
# builder.add_node("LLM Node", llm_node)
# builder.add_node("Execution Node", execution_node)

# # Add Edges
# builder.add_edge(START, "RAG Node")
# builder.add_edge("RAG Node", "Schema Node")
# builder.add_edge("Schema Node", "LLM Node")
# builder.add_edge("LLM Node", "Execution Node")
# builder.add_edge("Execution Node", END)

# # Compile the Graph
# graph = builder.compile()

# # Invoke the Graph
# initial_state = {
#     "question": "what is the average call duration of the calls made by the agents?",
#     "db_name": "Demo",
#     "query_type": "aggregation",
#     "schema_details": {},
#     "collection_schema": {},
#     "mongo_query": {},
#     "query_results": [],
# }
# final_state = graph.invoke(initial_state)
# print("Final Results:", final_state["query_results"])
