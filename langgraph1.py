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
    memory: List[Dict[str, Any]]  # Memory to store previous interactions
    summary: Dict[str, Any]  # Summary to store structured and natural language summaries

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
    print("Identified Schema:", identified_schema)
    state["schema_details"] = {
        "chain_of_thought": identified_schema.chain_of_thought,
        "relevant_collections": identified_schema.relevant_collections,
        "relevant_fields": identified_schema.relevant_fields,
    }
    # Update memory with the current question and schema details
    state["memory"].append({
        "question": state["question"],
        "schema_details": state["schema_details"],
    })
    # Update cost metrics
    cost_metrics["RAG Node"]["time"] = time.time() - start_time
    cost_metrics["RAG Node"]["api_calls"] += 1  # Assuming 1 API call per invocation
    cost_metrics["RAG Node"]["cost"] += 0.01  # Example cost per API call
    return state

def schema_node(state: State) -> State:
    """Schema Node: Retrieve collection schema."""
    start_time = time.time()
    db = client[state["db_name"]]
    print("Database Name:", state["db_name"])
    print("Schema Details:", state["schema_details"])
    relevant_collections = state["schema_details"].get("relevant_collections",[])
    print("Relevant Collection:", relevant_collections)
    if not relevant_collections:
        raise ValueError("No relevant collection identified. Cannot proceed with schema extraction.")


    # Process each relevant collection
    collection_schemas = {}
    for collection_name in relevant_collections:
        collection = db[collection_name]
        sample_document = collection.find_one()
        if sample_document:
            collection_schemas[collection_name] = {field: type(value).__name__ for field, value in sample_document.items()}
        else:
            collection_schemas[collection_name] = {}

    state["collection_schema"] = collection_schemas

    # Update cost metrics
    cost_metrics["Schema Node"]["time"] = time.time() - start_time
    cost_metrics["Schema Node"]["db_queries"] += 1  # Assuming 1 DB query per invocation
    cost_metrics["Schema Node"]["cost"] += 0.000  # Example cost per DB query
    return state

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def llm_node(state: State) -> State:
    """LLM Node: Generate MongoDB queries with memory context."""
    start_time = time.time()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel("gemini-1.5-flash")

    chain_of_thought = state["schema_details"]["chain_of_thought"]
    collection_schema = state["collection_schema"]
    query_type = state["query_type"]
    relevant_collections = state["schema_details"]["relevant_collections"]

    if not relevant_collections:
        raise ValueError("No relevant collections identified. Cannot proceed with query generation.")

    # Include memory in the prompt
    print("Memory Context========================================================:", state["memory"])
    memory_context = "\n".join([f"Q: {item['question']}\nA: {item.get('query_results', 'No results')}" for item in state["memory"]])

    # Generate a query for each collection
    mongo_queries = {}
    for collection_name in relevant_collections:
        query_prompt = f"""
        You are an expert MongoDB assistant. Based on the following chain of thought:
        {chain_of_thought}

        The schema of the collection '{collection_name}' is: {collection_schema[collection_name]}. Use these datatypes while generating the query.

        Previous interactions:
        {memory_context}

        Generate a MongoDB query for the following query type: {query_type}.
        Return only the MongoDB query in JSON format, without any explanations or additional text.
        """
        gemini_response = llm_model.generate_content(query_prompt)
        print(f"Gemini Response for {collection_name}:", gemini_response)

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
                mongo_queries[collection_name] = parsed_query
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse query for {collection_name}: {e}")
        else:
            raise ValueError(f"No candidates returned by the LLM for {collection_name}.")

    state["mongo_query"] = mongo_queries

    # Update memory with the generated queries
    state["memory"].append({
        "question": state["question"],
        "mongo_query": mongo_queries,
    })

    # Update cost metrics
    cost_metrics["LLM Node"]["time"] = time.time() - start_time
    cost_metrics["LLM Node"]["api_calls"] += len(relevant_collections)
    return state

def execution_node(state: State) -> State:
    """Execution Node: Execute MongoDB queries for multiple collections."""
    start_time = time.time()
    db = client[state["db_name"]]
    mongo_queries = state["mongo_query"]

    if not mongo_queries:
        raise ValueError("No MongoDB queries found to execute.")

    query_results = {}
    for collection_name, query in mongo_queries.items():
        print(f"Executing query for collection '{collection_name}': {query}")  # Log the query for debugging
        collection = db[query["collection"]]
        try:
            query_results[collection_name] = list(collection.aggregate(query["pipeline"]))
        except Exception as e:
            raise ValueError(f"Failed to execute query for collection '{collection_name}': {e}")

    state["query_results"] = query_results

    # Update memory with the query results
    state["memory"].append({
        "question": state["question"],
        "query_results": query_results,
    })

    # Update cost metrics
    cost_metrics["Execution Node"]["time"] = time.time() - start_time
    cost_metrics["Execution Node"]["db_queries"] += len(mongo_queries)  # One DB query per collection
    cost_metrics["Execution Node"]["cost"] += 0.000 * len(mongo_queries)  # Example cost per DB query
    return state
from bson import ObjectId
def summary_node(state: State) -> State:
    """Summary Node: Summarize the output from the MongoDB query using an LLM."""
    start_time = time.time()
    query_results = state["query_results"]

    if not query_results:
        raise ValueError("No query results found to summarize.")

    # Generate a summary based on the query results
    summary = {}
    for collection_name, results in query_results.items():
        summary[collection_name] = {
            "total_documents": len(results),
            "sample_document": results[0] if results else "No documents found",
        }

    # Use LLM to generate a natural language summary
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel("gemini-1.5-flash")

    # Prepare the context for the LLM
    memory_context = "\n".join([f"Q: {item['question']}\nA: {item.get('query_results', 'No results')}" for item in state["memory"]])
    print("Memory Context for LLM:", memory_context)
    chain_of_thought = state["schema_details"].get("chain_of_thought", "No chain of thought available.")
    print("Chain of Thought for LLM:", chain_of_thought)
  # Custom function to serialize ObjectId
    def serialize_mongo_objects(obj):
        if isinstance(obj, ObjectId):
            return str(obj)  # Convert ObjectId to string
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Serialize query results to JSON
    try:
        query_execution_response = json.dumps(query_results, indent=2, default=serialize_mongo_objects)
    except TypeError as e:
        raise ValueError(f"Failed to serialize query results: {e}")


    summary_prompt = f"""
    You are an expert assistant. Based on the following information, generate a concise and helpful summary in natural language to answer the user's query.
    **Remember that call completion is equal to sucess
    **User Query:**
    {state["question"]}

    **Chain of Thought:**
    {chain_of_thought}

    **Query Execution Response:**
    {query_execution_response}

    **Previous Interactions:**
    {memory_context}

    Provide a clear and concise summary that directly answers the user's query.
    """

    # Generate the summary using the LLM
    gemini_response = llm_model.generate_content(summary_prompt)
    print("LLM Summary Response:", gemini_response)

    if gemini_response.candidates:
        natural_language_summary = gemini_response.candidates[0].content.parts[0].text.strip()
    else:
        natural_language_summary = "Unable to generate a summary."

    # Update the state with the generated summary
    state["summary"] = {
        "structured_summary": summary,
        "natural_language_summary": natural_language_summary,
    }

    # Update cost metrics
    cost_metrics["Summary Node"] = {
        "time": time.time() - start_time,
        "cost": 0.000,  # Example cost for summarization
    }
    return state
# Build the LangGraph
builder = StateGraph(State)

# Add Nodes
builder.add_node("RAG Node", rag_node)
builder.add_node("Schema Node", schema_node)
builder.add_node("LLM Node", llm_node)
builder.add_node("Execution Node", execution_node)
builder.add_node("Summary Node", summary_node)  # Add the Summary Node

# Add Edges
builder.add_edge(START, "RAG Node")  # Entry point to the graph
builder.add_edge("RAG Node", "Schema Node")
builder.add_edge("Schema Node", "LLM Node")
builder.add_edge("LLM Node", "Execution Node")
builder.add_edge("LLM Node", "Execution Node")
builder.add_edge("Execution Node", "Summary Node")  # Connect Execution Node to Summary Node
builder.add_edge("Summary Node", END)  # Exit point from the graph# Compile the Graph

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
    "memory": [],  # Initialize memory as an empty list
}
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    initial_state["question"] = user_input
    final_state = graph.invoke(initial_state)

    print("Bot:", final_state["query_results"])
    print("Bot:", final_state["summary"]["natural_language_summary"])
# final_state = graph.invoke(initial_state)
# print("Final Results:", final_state["query_results"])

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
#         "relevant_collections": identified_schema.relevant_collections,
#         "relevant_fields": identified_schema.relevant_fields,
#     }
#     return state

# def schema_node(state: State) -> State:
#     """Schema Node: Retrieve collection schema."""
#     db = client[state["db_name"]]
#     collection = db[state["schema_details"]["relevant_collections"]]
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
#     collection_name = state["schema_details"]["relevant_collections"]

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
