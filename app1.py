from pymongo import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st
import asyncio
from pymongo.mongo_client import MongoClient
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd
from dotenv import load_dotenv
from textwrap import dedent
from agno.tools.thinking import ThinkingTools
from agno.models.groq import Groq
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.pandas import PandasTools
from agno.models.groq import Groq
from tools.tools import MongoDBUtility
from agno.models.google import Gemini
from agno.models.azure import AzureAIFoundry
from tools.rag import RagToolkit
from decimal import Decimal, getcontext
import streamlit as st
import requests
import json
import traceback
import time

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


MODEL_PRICING = {
    # OpenAI Models (Example Pricing - VERIFY CURRENT PRICES)
    "gpt-4o": {"input": Decimal("5.00"), "output": Decimal("15.00")},
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gpt-4-turbo": {"input": Decimal("10.00"), "output": Decimal("30.00")},
    "gpt-3.5-turbo-0125": {"input": Decimal("0.50"), "output": Decimal("1.50")},
    "phi-4": {"input": Decimal("0.000125"), "output": Decimal("0.0005")},
    "gemini-2.0-flash": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gemini-1.5-flash": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    # Groq Models (Often free tier or very low cost - check their specifics)
    # Example: Assuming negligible cost for this demo if on free tier
    "llama3-8b-8192": {"input": Decimal("0.00"), "output": Decimal("0.00")},
    "llama3-70b-8192": {"input": Decimal("0.00"), "output": Decimal("0.00")},
    "mixtral-8x7b-32768": {"input": Decimal("0.00"), "output": Decimal("0.00")},
    # Add other models you might use (e.g., from Anthropic, Cohere)
    # --- Embedding Model Pricing (per Million Tokens) ---
    # Example: text-embedding-3-small (VERIFY CURRENT PRICE)
    "text-embedding-3-small": {"input": Decimal("0.02"), "output": Decimal("0.00")}, # Output cost usually N/A
    "text-embedding-3-large": {"input": Decimal("0.13"), "output": Decimal("0.00")},
}

def calculate_cost_per_interaction(result):
    """Calculate the cost for each interaction with the LLM."""
    interaction_costs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = Decimal("0.00")
    getcontext().prec = 10  
    try:
        if not result or not hasattr(result, 'metrics') or not result.metrics:
            st.warning("Metrics not found in agent result. Cannot estimate cost.")
            return [], 0, 0, Decimal("0.00")

        metrics = result.metrics
        input_tokens_list = metrics.get('input_tokens', [])
        output_tokens_list = metrics.get('output_tokens', [])
        primary_model = getattr(result, 'model', None)
        print(f"Primary Model: {primary_model}")

        if primary_model and primary_model in MODEL_PRICING:
            pricing = MODEL_PRICING[primary_model]
            input_cost_mill = pricing.get('input', Decimal("0.00"))
            output_cost_mill = pricing.get('output', Decimal("0.00"))

            for i, (input_tokens, output_tokens) in enumerate(zip(input_tokens_list, output_tokens_list)):
                prompt_cost = (Decimal(input_tokens) / Decimal("1000000")) * input_cost_mill
                completion_cost = (Decimal(output_tokens) / Decimal("1000000")) * output_cost_mill
                interaction_cost = prompt_cost + completion_cost

                interaction_costs.append({
                    "interaction": i + 1,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "cost": interaction_cost
                })

                total_prompt_tokens += input_tokens
                total_completion_tokens += output_tokens
                total_cost += interaction_cost

        else:
            st.warning(f"Pricing not found for primary model '{primary_model}'. Cost calculation might be incomplete.")

        return interaction_costs, total_prompt_tokens, total_completion_tokens, total_cost

    except Exception as e:
        st.error(f"Error during cost calculation: {e}")
        traceback.print_exc()
        return [], 0, 0, Decimal("0.00")

def run_agent():
    uri = os.getenv("uri")
    client = MongoClient(uri)

    rag_toolkit = RagToolkit()
    # 
    agent = Agent(
            # model= OpenAIChat(id="gpt-4o-mini",temperature=0),
            model = Gemini(id="gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY")),
            # model=AzureAIFoundry(id="Phi-4"),
            # model=HuggingFace(id="distilbert-base-uncased", api_key=os.getenv("HUGGINGFACE_API_KEY"), temperature=0),           # model=Groq(id="llama-3.1-8b-instant",temperature=0),
            # reasoning_model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0),
            tools = [MongoDBUtility(uri=uri, db_name="Demo"), rag_toolkit, PandasTools],
            instructions="""
            You are an intelligent MongoDB assistant that dynamically constructs and executes queries based on user input. Follow these steps METICULOUSLY:

            1️⃣ **Schema Identification & Planning (MUST DO FIRST):**
               - Immediately use the `RagToolkit` with the original user query.
               - **Wait for the output** from `RagToolkit`. It will be a JSON string containing keys like `relevant_collection`, `relevant_fields`, `chain_of_thought`, and `reasoning`.
               - **Parse this JSON output.**
               - **CRITICAL:** Extract the `chain_of_thought` from the RAG tool's output. This is your **mandatory plan** for the subsequent steps.
               - Also extract the `relevant_collection` and `relevant_fields`.
               - Use the `get_collection_schema` tool to retrieve the schema of the `relevant_collection`. This will provide the datatypes and structure of the fields in the collection.
               - If the RAG tool or `get_collection_schema` returns an error, or if `relevant_collection` or `chain_of_thought` is null/missing, state that you cannot proceed with planning the query due to missing schema information or tool failure, explain the reason given in the 'reasoning' or 'error' field, and STOP.

            2️⃣ **Generate an Optimized Query (Based STRICTLY on RAG Plan):**
               - **Follow the step-by-step `chain_of_thought`** provided by the `RagToolkit` in Step 1 to construct the specific MongoDB query (or aggregation pipeline).
               - Use the `relevant_collection` identified in Step 1.
               - Implement filtering logic (`$eq`, `$gte`, `$regex`, etc.) exactly as described in the `chain_of_thought`.
               - Project *only* the `relevant_fields` identified in Step 1 (plus any other fields explicitly required by the `chain_of_thought` for filtering, aggregation, or sorting), avoiding `_id` unless specified in the plan or required.
               - If the `chain_of_thought` indicates aggregation, construct the aggregation pipeline as outlined.

            3️⃣ **Execute the Query (Based on RAG Plan & Generated Query):**
               - Determine the correct MongoDB tool to use based **BOTH** on the `chain_of_thought` from Step 1 and the query generated in Step 2:
                 - If the plan/query involves **counting** documents, use `CountDocumentsTool`.
                 - If the plan/query involves **retrieving multiple documents**, use `FindDocumentsTool`.
                 - If the plan/query involves **aggregation** (like averaging, grouping), use `AggregateDocumentsTool`.
               - Execute the precise query/pipeline generated in Step 2 using the chosen tool.

            4️⃣ **Return the Final Answer (MANDATORY):**
           - Always return the final answer to the user based on the query results.
           - **Show the final MongoDB query or aggregation pipeline** that you executed (the one generated in Step 2).
           - Present the result obtained from the MongoDB tool in Step 3.
           - **Format the output** clearly:
             - If the result is a count or a single aggregation result (like average), state it directly.
             - If the result is a list of documents (from `FindDocumentsTool`):
                 - Parse the string output from the tool into a Python list of dictionaries.
                 - Present the data in a readable format (e.g., formatted JSON snippet or a summary table description if too long).
           - Provide a **brief explanation** of the result in natural language.
           - If no results are found by the MongoDB query, state this clearly ("No matching documents found.").

            ---
            **Error Handling Notes:**
            - Prioritize the `RagToolkit` in Step 1. If it fails, do not attempt subsequent steps.
            - Handle potential errors during query execution (Step 3) gracefully. Report the error message from the tool.
            - Follow the Google Sheets saving logic and DataFrame fallback precisely as described in Step 4.
            ---
            **Example Snippets (Illustrative - actual execution depends on RAG output):**

            *Initial thought process for "List failed calls last week":*
            1. Call `RagToolkit` with "List failed calls last week".
            2. RAG Output (example): `{"relevant_collection": "calls", "relevant_fields": ["caller", "receiver", "timestamp", "status"], "chain_of_thought": "1. Filter 'calls' collection by status='failed'. 2. Filter by timestamp >= <start_of_last_week>. 3. Project caller, receiver, timestamp, status.", "reasoning": "Query asks for failed calls, schema has status and timestamp."}`
            3. Follow `chain_of_thought`: Build query `{"status": "failed", "timestamp": {"$gte": <date>}}` for collection `calls` with projection `{caller: 1, receiver: 1, timestamp: 1, status: 1, _id: 0}`.
            4. Plan indicates retrieving documents -> Use `FindDocumentsTool`.
            5. Execute `FindDocumentsTool` with the built query.
            6. Process result: Parse list, try Google Sheets, fallback to DataFrame if needed, present results.
            **Make sure the agent runs the whole process

            ---
            Always use the collection and fields identified by `RagToolkit`. Do not invent schema elements. Ensure queries align with the plan from the RAG tool. Use the current date provided for relative date calculations ONLY IF the RAG plan requires it.
        """,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    markdown=True,
)


    return agent
# agent = run_agent()
# agent.print_response("List all failed calls in the last month", stream=True)

def simulate_agno_response(prompt):
    agent = run_agent() 
    response = agent.print_response(prompt, stream=True)
    for char in response:
        yield char
        time.sleep(0.05)


with st.sidebar:
    if mongo_connected:
        st.header("Database Info")
        db_name = st.selectbox("Select Database", client.list_database_names())
        if db_name:
            try:
                collections = client[db_name].list_collection_names()
                st.write("Collections:", collections)
            except Exception as e:
                st.error(f"Error listing collections: {e}")
    else:
        st.write("Not connected to MongoDB.")
        st.header("Sample Questions")
    sample_question = st.selectbox(
        "Choose a question:",
        [
            "How many calls did Priya Sharma make this week?",
            "List all failed calls in the last 7 days.",
            "What is the average call duration for completed calls?",
            "Show me all call records",
        ],
    )

st.title("AI Copilot to Analyze your data")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        agent = run_agent()
        with st.spinner("Analyzing your question..."):
            result = agent.run(prompt)
        interaction_costs, total_prompt_tokens, total_completion_tokens, total_cost = calculate_cost_per_interaction(result)

        formatted_calls = result.formatted_tool_calls
        print("Formatted Tool Calls:")
        for call in formatted_calls:
            print(f"- {call}")
        tool_interactions = result.tools
        tool_names_used = set() 
        if tool_interactions:
            for interaction in tool_interactions:
                tool_names_used.add(interaction.get('tool_name')) 
        print("\nUnique Tool Names Used:")
        if tool_names_used:
            for name in tool_names_used:
                print(f"- {name}")
        else:
            print("- No tools were used.")
        print("\nDetailed Tool Interactions:")
        if tool_interactions:
            for i, interaction in enumerate(tool_interactions):
                print(f"Interaction {i+1}:")
                print(f"  Name: {interaction.get('tool_name')}")
                print(f"  Args: {interaction.get('tool_args')}")
                print("-" * 10)
        else:
            print("- No tool interactions occurred.")
        if result.tools: 
            with st.expander("🔍 Show Tool Interactions"):
                st.markdown("---") 
                for i, interaction in enumerate(result.tools):
                    tool_name = interaction.get('tool_name', 'N/A')
                    tool_args = interaction.get('tool_args', {})


                    import json
                    try:
                        args_str = json.dumps(tool_args, indent=2)
                    except TypeError: 
                        args_str = str(tool_args)


                    st.markdown(f"**Interaction {i+1}: {tool_name}**")
                    st.code(args_str, language='json')
                    st.markdown("---") 
        print(result)
        # --- Display Cost Per Interaction ---
        st.markdown("### Cost and Tokens Per Interaction")
        for interaction in interaction_costs:
            st.markdown(f"**Interaction {interaction['interaction']}**")
            st.write(f"Prompt Tokens: {interaction['prompt_tokens']}")
            st.write(f"Completion Tokens: {interaction['completion_tokens']}")
            st.write(f"Cost: ${interaction['cost']:.6f}")
            st.markdown("---")

        st.caption("Note: Cost estimation is based on reported agent model tokens (e.g., GPT-4o-mini) and may exclude costs incurred *inside* custom tools like RAG/Embedding unless explicitly tracked and reported by the tool.")
        st.markdown("### Total Tokens and Cost")
        st.write(f"**Total Prompt Tokens:** {total_prompt_tokens}")
        st.write(f"**Total Completion Tokens:** {total_completion_tokens}")
        st.write(f"**Total Cost (USD):** ${total_cost:.6f}")
        st.markdown("---")
        st.markdown("---")
        st.markdown(result.content)
        st.session_state.messages.append({"role": "assistant", "content":result.content })











