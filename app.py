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
from tools.tools1 import MongoDBUtility
from agno.models.google import Gemini
from tools.rag import RagToolkit
from tools.mcptool import MCPClient
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

MODEL_PRICING = {
    # OpenAI Models (Example Pricing - VERIFY CURRENT PRICES)
    "gpt-4o": {"input": Decimal("5.00"), "output": Decimal("15.00")},
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gpt-4-turbo": {"input": Decimal("10.00"), "output": Decimal("30.00")},
    "gpt-3.5-turbo-0125": {"input": Decimal("0.50"), "output": Decimal("1.50")},
    "gemini-2.0-flash": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gemini-1.5-flash": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    # Groq Models (Often free tier or very low cost - check their specifics)
    # Example: Assuming negligible cost for this callCrm if on free tier
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
    
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory, UserMemory
from agno.storage.sqlite import SqliteStorage
from rich.pretty import pprint
# UserId for the memories
# Database file for memory and storage
db_file = "tmp/agent.db"
# Initialize MCP Client
mcp_client = MCPClient(
    server_type="sse",  # Use "stdio" for local MongoDB
    url="mongodb+srv://Admin:monikacrm@crm.e2i9tyo.mongodb.net/?retryWrites=true&w=majority&appName=CRM"
)
# Initialize memory.v2
memory = Memory(
    # Use any model for creating memories
    model=Gemini(id="gemini-2.0-flash-exp"),
    db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
    delete_memories=True,
    clear_memories=True,
)

storage = SqliteStorage(table_name="agent_sessions", db_file=db_file)

def run_agent():
    uri = os.getenv("uri")
    client = MongoClient(uri)

    rag_toolkit = RagToolkit()
    # 
    agent = Agent(
            # model= OpenAIChat(id="gpt-4o-mini",temperature=0),
            model = Gemini(id="gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY")),
            
            # model=HuggingFace(id="distilbert-base-uncased", api_key=os.getenv("HUGGINGFACE_API_KEY"), temperature=0),           # model=Groq(id="llama-3.1-8b-instant",temperature=0),
            # reasoning_model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0),
            tools = [MongoDBUtility(uri), rag_toolkit, PandasTools(), ThinkingTools(())],
            instructions="""
            You are an advanced AI assistant designed to help users analyze and manage their CRM data for Runo, a SIM-based Call Management CRM. Your primary goal is to assist users in querying and understanding their data, which includes call records, agent performance, company information, emails, and WhatsApp messages. Follow these steps meticulously to ensure accurate and insightful responses:

                1️⃣ **Understand the User's Query:**
                2️⃣ **Schema Identification & Planning:**
                - Use the `RagToolkit` to analyze the query and retrieve relevant schema details, including `relevant_collection`, `relevant_fields`, and `chain_of_thought`.
                - Extract the `chain_of_thought` to plan the query execution step-by-step.
                - If schema details are missing or the RAG tool fails, explain the issue clearly and suggest alternative ways to proceed.

                3️⃣ **Generate and Validate the Query:**
                - Construct a MongoDB query or aggregation pipeline based on the `chain_of_thought` and schema details.
                - Validate the query or pipeline to ensure it is properly formatted and includes all required arguments.
                - Log the query or pipeline for debugging purposes.

                4️⃣ **Execute the Query:**
                - Use the appropriate tool like mcp_client to execute the query or pipeline.
                - Handle errors gracefully and provide meaningful error messages to the user.

                4️⃣ **Provide a Clear and Actionable Response:**
                - Present the query results in a user-friendly format, such as a summary table, JSON snippet, or natural language explanation.
                - For listing queries, ensure the results are displayed as a Pandas DataFrame or a similar tabular format for better readability.
                - Highlight key insights, trends, or anomalies based on the data.

                5️⃣ **Enable Sentiment Analysis or analysis of performance:**
                - Analyze the sentiment of messages, emails, or WhatsApp conversations use think tool for that.
                - Classify the sentiment as Positive, Negative, or Neutral.
                - Provide a summary of the sentiment analysis, including the overall sentiment and key phrases contributing to the sentiment.
                - Example:
                    - "The sentiment of the messages is mostly positive, with key phrases like 'great service' and 'excellent support' contributing to this sentiment."

                5️⃣ **Enable Follow-Up Questions:**
                - Always include at least 2 follow-up questions based on the user's query and the data retrieved.
                - Format follow-up questions under a section titled "Follow-Up Suggestions:" in the response.
                - Example:
                    Follow-Up Suggestions:
                    - Would you like to filter these results further?
                    - Do you want to compare this data with another time period or agent?

                6️⃣ **Error Handling and Transparency:**
                - If an error occurs during query execution, explain the issue clearly and provide suggestions to resolve it.
                - Always show the MongoDB query or pipeline executed for transparency.
                - If the query is too complex or the data is too large, suggest breaking it down into smaller parts.
                7️⃣ **Data Privacy and Security:**        
                - Ensure that sensitive data is handled appropriately and not exposed in the responses.


                7️⃣ **Advanced Query Support:**
                - Handle complex queries involving multiple conditions, aggregations, or cross-referencing data (e.g., "Show me the average call duration for failed calls made by Priya Sharma last month").
                - Use the current date for relative date calculations if required by the query.

                8️⃣ **Interactive and Context-Aware Assistance:**
                - Maintain context across multiple queries in a session to enable seamless follow-up questions.
                - Use natural language to explain technical details when necessary.

                ---
                **Example Queries You Can Handle:**
                - "List all calls made by Priya Sharma this week."
                - "What is the average call duration for completed calls?"
                - "How many WhatsApp messages were sent by agents last month?"
                - "Show me the top 5 agents with the highest call success rate."
                - "What is the total number of emails sent to clients this quarter?"

                **Follow-Up Suggestions:**
                - "Would you like to filter these results further?"
                - "Do you want to compare this data with another time period or agent?"
                - "Would you like to export this data to a CSV or Google Sheet?"
                **Important Notes:**
                - Do not ask the user for in which collection they want to search. You should be able to identify the collection based on the query.
                - If you dont know the answer to a question, say "I don't know" instead of making up an answer.
                ---
                Always aim to provide accurate, actionable, and insightful responses to help users make data-driven decisions.
                """,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    memory=memory,
    # Give the Agent the ability to update memories
    enable_agentic_memory=False,
    # OR - Run the MemoryManager after each response
    enable_user_memories=False,
    # Store the chat history in the database
    storage=storage,
    # Add the chat history to the messages
    add_history_to_messages=True,
    # Number of history runs
    num_history_runs=3,
    markdown=True,
)

    return agent
# try:
#     print("Initializing agent...")
#     agent = run_agent()
#     print("Agent initialized. Sending query...")
#     agent.print_response("the number of calls each agent made with duration greater than 10 minutes?", stream=True)
# except AttributeError as e:
#     st.error("An error occurred while processing the response. Please try again.")
#     print("AttributeError:", e)
# except Exception as e:
#     st.error(f"Unexpected error: {e}")
#     print("Error details:", e)
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

## Initialize session state for conversation history and agent
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = run_agent()  # Initialize the agent once and store it in session state

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input

if prompt := st.chat_input("Ask me anything about your data"):
    # Retrieve previous context from user memories
    # Validate prompt
    if not prompt.strip():
        st.error("Prompt cannot be empty. Please enter a valid query.")
        st.stop()
    previous_context = memory.search_user_memories(
        user_id="user_id",
        query="Retrieve previous context",
        retrieval_method="agentic",
    )

    # Add user input to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        agent = st.session_state.agent  # Reuse the agent from session state
        with st.spinner("Analyzing your question..."):
            result = agent.run(prompt, context=previous_context)

        # Add the agent's response to user memory
        memory.add_user_memory(
            user_id="user_id",
            memory=UserMemory(memory=result.content),
        )

        # Add the agent's response to session state
        st.session_state.messages.append({"role": "assistant", "content": result.content})

        # Display the agent's response
        st.markdown(result.content)

        # Calculate costs and tokens
        interaction_costs, total_prompt_tokens, total_completion_tokens, total_cost = calculate_cost_per_interaction(result)

        # Display Tool Interactions
        print("result:==============================================", result)
        st.markdown("### Tool Interactions")
        tool_interactions = result.tools
        print("Tool Interactions:", tool_interactions)
        if tool_interactions:
            for i, interaction in enumerate(tool_interactions):
                tool_name = interaction.get('tool_name', 'N/A')
                tool_args = interaction.get('tool_args', {})
                tool_output = interaction.get('content', 'No output')

                with st.expander(f"🔧 Tool Interaction {i+1}: {tool_name}"):
                    st.markdown(f"**Tool Name:** {tool_name}")
                    st.markdown("**Arguments:**")
                    st.code(json.dumps(tool_args, indent=2), language='json')
                    st.markdown("**Output:**")
                    st.code(tool_output, language='json')

        else:
            st.write("No tools were used in this interaction.")

        # Display LLM Interactions
        st.markdown("### LLM Interactions")
        print("LLM Interactions:", interaction_costs)
        if interaction_costs:
            for interaction in interaction_costs:
                with st.expander(f"🧠 LLM Interaction {interaction['interaction']}"):
                    st.write(f"**Prompt Tokens:** {interaction['prompt_tokens']}")
                    st.write(f"**Completion Tokens:** {interaction['completion_tokens']}")
                    st.write(f"**Cost:** ${interaction['cost']:.6f}")
                # Display the input (prompt) and output (response) for the interaction
                    # Display the input (prompt) and output (response) for the interaction
                    interaction_index = interaction["interaction"] - 1  # Adjust index for zero-based indexing
                    if hasattr(result, "messages") and len(result.messages) > interaction_index:
                        llm_message = result.messages[interaction_index]

                        # Extract input and output from the message
                        input_content = (llm_message.content, "No input available") if llm_message.role == "user" else "No input available"
                        output_content = (llm_message.content, "No output available") if llm_message.role == "assistant" else "No output available"

                        st.markdown("**Input (Prompt):**")
                        st.code(input_content, language="text")
                        st.markdown("**Output (Response):**")
                        st.code(output_content, language="text")
        else:
            st.write("No LLM interactions were recorded.")
        # Display Total Costs and Tokens
        st.markdown("### Total Tokens and Cost")
        st.write(f"**Total Prompt Tokens:** {total_prompt_tokens}")
        st.write(f"**Total Completion Tokens:** {total_completion_tokens}")
        st.write(f"**Total Cost (USD):** ${total_cost:.6f}")

        # Include all costs in the final calculation
        tool_costs = sum(interaction.get('cost', Decimal("0.00")) for interaction in interaction_costs)
        final_total_cost = total_cost + tool_costs

        # Display Final Total Cost
        st.markdown("### Final Total Cost (Including Tools and LLM)")
        st.write(f"**Final Total Cost (USD):** ${final_total_cost:.6f}")