import os
import json
import pickle
import traceback
from datetime import datetime
import faiss
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.server_api import ServerApi
from pydantic import BaseModel, Field, ValidationError
import time 
import os
import json
import traceback
load_dotenv()


GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://monika:wOcbxCsRVJIDsphl@crm.hd2v6c5.mongodb.net/?retryWrites=true&w=majority&appName=CRM") # Use env var or default
FAISS_INDEX_PATH = "faiss_metadata_index.pkl"
METADATA_EMBEDDINGS_COLLECTION = "schema_embeddings_store"


if not GOOGLE_KEY:
    raise ValueError("Error: GOOGLE_API_KEY environment variable not set.")
if not OPENAI_KEY:
    raise ValueError("Error: OPENAI_API_KEY environment variable not set.")
if not MONGO_URI:
    raise ValueError("Error: MONGO_URI environment variable not set.")


os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY


app = FastAPI(
    title="MongoDB Schema RAG API",
    description="API to upload data, generate schema embeddings, and query which collections/fields are relevant.",
)


try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    db = client["Demo"]
except ConnectionFailure as e:
    print(f"MongoDB Connection Error: {e}")
    raise RuntimeError(f"Could not connect to MongoDB at {MONGO_URI}") from e
except Exception as e:
    print(f"An error occurred during MongoDB initialization: {e}")
    raise RuntimeError("Failed to initialize MongoDB client") from e


try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_KEY)
except Exception as e:
    print(f"Failed to initialize OpenAI Embeddings: {e}")
    raise RuntimeError("Could not initialize OpenAI embedding model.") from e


try:
    genai.configure(api_key=GOOGLE_KEY)
    llm_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Failed to initialize Google Generative AI model: {e}")
    raise RuntimeError("Could not initialize Google Generative AI model.") from e


class IdentifiedSchema(BaseModel):
    """Structure for the schema details identified by the LLM."""
    relevant_collection: Optional[str] = Field(
        None,
        description="The name of the most relevant MongoDB collection. Null if none found/applicable."
    )
    relevant_fields: List[str] = Field(
        default_factory=list,
        description="A list of fields within the collection needed to answer the query."
    )
    chain_of_thought: Optional[str] = Field(
        None,
        description="The step-by-step plan outlining how to construct the MongoDB query using the relevant collection/fields to answer the user query (e.g., identify filters, aggregations, sorting)."
    )
    reasoning: str = Field(..., description="The LLM's reasoning for selecting the collection and fields, or why it's not possible.")
    raw_llm_response: Optional[str] = Field(
        None,
        description="Raw text response from LLM, mainly for debugging JSON parsing issues.",
        exclude=True
    )


class QueryResponse(BaseModel):
    """Overall response structure for the /identify_collection_fields endpoint."""
    query: str
    identified_schema: IdentifiedSchema
    retrieved_schema_context: List[str]


@app.post("/upload_json/", tags=["Data Management"])
async def upload_json(file: UploadFile = File(..., description="JSON file containing data for a new collection.")):
    """
    Uploads a JSON file. Each top-level key in the JSON becomes a collection,
    or if the JSON is a list, the filename (without extension) becomes the collection name.
    Data (list of documents) is inserted into the corresponding collection.
    """
    try:
        table_name = os.path.splitext(file.filename)[0]
        contents = await file.read()
        data = json.loads(contents)


        if isinstance(data, dict):
            results = {}
            for name, docs in data.items():
                if isinstance(docs, list):
                    collection = db[name]
                    if docs:
                        insert_result = collection.insert_many(docs)
                        results[name] = len(insert_result.inserted_ids)
                    else:
                        results[name] = 0
                else:
                    print(f"Skipping key '{name}' in {file.filename}: value is not a list of documents.")
            total_inserted = sum(results.values())
            return {"message": "Data inserted from dictionary keys", "records_inserted": results, "total_records": total_inserted}


        elif isinstance(data, list):
            collection = db[table_name]
            if data:
                insert_result = collection.insert_many(data)
                return {"message": f"Data inserted into collection '{table_name}'", "total_records": len(insert_result.inserted_ids)}
            else:
                return {"message": f"Data inserted into collection '{table_name}'", "total_records": 0}
        else:
            raise HTTPException(status_code=400, detail="Invalid JSON format. Must be a list of objects or a dictionary where values are lists of objects.")


    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file provided.")
    except Exception as e:
        print(f"Error in upload_json: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/extract_schema", tags=["Schema Management"])
def extract_schema():
    """
    Extracts the schema (collection names and field names) from the MongoDB database.
    """
    try:
        collections = db.list_collection_names()
        collections = [c for c in collections if not c.startswith('system.') and c != METADATA_EMBEDDINGS_COLLECTION]


        schema_info = {}
        for collection_name in collections:


            sample_doc = db[collection_name].find_one({}, {"_id": 0})
            if sample_doc:
                schema_info[collection_name] = list(sample_doc.keys())
            else:
                indexes = db[collection_name].index_information()
                fields = set()
                for index_info in indexes.values():
                    for key_tuple in index_info.get('key', []):
                        fields.add(key_tuple[0])
                schema_info[collection_name] = list(fields) if fields else []
       
        if not schema_info:
             return {"message": "No user collections with data or indexes found to extract schema from."}


        return {
            "message": "Schema extracted successfully",
            "collections": list(schema_info.keys()),
            "schema": schema_info
        }
    except Exception as e:
        print(f"Error in extract_schema: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to extract schema: {str(e)}")




@app.post("/generate_schema_embeddings", tags=["Embeddings"])
def generate_schema_embeddings():
    """
    Extracts the database schema, splits it into chunks, generates embeddings
    for each chunk using OpenAI, and saves them to the MongoDB collection
    specified by METADATA_EMBEDDINGS_COLLECTION.
    """
    try:
        schema_data = extract_schema()
        collections_fields = schema_data.get("schema", {})


        if not collections_fields:
             return {"message": "No schema information found to generate embeddings."}




        structured_schema = collections_fields
        json_splitter = RecursiveJsonSplitter(max_chunk_size=200)
        schema_chunks = json_splitter.split_json(structured_schema)


        if not schema_chunks:
             raise HTTPException(status_code=500, detail="Schema splitting resulted in zero chunks.")


        print(f"Generated {len(schema_chunks)} schema chunks.")


        embeddings_to_store = []
        for i, chunk in enumerate(schema_chunks):


            text_chunk = json.dumps(chunk, sort_keys=True) if isinstance(chunk, dict) else str(chunk)


            if not text_chunk or text_chunk in ['{}', '[]', '""', "''"]:
                print(f"Skipping empty chunk {i}: {text_chunk}")
                continue


            print(f"Embedding chunk {i+1}/{len(schema_chunks)}: {text_chunk[:150]}...")
            embedding = embeddings_model.embed_query(text_chunk)
            embeddings_to_store.append({
                "chunk_text": text_chunk,
                "embedding": embedding    
            })


        if not embeddings_to_store:
             raise HTTPException(status_code=500, detail="No valid schema chunks could be embedded.")


        embedding_collection = db[METADATA_EMBEDDINGS_COLLECTION]
        print(f"Deleting old embeddings from '{METADATA_EMBEDDINGS_COLLECTION}'...")
        delete_result = embedding_collection.delete_many({})
        print(f"Deleted {delete_result.deleted_count} old documents.")


        print(f"Inserting {len(embeddings_to_store)} new schema embeddings...")
        insert_result = embedding_collection.insert_many(embeddings_to_store)
        print(f"Successfully inserted {len(insert_result.inserted_ids)} embeddings.")


        return {
            "message": f"Schema split, embedded, and saved to '{METADATA_EMBEDDINGS_COLLECTION}' successfully.",
            "num_chunks_generated": len(schema_chunks),
            "num_embeddings_stored": len(insert_result.inserted_ids),
        }


    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in generate_schema_embeddings: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate/save schema embeddings: {str(e)}")




@app.post("/build_save_faiss_index", tags=["FAISS"])
def build_save_faiss_index():
    """
    Loads schema embeddings from MongoDB (METADATA_EMBEDDINGS_COLLECTION),
    builds a FAISS index, and saves the index and corresponding text chunks
    to a pickle file (FAISS_INDEX_PATH). Overwrites existing file.
    """


    try:
        embedding_collection = db[METADATA_EMBEDDINGS_COLLECTION]
        embeddings_data = list(embedding_collection.find({}, {"_id": 0, "embedding": 1, "chunk_text": 1}))


        if not embeddings_data:
            raise HTTPException(status_code=404, detail=f"No embeddings found in MongoDB collection '{METADATA_EMBEDDINGS_COLLECTION}'. Run /generate_schema_embeddings first.")


        embeddings_list = [data["embedding"] for data in embeddings_data if "embedding" in data]
        chunks_list = [data["chunk_text"] for data in embeddings_data if "chunk_text" in data]


        if not embeddings_list or len(embeddings_list) != len(chunks_list):
             raise HTTPException(status_code=500, detail="Mismatch between embeddings and chunks found in DB or data is missing.")


        vectors = np.array(embeddings_list).astype('float32')
        if vectors.ndim != 2:
             raise HTTPException(status_code=500, detail=f"Embeddings do not form a valid 2D array. Shape: {vectors.shape}")


        dimension = vectors.shape[1]
        print(f"Building FAISS IndexFlatL2 with dimension {dimension} for {len(chunks_list)} vectors.")


        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        print(f"FAISS index built successfully. Total vectors added: {index.ntotal}")


        print(f"Saving FAISS index and chunks to '{FAISS_INDEX_PATH}'...")
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "chunks": chunks_list}, f)


        return {
            "message": f"FAISS index built and saved successfully to {FAISS_INDEX_PATH}!",
            "vectors_indexed": index.ntotal,
            "index_dimension": dimension
            }


    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in build_save_faiss_index: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to build/save FAISS index: {str(e)}")




# @app.post("/identify_collection_fields", tags=["Query"], response_model=QueryResponse)
# def identify_collection_fields(query: str = Query(..., description="User's natural language query about the data.")):
#     """
#     Takes a natural language query, finds relevant schema information using RAG,
#     and asks an LLM to identify the MongoDB collection and fields needed to answer the query.
#     Returns a structured response validated by Pydantic.
#     """
from agno.tools import Toolkit

def parse_llm_response(raw_llm_text: str) -> dict:
    """
    Parses the raw LLM response text into a JSON object.

    Args:
        raw_llm_text (str): The raw response text from the LLM.

    Returns:
        dict: Parsed JSON object if successful, or an error structure if parsing fails.
    """
    try:
        # Clean the response text
        cleaned_response_text = raw_llm_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[len("```json"):].strip()
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-len("```")].strip()

        # Parse the cleaned response text into JSON
        llm_output_dict = json.loads(cleaned_response_text)
        print("Successfully parsed LLM JSON output.")
        return llm_output_dict

    except json.JSONDecodeError as json_err:
        print(f"Warning: LLM response was not valid JSON: {json_err}")
        return {
            "error": "LLM output is not valid JSON",
            "reasoning": f"LLM did not provide valid JSON output. Error: {json_err}. Raw text: {raw_llm_text}",
            "raw_response": raw_llm_text
        }
    except Exception as e:
        print(f"Unexpected error during JSON parsing: {traceback.format_exc()}")
        return {
            "error": "Unexpected error during JSON parsing",
            "reasoning": str(e),
            "raw_response": raw_llm_text
        }
  
class RagToolkit(Toolkit):
    def __init__(self, FAISS_INDEX_PATH= "/faiss_metadata_index.pkl"):
        """Initialize Ragtoolkit connection."""
        super().__init__(name="rag_toolkit")
        self.register(self.ragcot)


    def ragcot(self, query: str) -> QueryResponse:
        cost_metrics = {
            "embedding_cost": 0,
            "llm_cost": 0,
            "execution_time": 0,
        }
        try:
            print(f"Embedding query: '{query}'")
            query_embedding = embeddings_model.embed_query(query)
            query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

            if not os.path.exists(FAISS_INDEX_PATH):
                raise HTTPException(status_code=404, detail=f"FAISS index file '{FAISS_INDEX_PATH}' not found. Please run /build_save_faiss_index first.")

            print(f"Loading FAISS index and chunks from '{FAISS_INDEX_PATH}'...")
            with open(FAISS_INDEX_PATH, "rb") as f:
                faiss_data = pickle.load(f)

            index = faiss_data.get("index")
            chunks_from_pickle = faiss_data.get("chunks")

            if index is None or chunks_from_pickle is None:
                raise HTTPException(status_code=500, detail=f"Pickle file '{FAISS_INDEX_PATH}' is missing 'index' or 'chunks' key.")

            k = 3
            print(f"Searching FAISS index (k={k})...")
            distances, indices = index.search(query_vector, k)
            print(f"FAISS search results (indices): {indices[0]}")
            print(f"FAISS search results (distances): {distances[0]}")

            relevant_schema_chunks = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(chunks_from_pickle):
                    chunk_text = chunks_from_pickle[idx]
                    relevant_schema_chunks.append(str(chunk_text))
                    print(f"  - Retrieved chunk {i} (index {idx}, distance {distances[0][i]}): {chunk_text[:150]}...")
                else:
                    print(f"Warning: Invalid index {idx} received from FAISS search.")

            if not relevant_schema_chunks:
                print("Warning: No relevant schema chunks found for the query.")
                context = "No relevant schema information found."
            else:
                context = "\n---\n".join(relevant_schema_chunks)

            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = f"""
            You are an expert MongoDB assistant. Your task is to analyze a user's query and the provided database schema context. First, identify the relevant collection and fields. Then, outline a step-by-step plan (chain of thought) for how you would construct the MongoDB query to answer the user's request using that information.

            **Instructions:**
            1. Analyze Query & Context: Carefully read the "User Query" and examine the "Schema Context" (snippets of MongoDB schema).
            2. Identify Collection & Fields: Determine the SINGLE most relevant MongoDB collection and the specific fields (for filtering and projection) necessary to answer the query based *only* on the schema context. If it's not possible, explain why in the reasoning.
            3. Plan Query Construction (Chain of Thought): Based on the User Query and the identified collection/fields, outline the step-by-step logical plan to construct the necessary MongoDB query. Detail the required operations like:
                * Filtering criteria (e.g., matching agent name, date ranges based on '{current_date}').
                * Required projections (which fields to return).
                * Any necessary aggregations (e.g., counting, summing, grouping).
                * Sorting requirements.
                * *Do NOT write the actual MongoDB syntax here, just describe the steps.* This plan should be included in the final JSON output under the "chain_of_thought" key.
            4. Reasoning: Briefly explain *why* the chosen collection and fields are appropriate (linking back to the user query and schema) OR provide the reasoning if the query cannot be answered or planned using the context.
            5. Output Format: Provide the output *ONLY* in valid JSON format with the following keys: "relevant_collection" (string or null), "relevant_fields" (list of strings), "chain_of_thought" (string describing the query plan steps from instruction 3), and "reasoning" (string).

            **Constraints:**
            * Do NOT invent collections or fields not present in the Schema Context.
            * The chain_of_thought should describe the *plan*, not the final query code.

            **Current Date:** {current_date}

            **Schema Context:**
            ```json
            {context}
            ```

            **User Query:**
            ```
            {query}
            ```

            **Output (JSON):**
            """
            print(f"\n--- Sending Prompt to LLM ---")
            print(f"Context included: {context[:500]}...")
            print("--- End Prompt ---")

            response = llm_model.generate_content(prompt)
            usage_metadata = response.usage_metadata
            input_tokens = usage_metadata.prompt_token_count
            output_tokens = usage_metadata.candidates_token_count

            input_cost = (input_tokens / 1_000_000) * 0.15  # $0.15 per million input tokens
            output_cost = (output_tokens / 1_000_000) * 0.60  # $0.60 per million output tokens
            cost_metrics["llm_cost"] += input_cost + output_cost
            print(f"LLM Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
            print(f"LLM Cost: ${input_cost + output_cost:.6f}")
            if not response.parts:
                print(f"Warning: LLM response was blocked. Feedback: {response.prompt_feedback}")
                identified_schema_obj = IdentifiedSchema(
                    relevant_collection=None,
                    relevant_fields=[],
                    reasoning=f"LLM response blocked due to safety settings. Feedback: {response.prompt_feedback}",
                    raw_llm_response=None
                )
                return QueryResponse(
                    query=query,
                    identified_schema=identified_schema_obj,
                    retrieved_schema_context=relevant_schema_chunks
                )

            raw_llm_text = response.text
            print(f"\n--- LLM Raw Response Text ({len(raw_llm_text)} chars) ---")
            print(raw_llm_text)
            print(f"--- End LLM Raw Response Text ---")

            try:
                cleaned_response_text = raw_llm_text.strip()
                if cleaned_response_text.startswith("```json"):
                    cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-len("```")].strip()

                llm_output_dict = json.loads(cleaned_response_text)

                identified_schema_obj = IdentifiedSchema(**llm_output_dict)
                print("Successfully parsed and validated LLM JSON output.")

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Error parsing or validating LLM response: {e}")
                identified_schema_obj = IdentifiedSchema(
                    relevant_collection=None,
                    relevant_fields=[],
                    reasoning=f"Failed to parse or validate LLM response: {e}",
                    raw_llm_response=raw_llm_text
                )

            return QueryResponse(
                query=query,
                identified_schema=identified_schema_obj,
                retrieved_schema_context=relevant_schema_chunks
            )

        except Exception as e:
            print(f"Unexpected error in ragcot: {traceback.format_exc()}")
            identified_schema_obj = IdentifiedSchema(
                relevant_collection=None,
                relevant_fields=[],
                reasoning=f"Unexpected error: {str(e)}",
                raw_llm_response=None
            )
            return QueryResponse(
                query=query,
                identified_schema=identified_schema_obj,
                retrieved_schema_context=[]
            )
#         #     identified_schema_obj: IdentifiedSchema
        #     try:


        #         response = llm_model.generate_content(prompt)


        #         if not response.parts:
        #             print(f"Warning: LLM response was blocked. Feedback: {response.prompt_feedback}")
        #             identified_schema_obj = IdentifiedSchema(
        #                 relevant_collection=None,
        #                 relevant_fields=[],
        #                 reasoning=f"LLM response blocked due to safety settings. Feedback: {response.prompt_feedback}",
        #                 raw_llm_response=None
        #             )
        #             print("pydantic response:",QueryResponse(
        #                 query=query,
        #                 identified_schema=identified_schema_obj,
        #                 retrieved_schema_context=relevant_schema_chunks
        #             ))
        #             return QueryResponse(
        #                 query=query,
        #                 identified_schema=identified_schema_obj,
        #                 retrieved_schema_context=relevant_schema_chunks
        #             )


        #         raw_llm_text = response.text


        #         print(f"\n--- LLM Raw Response Text ({len(raw_llm_text)} chars) ---")
        #         print(raw_llm_text)
        #         print(f"--- End LLM Raw Response Text ---")


        #         try:
        #             cleaned_response_text = raw_llm_text.strip()
        #             if cleaned_response_text.startswith("```json"):
        #                 cleaned_response_text = cleaned_response_text[len("```json"):].strip()
        #             if cleaned_response_text.endswith("```"):
        #                 cleaned_response_text = cleaned_response_text[:-len("```")].strip()


        #             llm_output_dict = json.loads(cleaned_response_text)


        #             try:
        #                 if "relevant_fields" in llm_output_dict and llm_output_dict["relevant_fields"] is None:
        #                     llm_output_dict["relevant_fields"] = []


        #                 identified_schema_obj = IdentifiedSchema(**llm_output_dict)
        #                 print("Successfully parsed and validated LLM JSON output.")
        #             except ValidationError as val_err:
        #                 print(f"Pydantic Validation Error: {val_err}")
        #                 identified_schema_obj = IdentifiedSchema(
        #                     relevant_collection=llm_output_dict.get("relevant_collection"),
        #                     relevant_fields=llm_output_dict.get("relevant_fields", []),
        #                     reasoning=f"LLM response parsed as JSON, but failed Pydantic validation: {val_err}. Raw JSON tried: {cleaned_response_text}",
        #                     raw_llm_response=raw_llm_text
        #                 )
        #         except (json.JSONDecodeError, AttributeError) as json_err:
        #             print(f"Warning: LLM response was not valid JSON: {json_err}")
        #             identified_schema_obj = IdentifiedSchema(
        #                 relevant_collection=None,
        #                 relevant_fields=[],
        #                 reasoning=f"LLM did not provide valid JSON output. Error: {json_err}. Raw text: {raw_llm_text}",
        #                 raw_llm_response=raw_llm_text
        #             )


        #     except Exception as gen_err:
        #         print(f"Error during LLM content generation or processing: {traceback.format_exc()}")
        #         identified_schema_obj = IdentifiedSchema(
        #             relevant_collection=None,
        #             relevant_fields=[],
        #             reasoning=f"LLM content generation or processing failed: {str(gen_err)}",
        #             raw_llm_response=None
        #         )
        #     return str(QueryResponse(
        #         query=query,
        #         identified_schema=identified_schema_obj,
        #         retrieved_schema_context=relevant_schema_chunks
        #     ))
        # except FileNotFoundError:
        #     print(f"Error: FAISS index file not found at '{FAISS_INDEX_PATH}'")
        #     raise HTTPException(status_code=404, detail=f"FAISS index file '{FAISS_INDEX_PATH}' not found. Run build_save_faiss_index first.")
        # except pickle.UnpicklingError as pkl_err:
        #     print(f"Error unpickling FAISS index from '{FAISS_INDEX_PATH}': {pkl_err}")
        #     raise HTTPException(status_code=500, detail=f"Error loading FAISS index from {FAISS_INDEX_PATH}. File might be corrupted. Regenerate it.")
        # except RuntimeError as fe:
        #     print(f"Runtime error encountered: {traceback.format_exc()}")
        #     raise HTTPException(status_code=500, detail=f"Runtime error during processing: {str(fe)}")
        # except HTTPException as http_exc:
        #     raise http_exc
        # except Exception as e:
        #     print(f"Unexpected error in identify_collection_fields: {traceback.format_exc()}")
        #     raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")




# # from your_agent_framework import BaseTool # Replace with actual framework import
# from agno.tools import Toolkit


# # Define paths (ensure these are accessible where the tool runs)
# FAISS_INDEX_PATH = "/faiss_metadata_index.pkl" # IMPORTANT: Set the correct path


# class SchemaRAGTool(Toolkit):


#     self.register(self._run)
#     name: str = "SchemaRAGTool"
#     description: str = (
#         "Analyzes a user query and database schema context (via RAG) "
#         "to identify the relevant MongoDB collection, fields, and generate a step-by-step "
#         "plan (chain of thought) for constructing the MongoDB query. "
#         "This should be the FIRST step before trying to build or execute a query."
#     )


#     def _run(self, query: str) -> str:
#         """Executes the RAG process and returns the structured JSON result."""
#         try:
#             print(f"SchemaRAGTool: Embedding query: '{query}'")
#             query_embedding = embeddings_model.embed_query(query)
           
#             if isinstance(query_embedding, list):
#                  query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
#             elif hasattr(query_embedding, 'detach'):
#                 query_vector = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)
#             else:
#                 query_vector = query_embedding.astype('float32').reshape(1, -1)




#             if not os.path.exists(FAISS_INDEX_PATH):
#                 return json.dumps({
#                     "error": f"FAISS index file '{FAISS_INDEX_PATH}' not found. Cannot plan query."
#                 })


#             print(f"SchemaRAGTool: Loading FAISS index and chunks from '{FAISS_INDEX_PATH}'...")
#             with open(FAISS_INDEX_PATH, "rb") as f:
#                 faiss_data = pickle.load(f)


#             # --- FAISS Index Loading ---
#             # Assuming the pickle contains a raw FAISS index object
#             # If using Langchain VectorStore, the loading might be different
#             # Example assumes raw faiss index object was pickled
#             try:
#                 # If you pickled a langchain FAISS object:
#                 # index = faiss_data # Or however it's stored
#                 # relevant_docs = index.similarity_search_by_vector(query_vector[0], k=3)
#                 # relevant_schema_chunks = [doc.page_content for doc in relevant_docs]


#                 # If you pickled the raw index and chunks separately:
#                 index = faiss_data.get("index") # Assuming 'index' stores the faiss.Index object
#                 chunks_from_pickle = faiss_data.get("chunks") # Assuming 'chunks' stores list[str]


#                 if index is None or chunks_from_pickle is None:
#                      return json.dumps({
#                         "error": f"Pickle file '{FAISS_INDEX_PATH}' is missing 'index' or 'chunks' key."
#                     })


#                 # Ensure FAISS index is loaded correctly if it wasn't deserialized properly
#                 # This step might be needed depending on how faiss was pickled
#                 # import faiss
#                 # index = faiss.deserialize_index(index_bytes) # If you saved bytes


#                 k = 3
#                 print(f"SchemaRAGTool: Searching FAISS index (k={k})...")
#                 # FAISS search returns distances, indices
#                 distances, indices = index.search(query_vector, k)
#                 print(f"SchemaRAGTool: FAISS search results (indices): {indices[0]}")


#                 relevant_schema_chunks = []
#                 for i, idx in enumerate(indices[0]):
#                      # Check index bounds BEFORE accessing chunks_from_pickle
#                     if 0 <= idx < len(chunks_from_pickle):
#                         chunk_text = chunks_from_pickle[idx]
#                         relevant_schema_chunks.append(str(chunk_text)) # Ensure string
#                         print(f"  - Retrieved chunk {i} (index {idx}, distance {distances[0][i]})")
#                     else:
#                         print(f"Warning: Invalid index {idx} (out of bounds for {len(chunks_from_pickle)}) received from FAISS search.")




#             except Exception as faiss_err:
#                  print(f"SchemaRAGTool: Error during FAISS search/load: {traceback.format_exc()}")
#                  return json.dumps({"error": f"Failed during FAISS operation: {str(faiss_err)}"})




#             if not relevant_schema_chunks:
#                 print("SchemaRAGTool: Warning: No relevant schema chunks found.")
#                 context = "No relevant schema information found."
#             else:
#                 context = "\n---\n".join(relevant_schema_chunks)


#             current_date = datetime.now().strftime("%Y-%m-%d")
#             prompt = f"""
#             You are an expert MongoDB assistant. Your task is to analyze a user's query and the provided database schema context. First, identify the relevant collection and fields. Then, outline a step-by-step plan (chain of thought) for how you would construct the MongoDB query to answer the user's request using that information.


#             **Instructions:**
#             1.  **Analyze Query & Context:** Carefully read the "User Query" and examine the "Schema Context" (snippets of MongoDB schema).
#             2.  **Identify Collection & Fields:** Determine the SINGLE most relevant MongoDB collection and the specific fields (for filtering and projection) necessary to answer the query based *only* on the schema context. If it's not possible, explain why in the reasoning.
#             3.  **Plan Query Construction (Chain of Thought):** Based on the User Query and the identified collection/fields, outline the step-by-step logical plan to construct the necessary MongoDB query. Detail the required operations like:
#                 *   Filtering criteria (e.g., matching agent name, date ranges based on '{current_date}').
#                 *   Required projections (which fields to return).
#                 *   Any necessary aggregations (e.g., counting, summing, grouping).
#                 *   Sorting requirements.
#                 *   *Do NOT write the actual MongoDB syntax here, just describe the steps.* This plan should be included in the final JSON output under the "chain_of_thought" key.
#             4.  **Reasoning:** Briefly explain *why* the chosen collection and fields are appropriate (linking back to the user query and schema) OR provide the reasoning if the query cannot be answered or planned using the context.
#             5.  **Output Format:** Provide the output *ONLY* in valid JSON format with the following keys: "relevant_collection" (string or null), "relevant_fields" (list of strings), "chain_of_thought" (string describing the query plan steps from instruction 3), and "reasoning" (string).


#             **Constraints:**
#             *   Do NOT invent collections or fields not present in the Schema Context.
#             *   The chain_of_thought should describe the *plan*, not the final query code.


#             **Current Date:** {current_date}


#             **Schema Context:**
#             ```json
#             {context}
#             ```


#             **User Query:**
#             ```
#             {query}
#             ```


#             **Output (JSON):**
#             """
#             print(f"\n--- SchemaRAGTool: Sending Prompt to LLM ---")
#             # print(f"Context included: {context[:500]}...")
#             # print(prompt) # Optional: print full prompt for debugging
#             print("--- SchemaRAGTool: End Prompt ---")


#             try:
#                 response = llm_model.generate_content(prompt) # Adjust based on your LLM wrapper


#                 # Handle blocked responses (common with generative AI safety filters)
#                 if not response.parts:
#                     feedback = getattr(response, 'prompt_feedback', 'No feedback available')
#                     print(f"SchemaRAGTool: Warning: LLM response was blocked. Feedback: {feedback}")
#                     # Return a structured error message
#                     error_output = {
#                         "relevant_collection": None,
#                         "relevant_fields": [],
#                         "chain_of_thought": None,
#                         "reasoning": f"LLM response blocked. Cannot plan query. Feedback: {feedback}",
#                         "error": "LLM response blocked"
#                      }
#                     return json.dumps(error_output)




#                 raw_llm_text = response.text # Adjust based on your LLM wrapper


#                 print(f"\n--- SchemaRAGTool: LLM Raw Response Text ---")
#                 print(raw_llm_text)
#                 print(f"--- SchemaRAGTool: End LLM Raw Response Text ---")


#                 # Attempt to parse JSON
#                 cleaned_response_text = raw_llm_text.strip()
#                 if cleaned_response_text.startswith("```json"):
#                     cleaned_response_text = cleaned_response_text[len("```json"):].strip()
#                 if cleaned_response_text.endswith("```"):
#                     cleaned_response_text = cleaned_response_text[:-len("```")].strip()


#                 try:
#                     llm_output_dict = json.loads(cleaned_response_text)


#                     # Validate with Pydantic (optional but recommended)
#                     try:
#                         # Handle potential None for fields before validation
#                         if "relevant_fields" in llm_output_dict and llm_output_dict["relevant_fields"] is None:
#                             llm_output_dict["relevant_fields"] = []
#                         # Ensure all required fields for the tool's purpose are present
#                         if "relevant_collection" not in llm_output_dict: llm_output_dict["relevant_collection"] = None
#                         if "chain_of_thought" not in llm_output_dict: llm_output_dict["chain_of_thought"] = None
#                         if "reasoning" not in llm_output_dict: llm_output_dict["reasoning"] = "N/A"




#                         # We don't need full Pydantic validation here, just return the dict
#                         # identified_schema_obj = IdentifiedSchema(**llm_output_dict)
#                         print("SchemaRAGTool: Successfully parsed LLM JSON output.")
#                         return json.dumps(llm_output_dict) # Return JSON string


#                     except ValidationError as val_err:
#                         print(f"SchemaRAGTool: Pydantic Validation Error: {val_err}")
#                         # Return structured error with details
#                         error_output = {
#                             "relevant_collection": llm_output_dict.get("relevant_collection"),
#                             "relevant_fields": llm_output_dict.get("relevant_fields", []),
#                             "chain_of_thought": llm_output_dict.get("chain_of_thought"),
#                             "reasoning": f"LLM response parsed, but failed validation: {val_err}. Raw JSON tried: {cleaned_response_text}",
#                             "error": "LLM output validation failed"
#                         }
#                         return json.dumps(error_output)


#                 except json.JSONDecodeError as json_err:
#                     print(f"SchemaRAGTool: Warning: LLM response was not valid JSON: {json_err}")
#                      # Return structured error
#                     error_output = {
#                         "relevant_collection": None,
#                         "relevant_fields": [],
#                         "chain_of_thought": None,
#                         "reasoning": f"LLM did not provide valid JSON output. Error: {json_err}. Raw text: {raw_llm_text}",
#                         "error": "LLM output is not valid JSON"
#                     }
#                     return json.dumps(error_output)


#             except Exception as gen_err:
#                 print(f"SchemaRAGTool: Error during LLM generation/processing: {traceback.format_exc()}")
#                 # Return structured error
#                 error_output = {
#                     "relevant_collection": None,
#                     "relevant_fields": [],
#                     "chain_of_thought": None,
#                     "reasoning": f"LLM content generation or processing failed: {str(gen_err)}",
#                     "error": "LLM generation failed"
#                 }
#                 return json.dumps(error_output)


#         except FileNotFoundError:
#              print(f"SchemaRAGTool: Error: FAISS index file not found at '{FAISS_INDEX_PATH}'")
#              return json.dumps({"error": f"FAISS index file '{FAISS_INDEX_PATH}' not found."})
#         except pickle.UnpicklingError as pkl_err:
#              print(f"SchemaRAGTool: Error unpickling FAISS index from '{FAISS_INDEX_PATH}': {pkl_err}")
#              return json.dumps({"error": f"Error loading FAISS index from {FAISS_INDEX_PATH}. File might be corrupted."})
#         except Exception as e:
#             print(f"SchemaRAGTool: Unexpected error: {traceback.format_exc()}")
#             return json.dumps({"error": f"An unexpected error occurred in SchemaRAGTool: {str(e)}"})


#     async def _arun(self, query: str) -> str:
#         # If your RAG function or LLM calls are async, implement here
#         # Otherwise, wrap the synchronous call
#         # from concurrent.futures import ThreadPoolExecutor
#         # with ThreadPoolExecutor() as executor:
#         #     result = await asyncio.get_event_loop().run_in_executor(executor, self._run, query)
#         # return result
#         # For simplicity, if _run is blocking but acceptable:
#         return self._run(query)

