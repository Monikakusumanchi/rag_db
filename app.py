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
from pydantic import BaseModel, Field, ValidationError # Added BaseModel, Field, ValidationError


load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://monika:wOcbxCsRVJIDsphl@crm.hd2v6c5.mongodb.net/?retryWrites=true&w=majority&appName=CRM") # Use env var or default
FAISS_INDEX_PATH = "faiss_metadata_index.pkl" # Renamed for clarity
METADATA_EMBEDDINGS_COLLECTION = "schema_embeddings_store" # Renamed for clarity

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
    reasoning: str = Field(
        ...,
        description="The LLM's reasoning for selecting the collection and fields, or why it's not possible."
    )
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


@app.post("/identify_collection_fields", tags=["Query", "RAG"], response_model=QueryResponse)
def identify_collection_fields(query: str = Query(..., description="User's natural language query about the data.")):
    """
    Takes a natural language query, finds relevant schema information using RAG,
    and asks an LLM to identify the MongoDB collection and fields needed to answer the query.
    Returns a structured response validated by Pydantic.
    """
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
                relevant_schema_chunks.append(str(chunk_text)) # Ensure string
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
                    You are an expert MongoDB assistant. Your task is to analyze a user's query and the provided database schema context to determine which collection and which specific fields are most relevant for answering that query.

                    **Instructions:**
                    1.  Carefully read the "User Query".
                    2.  Examine the "Schema Context" which contains snippets of the MongoDB database schema (collection names and their fields).
                    3.  Identify the SINGLE most relevant MongoDB collection for the query. If no single collection seems appropriate based *only* on the context, state null.
                    4.  List the specific fields within that collection that are NECESSARY to answer the user's query. Include fields needed for filtering (e.g., dates, names) and fields needed for the result (e.g., counts, values). If no collection is relevant, this should be an empty list.
                    5.  If the query cannot be answered with the provided schema context, state that clearly in the reasoning and explain why (e.g., missing collection, missing field in context).
                    6.  Do NOT attempt to write the full MongoDB query.
                    7.  Do NOT invent collections or fields not present in the context.
                    8.  Provide the output *ONLY* in JSON format with keys "relevant_collection" (string or null), "relevant_fields" (list of strings), and "reasoning" (string).

                    **Current Date:** {current_date}

                    **Schema Context:**
                    ```json
                    {context} 
                    User Query:
                    {query}
                    Output (JSON):

                    """  
        print(f"\n--- Sending Prompt to LLM ---")
        print(f"Context included: {context[:500]}...")
        print("--- End Prompt ---")

        identified_schema_obj: IdentifiedSchema
        try:

            response = llm_model.generate_content(prompt)

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

            raw_llm_text = response.text # Get text

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

                try:
                    if "relevant_fields" in llm_output_dict and llm_output_dict["relevant_fields"] is None:
                        llm_output_dict["relevant_fields"] = []

                    identified_schema_obj = IdentifiedSchema(**llm_output_dict)
                    print("Successfully parsed and validated LLM JSON output.")
                except ValidationError as val_err:
                    print(f"Pydantic Validation Error: {val_err}")
                    identified_schema_obj = IdentifiedSchema(
                        relevant_collection=llm_output_dict.get("relevant_collection"),
                        relevant_fields=llm_output_dict.get("relevant_fields", []),
                        reasoning=f"LLM response parsed as JSON, but failed Pydantic validation: {val_err}. Raw JSON tried: {cleaned_response_text}",
                        raw_llm_response=raw_llm_text
                    )
            except (json.JSONDecodeError, AttributeError) as json_err:
                print(f"Warning: LLM response was not valid JSON: {json_err}")
                identified_schema_obj = IdentifiedSchema(
                    relevant_collection=None,
                    relevant_fields=[],
                    reasoning=f"LLM did not provide valid JSON output. Error: {json_err}. Raw text: {raw_llm_text}",
                    raw_llm_response=raw_llm_text
                )

        except Exception as gen_err:
            print(f"Error during LLM content generation or processing: {traceback.format_exc()}")
            identified_schema_obj = IdentifiedSchema(
                relevant_collection=None,
                relevant_fields=[],
                reasoning=f"LLM content generation or processing failed: {str(gen_err)}",
                raw_llm_response=None
            )
        return QueryResponse(
            query=query,
            identified_schema=identified_schema_obj,
            retrieved_schema_context=relevant_schema_chunks
        )

    except FileNotFoundError:
        print(f"Error: FAISS index file not found at '{FAISS_INDEX_PATH}'")
        raise HTTPException(status_code=404, detail=f"FAISS index file '{FAISS_INDEX_PATH}' not found. Run build_save_faiss_index first.")
    except pickle.UnpicklingError as pkl_err:
        print(f"Error unpickling FAISS index from '{FAISS_INDEX_PATH}': {pkl_err}")
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index from {FAISS_INDEX_PATH}. File might be corrupted. Regenerate it.")
    except RuntimeError as fe:
        print(f"Runtime error encountered: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Runtime error during processing: {str(fe)}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in identify_collection_fields: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")  
   