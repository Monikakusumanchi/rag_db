from agno.tools import Toolkit 
from agno.agent import Agent
from agno.tools.thinking import ThinkingTools
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from typing import List, Dict, Optional
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st
import requests
 
sheet_id =  os.getenv("sheet_id")
uri = os.getenv("uri")

from agno.models.groq import Groq
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from typing import List, Dict, Optional
from pymongo import MongoClient

uri = os.getenv("uri")

class MongoDBUtility(Toolkit):
    def __init__(self, uri=uri, db_name="Demo"):
        """Initialize MongoDB connection."""
        super().__init__(name="mongo_db_toolkit")
        self.uri = os.getenv("uri")
        print(f"MongoDBUtility __init__: URI from env: {self.uri}") # Debug: Show URI
        try:
            print("MongoDBUtility __init__: Trying to connect to MongoDB...")
            self.client = MongoClient(self.uri)
            print("MongoDBUtility __init__: MongoClient created.")

            self.client.admin.command('ping')  # Check connection
            print("MongoDBUtility __init__: Ping successful. MongoDB Connection Successful!")

            self.db_name = self.uri.split("/")[-1].split("?")[0]
            print(f"MongoDBUtility __init__: Database name extracted: {self.db_name}")

            self.db = self.client[db_name]
            print(f"MongoDBUtility __init__: Database object created: {self.db}")

        except Exception as e:
            print(f"MongoDBUtility __init__: ERROR Connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.db_name = None
        self.uri = os.getenv("uri")  # Access URI from environment variable
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.register(self.query_mongodb)
        self.register(self.get_collection_schema)
        self.register(self.count_documents)
        self.register(self.find_documents)
        self.register(self.aggregate_documents)

    def query_mongodb(self, collection: str, filter_query: Optional[Dict] = None, projection: Optional[Dict] = None) -> List[Dict]:
        """Executes a MongoDB find() query."""
        if not isinstance(filter_query, dict):
            raise TypeError("Query must be a dictionary")

        col = self.db[collection]
        cursor = col.find(filter_query or {}, projection)
        return list(cursor)

    def get_collection_schema(self, collection: str) -> Dict[str, str]:
        """Returns schema-like information for a collection based on sampled documents."""
        col = self.db[collection]
        sample = col.find_one()
        if sample:
            return {field: type(value).__name__ for field, value in sample.items()}

        return {"message": "No documents found in the collection."}

    def count_documents(self, collection: str, filter_query: Optional[Dict] = None) -> int:
        """Counts documents in a collection based on a filter query."""
        if not isinstance(filter_query, dict):
            raise TypeError("Filter query must be a dictionary.")

        col = self.db[collection]
        return col.count_documents(filter_query or {})

    def find_documents(self, collection: str, filter_query: Optional[Dict] = None, projection: Optional[Dict] = None) -> List[Dict]:
        """Finds documents in a MongoDB collection."""
        if not isinstance(filter_query, dict):
            raise TypeError("Filter query must be a dictionary.")

        col = self.db[collection]
        cursor = col.find(filter_query or {}, projection)
        return list(cursor)

    def aggregate_documents(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        print(f"Running aggregation on collection: {collection}, pipeline: {pipeline}")
        col = self.db[collection]
        return list(col.aggregate(pipeline))
    
