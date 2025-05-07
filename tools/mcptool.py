import requests
import subprocess
import json
from agno.tools import Toolkit 
from pymongo import MongoClient



class MCPClient(Toolkit):
    def __init__(self, server_type, command=None, url=None):
        super().__init__(name="mcp_client")  # Initialize the Toolkit base class
        self.server_type = server_type
        self.command = command
        self.url = url
        self.register(self.execute_query)
        self.register(self._execute_stdio)
        self.register(self._execute_sse)


    def execute_query(self, query):
        if self.server_type == "stdio":
            return self._execute_stdio(query)
        elif self.server_type == "sse":
            return self._execute_sse(query)
        else:
            raise ValueError("Unsupported server type")

    def _execute_stdio(self, query):
        """Execute query using stdio communication."""
        try:
            process = subprocess.Popen(
                [self.command] + query,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Error executing query: {stderr}")
            return json.loads(stdout)
        except Exception as e:
            print(f"Error in _execute_stdio: {e}")
            return {"error": str(e)}

    def _execute_sse(self, query):
        """Execute query using MongoDB connection string."""
        try:
            # Connect to MongoDB using the connection string
            client = MongoClient(self.url)
            db_name = "callCrm" # Extract database name from URL
            db = client[db_name]

            # Extract collection and query details
            collection_name = query.get("collection")
            filter_query = query.get("filter", {})
            projection = query.get("projection", {})

            if not collection_name:
                raise ValueError("Collection name is required in the query.")

            # Execute the query
            collection = db[collection_name]
            results = list(collection.find(filter_query, projection))
            return results

        except Exception as e:
            print(f"Error in _execute_sse: {e}")
            return {"error": str(e)}