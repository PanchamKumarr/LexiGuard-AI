import os
import requests
from typing import Optional, List, Dict
from langchain_core.tools import tool

class IndianKanoonTool:
    """
    Specialized tool for searching legal documents on Indian Kanoon.
    """
    BASE_URL = "https://api.indiankanoon.org/search/"

    def __init__(self):
        self.api_key = os.getenv("INDIAN_KANOON_API_KEY")

    def search(self, 
               query: str, 
               doctypes: Optional[str] = None, 
               fromdate: Optional[str] = None, 
               todate: Optional[str] = None, 
               pagenum: int = 0) -> Dict:
        """
        Performs a search on Indian Kanoon API.
        
        Args:
            query: The search query.
            doctypes: Comma-separated document types (e.g., 'supremecourt,judgments,laws').
            fromdate: Minimum publication date in DD-MM-YYYY format.
            todate: Maximum publication date in DD-MM-YYYY format.
            pagenum: Page number (starts at 0).
        """
        if not self.api_key:
            return {"error": "INDIAN_KANOON_API_KEY not found in environment variables."}

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json"
        }

        params = {
            "formInput": query,
            "pagenum": pagenum
        }

        if doctypes:
            params["doctypes"] = doctypes
        if fromdate:
            params["fromdate"] = fromdate
        if todate:
            params["todate"] = todate

        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def format_results(self, response: Dict) -> str:
        """
        Parses the JSON response and formats it for the LLM.
        Extracts title, docsource, and headline.
        """
        if "error" in response:
            return response["error"]

        docs = response.get("docs", [])
        if not docs:
            return "No documents found for the given query."

        formatted_docs = []
        for doc in docs:
            title = doc.get("title", "No Title")
            docsource = doc.get("docsource", "Unknown Source")
            headline = doc.get("headline", "No Description")
            # Remove HTML tags from headline if any (though API docs suggest it's plain text or handled by user)
            # For simplicity, we'll keep it as is or do a basic clean.
            formatted_docs.append(f"Title: {title}\nSource: {docsource}\nSummary: {headline}\n---")

        return "\n\n".join(formatted_docs)

@tool
def indian_kanoon_search(query: str, doctypes: Optional[str] = None, fromdate: Optional[str] = None, todate: Optional[str] = None) -> str:
    """
    Searches Indian legal statutes, judgments, and laws using the Indian Kanoon API.
    Use this tool when you need real-time legislative cross-referencing or specific Indian legal cases.
    """
    ik_tool = IndianKanoonTool()
    results = ik_tool.search(query, doctypes, fromdate, todate)
    return ik_tool.format_results(results)
