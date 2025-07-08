# tools.py
import os
import requests
from langchain.tools import Tool

SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # Set this in .env or your terminal

def google_search(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "q": query
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return "❌ Failed to fetch search results."

    results = response.json().get("organic", [])
    if not results:
        return "⚠️ No relevant results found."

    output = ""
    for r in results[:3]:  # Top 3 results
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        output += f"🔹 **{title}**\n{snippet}\n🔗 {link}\n\n"

    return output.strip()

# LangChain Tool
google_tool = Tool(
    name="Google Search",
    func=google_search,
    description="Use this tool to answer questions when PDF content is insufficient."
)
