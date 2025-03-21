from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import trafilatura
import streamlit as st

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=5)
search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

async def research_tool(query: str) -> str:
    """
    Calls the DuckDuckGo search API and returns summarized results.
    """
    with st.spinner("Searching the Web"):
        try:
            results = search_tool.invoke(query)
            return results  # Directly return the search results string
        except Exception as e:
            return f"Error fetching search results: {str(e)}"
    

async def scrape_webpage_tool(url: str) -> dict:
    with st.spinner("Reading Web Pages"):
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {"error": "Failed to download content."}
            
            extracted_text = trafilatura.extract(downloaded)
            if not extracted_text:
                return {"error": "Could not extract meaningful content."}
            
            # Trim content to 4000 characters
            trimmed_text = extracted_text[:4000]
            
            return {"url": url, "content": trimmed_text}    
        except Exception as e:
            return {"error": f"Scraping failed: {str(e)}"}