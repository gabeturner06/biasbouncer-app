from langchain_community.tools import DuckDuckGoSearchResults
import trafilatura
import streamlit as st
import asyncio

search_tool = DuckDuckGoSearchResults(output_format="list")

async def research_tool(query: str) -> str:
    """
    Calls the DuckDuckGo search API and returns summarized results with a timeout.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(search_tool.run, query), timeout=10)  # 10 sec timeout
    except asyncio.TimeoutError:
        return "Error: Research operation timed out."
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