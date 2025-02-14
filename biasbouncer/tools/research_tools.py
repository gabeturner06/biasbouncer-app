from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(output_format="list")

async def research_tool(query: str) -> str:
    """
    Calls the DuckDuckGo search API and returns summarized results.
    """
    try:
        results = search_tool.run(query)  # Use .run() instead of .invoke()
        return results  # Directly return the search results string
    except Exception as e:
        return f"Error fetching search results: {str(e)}"