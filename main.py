from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
print(search.invoke("Obama's first name?"))