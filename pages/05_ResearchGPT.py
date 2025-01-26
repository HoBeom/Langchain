import os 
import requests
from typing import Any, Type
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.memory import ConversationSummaryBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain.prompts import PromptTemplate


# 스트림릿 설정
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍",
)

st.title("ResearchGPT")

st.markdown(
    """
Welcome to **ResearchGPT**! 🎓

This intelligent assistant is designed to help you find information from various sources like Wikipedia, DuckDuckGo, Arxiv, and Google. It keeps track of your conversation and uses previous search history to provide more informed responses.
"""
)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "summary" not in st.session_state:
    st.session_state["summary"] = ""

# Sidebar for API key and tool selection
with st.sidebar:
    st.subheader("OpenAI API Key")
    key = st.text_input("API Key", type="password")
    if "key_test" not in st.session_state:
        st.session_state["key_test"] = False
    if key != "":
        if not st.session_state["key_test"]:
            try:
                llm_test = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=key,
                    temperature=0.7,
                    max_tokens=10,
                )
                llm_test.predict("Test connection.")
                st.success("✅ API Key is valid!")
                st.session_state["key_test"] = True
            except Exception:
                st.error(f"❌ Invalid API Key")
                st.session_state["key_test"] = False
        else:
            st.success("✅ API Key is valid!")
    st.subheader("Search Tools")
    search_tools = st.multiselect(
        "Select search tools to use:",
        ["DuckDuckGo", "Wikipedia", "Arxiv", "Google"],
        default=["DuckDuckGo", "Wikipedia"],
    )

    st.subheader("Prompt Style")
    prompt_style = st.selectbox(
        "Choose your prompt style:",
        ["Concise", "Detailed", "Creative"],
    )

if not st.session_state["key_test"]:
    st.markdown(
        """
        ##### Features:
        - Choose your preferred search tools: **Wikipedia**, **DuckDuckGo**, **Arxiv**, and **Google**.
        - Generate concise, detailed, or creative responses based on your style preference.
        - Retain and reference previous search history for improved accuracy.

        Get started by selecting your tools in the sidebar and asking a question! 🚀
        """
    )
    st.error("⚠️ Please enter a valid OpenAI API Key to proceed.")
    st.stop()
elif st.session_state["messages"] == []:
    st.success("✅ API Key is valid!")


class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
    Use this tool to perform web searches using the DuckDuckGo search engine.
    It takes a query as an argument.
    Example query: "Latest technology news"
    """
    
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema
    def _run(self, query: str):
        try:
            search = DuckDuckGoSearchResults(
                requests_kwargs={"timeout": 10}
            )
            return search.run(query)
        except Exception as e:
            return f"Error during DuckDuckGo search: {str(e)}"

class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
    Use this tool to perform searches on Wikipedia.
    It takes a query as an argument.
    Example query: "Artificial Intelligence"
    """
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query) -> Any:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)


class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")


class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = """
    If you found the website link in DuckDuckGo,
    Use this to get the content of the link for my research.
    """
    args_schema: Type[WebScrapingToolArgsSchema] = WebScrapingToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        return text

class GoogleSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The search query to perform on Google.")
    num_results: int = Field(
        default=5, description="The number of search results to fetch."
    )


class GoogleSearchTool(BaseTool):
    name = "GoogleSearchTool"
    description = """
    Use this tool to perform searches on Google. 
    It takes a query as an argument and returns search results.
    """
    args_schema: Type[GoogleSearchToolArgsSchema] = GoogleSearchToolArgsSchema

    def _run(self, query: str, num_results: int = 5):
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not cse_id:
            return "Google API Key or CSE ID not configured."
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&num={num_results}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get("items", [])
            return "\n".join(
                f"Title: {item['title']}\nLink: {item['link']}"
                for item in results
            )
        except Exception as e:
            return f"Error during Google Search: {e}"


class ArxivSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The topic or keywords to search for in arXiv.")
    max_results: int = Field(
        default=5, description="The maximum number of results to fetch from arXiv."
    )


class ArxivSearchTool(BaseTool):
    name = "ArxivSearchTool"
    description = "Search for academic papers in arXiv."
    args_schema: type = ArxivSearchToolArgsSchema

    def _run(self, query: str, max_results: int = 5):
        try:
            # Construct the arXiv API query
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
            response = requests.get(url)

            if response.status_code == 200:
                # Parse the response
                entries = []
                for entry in response.text.split("<entry>")[1:]:
                    title_start = entry.find("<title>") + len("<title>")
                    title_end = entry.find("</title>")
                    summary_start = entry.find("<summary>") + len("<summary>")
                    summary_end = entry.find("</summary>")
                    link_start = entry.find("<id>") + len("<id>")
                    link_end = entry.find("</id>")

                    title = entry[title_start:title_end].strip()
                    summary = entry[summary_start:summary_end].strip()
                    link = entry[link_start:link_end].strip()

                    entries.append(f"Title: {title}\nSummary: {summary}\nLink: {link}")

                return "\n\n".join(entries)
            else:
                return f"Error: Unable to fetch data from arXiv. Status code: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize selected tools
tools = [WebScrapingTool()]
if "DuckDuckGo" in search_tools:
    tools.append(DuckDuckGoSearchTool())
if "Wikipedia" in search_tools:
    tools.append(WikipediaSearchTool())
if "Google" in search_tools:
    tools.append(GoogleSearchTool())
if "Arxiv" in search_tools:
    tools.append(ArxivSearchTool())

concise_system_message = SystemMessage(
    content="""
        You are a research expert.

        Your task is to use Wikipedia, DuckDuckGo, Google, or Arxiv to gather comprehensive and accurate information about the query provided.

        Use the available tools to find relevant and concise information. If similar questions have been asked before, leverage previous search history and results to avoid redundant searches.

        Provide a concise and clear answer, focusing on the most relevant details. Include references and links (URLs) for all sources used.
        """
)
detailed_system_message = SystemMessage(
    content="""
        You are a research expert.

        Your task is to use Wikipedia, DuckDuckGo, Google, and Arxiv to gather comprehensive and accurate information about the query provided.

        Use the available tools to find relevant and detailed information. If similar questions have been asked before, refer to previous search history and results to build on existing findings and avoid redundant searches.

        Provide a detailed, well-organized answer, including background context and insights from multiple sources. Include references and links (URLs) for all sources used.
        """
)
creative_system_message = SystemMessage(
    content="""
        You are a research expert with a creative flair.

        Your task is to use Wikipedia, DuckDuckGo, Google, and Arxiv to gather comprehensive and accurate information about the query provided.

        Use the available tools to find relevant and creative insights. If similar questions have been asked before, refer to previous search history and results to add depth to your answer and avoid redundant searches.

        Provide an engaging, creative response using storytelling, analogies, or unique examples, while maintaining factual accuracy. Include references and links (URLs) for all sources used.
        """
)
# Prompt templates for different styles
temperatures = {
    "Concise": 0.9,
    "Detailed": 0.6,
    "Creative": 0.6,
}
system_messages = {
    "Concise": concise_system_message,
    "Detailed": detailed_system_message,
    "Creative": creative_system_message,
}

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=key,
    temperature=temperatures[prompt_style],
)

# TODO: Implement memory
# memory = ConversationSummaryBufferMemory(
#     llm=ChatOpenAI(
#         openai_api_key=key, 
#         # model="gpt-4o-mini", 
#         model="gpt-3.5-turbo-1106", # gpt-4o-mini not supported in current version
#         temperature=0.9),
#     memory_key="message_history",
#     max_token_limit=400,
#     return_messages=True,
#     )

summary_prompt = PromptTemplate.from_template(
    """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

    EXAMPLE
    Current summary:
    The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

    New lines of conversation:
    Human: Why do you think artificial intelligence is a force for good?
    AI: Because artificial intelligence will help humans reach their full potential.

    New summary:
    The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
    END OF EXAMPLE

    Current summary: {summary}

    New lines of conversation:
    Human: {input}
    AI: {output}
    """
)

# Function to manage memory and generate updated summaries
def load_memory(history_max=2):
    """
    Load and manage the message history.
    If the history exceeds the allowed limit, summarize older messages and keep the most recent ones.
    
    Args:
        history_max (int): Maximum number of recent human-AI message pairs to keep.
    
    Returns:
        tuple: A tuple containing:
            - Updated summary of older messages.
            - Recent message history formatted as a string.
    """
    messages = st.session_state.get("messages", [])
    summary = st.session_state.get("summary", "")
    messages = messages[:-1]
    if not messages:
        st.session_state["summary"] = ""
        return "", ""

    if len(messages) <= history_max * 2:
        recent_message_text = "\n".join(
            f"{msg['role']}: {msg['message']}" for msg in messages
        )
        return summary, recent_message_text

    summary_index = len(messages) - history_max * 2
    new_summary = llm.predict(
        summary_prompt.format(
            summary=summary,
            input=messages[summary_index - 2]["message"],
            output=messages[summary_index - 1]["message"],
        )
    )
    st.session_state["summary"] = new_summary
    st.session_state["messages"] = messages[-history_max * 2:]
    recent_message_text = "\n".join(
        f"{msg['role']}: {msg['message']}" for msg in st.session_state["messages"]
    )
    return new_summary, recent_message_text


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

prompt = PromptTemplate.from_template(
    """
    {system_message}

    Message Summary:
    {summary}
    Message History: 
    {message_history}
    Query: {query}
    """
)


def invoke_agent(query):
    summary, message_history = load_memory()
    system_message = system_messages[prompt_style]
    prompt_text = prompt.format(
        system_message=system_message,
        summary=summary,
        message_history=message_history,
        query=query,
    )
    response = agent.run(prompt_text)
    return response

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

paint_history()
message = st.chat_input("Ask your question here...")
if message:
    send_message(message, "human")
    response = invoke_agent(message)
    send_message(response, "ai")