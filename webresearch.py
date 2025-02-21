""" Web research assistant -- Can repeatedly research the web given a query """
import os
from typing import List, Literal
from utility import make_supervisor_node
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatPerplexity
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

""" Web query search agent """ 
@tool
def perplexity_search(query: str) -> str:
    """Search the web using Perplexity AI."""
    chat = ChatPerplexity(temperature=0.7, model="llama-3.1-sonar-small-128k-online")
    response = chat.invoke(query)
    return response.content

def create_search_node(state: MessagesState) -> Command[Literal["web_research_supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_search_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="web_research_supervisor",
    )

""" Extract webpage agent """ 
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

def create_web_scraper_node(state: MessagesState) -> Command[Literal["web_research_supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="web_research_supervisor",
    )


def create_web_research_graph(llm):
    #tavily_tool = TavilySearchResults(max_results=5)
    global search_agent, web_scraper_agent

    web_prompt =""" 
    Your only task is to return the data that you get using tools , Dont do any other tasks like writing linkedin posts etc
    when something varies on several factors, try to give a genral idea in ranges.
    """
    search_agent = create_react_agent(llm, tools=[perplexity_search],state_modifier=SystemMessage(content=web_prompt))

    scrap_prompt= """Your only task is to scrape webpages . 
    You have to find the answer in given webpages .Dont do anything else like writing linkedin posts etc. 
    """
    web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages],state_modifier=SystemMessage(content=scrap_prompt))

    """ Web research supervisor """ 
    custom_prompt = """
    You are a supervisor tasked with managing a conversation between web_search_agent, web_scrapper_agent and FINISH . Given a user request
    respond with the worker to act next. web_search_agent is responsible for searching the internet and returning the related information .
    web_scraper_agent can return the data contained in given url .It cant do any other task
    When you have the exact answer, or don't have any worker who could help respond with FINISH.
    """

    web_research_supervisor_node = make_supervisor_node(llm, ["web_search_agent", "web_scraper_agent"],custom_prompt)

    web_research_builder = StateGraph(MessagesState)
    web_research_builder.add_node("web_research_supervisor", web_research_supervisor_node)
    web_research_builder.add_node("web_search_agent", create_search_node)
    web_research_builder.add_node("web_scraper_agent", create_web_scraper_node)
    web_research_builder.add_edge(START, "web_research_supervisor")
    web_research_graph = web_research_builder.compile()
    return web_research_graph


""" Create a node for web research assistant """ 
def call_web_research_assistant(state: MessagesState, ) -> Command[Literal["xyro_employee_base"]]:
    llm = ChatOpenAI(model="gpt-4o")
    result = create_web_research_graph(llm).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_research_assistant")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="xyro_employee_base",
    )