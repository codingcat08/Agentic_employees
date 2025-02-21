""" Interlal RAG assistant """ 
""" Can query internal documentation for information about products """ 
import os
from typing import Annotated, Literal
from utility import make_supervisor_node
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
#from pinecone import Message
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.tools import tool
pc = Pinecone(os.getenv("PINECONE_API_KEY"))


""" interal RAG tool """ 
@tool
def internal_rag_tool(
    query: Annotated[str, "The query on internal product information that will be answered by this chat assistant."],
) -> str:
    """ Use this for internal information retrieval. The call is to a RAG service setup with pinecone """
    try:
        assistant = pc.assistant.Assistant(assistant_name="example-assistant-3")
        msg = Message(role="user", content=query)
        resp = assistant.chat(messages=[msg])
        return resp
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


def create_internal_rag_node(state: MessagesState) -> Command[Literal["internal_rag_supervisor"]]:
    result = internal_rag_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="internal_rag_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="internal_rag_supervisor",
    )

def create_internal_rag_graph(llm):
    """ Internal RAG agent """ 
    global internal_rag_agent
    system_prompt="""
        You have only one task that you should retrieve whatever information related to the query is in internal documents.
        Just return the information you have and dont do any other tasks like writing Linkedin Posts etc
    """
    internal_rag_agent = create_react_agent(llm, tools=[internal_rag_tool],state_modifier=SystemMessage(content=system_prompt))

    """ Internal RAG supervisor """ 
    rag_supervisor_prompt = """ 
    You are a supervisor tasked with choosing between internal_rag_agent and FINISH. Given a user request
    respond with the worker to act next. Given a user query you should ask the internal_rag_agent to 
    When you have the exact answer, or don't have any worker who could help respond with FINISH.

    """
    internal_rag_supervisor_node = make_supervisor_node(llm, ["internal_rag_agent"],rag_supervisor_prompt)
    internal_rag_builder = StateGraph(MessagesState)
    internal_rag_builder.add_node("internal_rag_supervisor", internal_rag_supervisor_node)
    internal_rag_builder.add_node("internal_rag_agent", create_internal_rag_node)
    internal_rag_builder.add_edge(START, "internal_rag_supervisor")
    internal_rag_graph = internal_rag_builder.compile()
    return internal_rag_graph

""" Create a node for RAG assistant """ 
def call_internal_rag_assistant(state: MessagesState) -> Command[Literal["xyro_employee_base"]]:
    llm = ChatOpenAI(model="gpt-4o")
    result = create_internal_rag_graph(llm).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="internal_rag_assistant")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="xyro_employee_base",
    )