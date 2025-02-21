""" Python coding assistant """ 
""" Can code in python - especially useful for mathematical operations which LLMs can be bad at """ 

from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage , SystemMessage
from utility import make_supervisor_node
from langchain_openai import ChatOpenAI


""" Python execution tool """ 
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"


"""Python coding agent """ 
def create_python_coding_node(state: MessagesState) -> Command[Literal["python_coding_supervisor"]]:
    result = python_coding_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="python_coding_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="python_coding_supervisor",
    )


def create_python_coding_graph(llm):
    global python_coding_agent
    python_coding_agent = create_react_agent(llm, tools=[python_repl_tool])
    """ Python coding supervisor """ 
    python_coding_supervisor_node = make_supervisor_node(llm, ["python_coding_agent"])
    python_coding_builder = StateGraph(MessagesState)
    python_coding_builder.add_node("python_coding_supervisor", python_coding_supervisor_node)
    python_coding_builder.add_node("python_coding_agent", create_python_coding_node())
    python_coding_builder.add_edge(START, "python_coding_supervisor")
    python_coding_graph = python_coding_builder.compile()
    return python_coding_graph


""" Create a node for coding assistant """ 
def call_python_coding_assistant(state: MessagesState) -> Command[Literal["xyro_employee_base"]]:
    llm = ChatOpenAI(model="gpt-4o")
    result = create_python_coding_graph(llm).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="python_coding_assistant")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="xyro_employee_base",
    )
