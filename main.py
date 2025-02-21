from langchain_openai import ChatOpenAI
from utility import make_supervisor_node
from webresearch import call_web_research_assistant
from coding import call_python_coding_assistant
from linkedin import call_linkedin_post_assistant
from prodrag import call_internal_rag_assistant
from erp import call_erp_assistant
from pathlib import Path
from tempfile import TemporaryDirectory
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
import os
import json

def process_query(query_text):
    """
    Process a query through the XyRo agent system
    """
    try:
        # Load environment variables
        load_dotenv()

        _TEMP_DIRECTORY = TemporaryDirectory()
        WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

        # Setup LLM powering the digitial employees
        llm = ChatOpenAI(model="gpt-4o")

        # Create employee base node
        xyro_employee_base_node = make_supervisor_node(llm, [
            "web_research_assistant",
            "internal_rag_assistant",
            "python_coding_assistant",
            "linkedin_post_assistant",
            "erp_assistant"
        ])

        # Build the graph
        xyro_employee_base_node_builder = StateGraph(MessagesState)
        xyro_employee_base_node_builder.add_node("xyro_employee_base", xyro_employee_base_node)
        xyro_employee_base_node_builder.add_node("web_research_assistant", call_web_research_assistant)
        xyro_employee_base_node_builder.add_node("internal_rag_assistant", call_internal_rag_assistant)
        xyro_employee_base_node_builder.add_node("python_coding_assistant", call_python_coding_assistant)
        xyro_employee_base_node_builder.add_node("linkedin_post_assistant", call_linkedin_post_assistant)
        xyro_employee_base_node_builder.add_node("erp_assistant", call_erp_assistant)
        xyro_employee_base_node_builder.add_edge(START, "xyro_employee_base")
        xyro_employee_base_graph = xyro_employee_base_node_builder.compile()

        # Process the query
        responses = []
        for response in xyro_employee_base_graph.stream(
            {
                "messages": [
                    ("user", query_text),
                ],
            },
            {
                "subgraphs": True,
                "recursion_limit": 25
            },
        ):
            responses.append(response)

        # Return all responses
        return {
            "status": "success",
            "responses": responses,
            "final_response": responses[-1] if responses else None
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # This section only runs if the file is run directly (not through Flask)
    test_query = "what is xyro's mission"
    result = process_query(test_query)
    response= {
            "response" : str(result)
        }
    print(json.dumps(response, indent=2))