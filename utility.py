### Utility functions ###
import sys
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict
from typing import List, Optional, Literal, Union
from langgraph.graph import MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage , SystemMessage




def make_supervisor_node(llm: BaseChatModel, members: list[str], custom_system_prompt: str | None = None) -> str:
    options = ["FINISH"] + members
    default_system_prompt = (
        " You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. If the worker responds with partial information," 
        " consider calling another worker to get all the information you need. Always consider all the responses and messages"
        " before deciding who to call next, what query to send them or what to respond to the user." 
        " When you are done, or don't have any worker who could help, respond with FINISH."
        " Whenever you use an assistant, you should provide it all the information so that it does not hallucinate"
    ).format(members=", ".join(members))
    
    system_prompt = custom_system_prompt if custom_system_prompt is not None else default_system_prompt

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]
        clarification: Optional[str]

    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto_ = response["next"]

        if goto_ == "FINISH":
            goto_ = END
            return Command(goto=goto_)
        else:
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=response["clarification"], name="supervisor")
                    ]
                },
                goto=goto_,
            )


    return supervisor_node