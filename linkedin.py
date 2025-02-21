from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Command

llm = ChatOpenAI(model="gpt-4o")

# Move agent definitions to global scope
hashtag_system_prompt = """You are a LinkedIn hashtag optimization specialist.
Your role is to analyze the post content and add relevant, trending hashtags that will maximize visibility.
Ensure hashtags are naturally integrated and don't overwhelm the post."""

hashtag_optimizer_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=SystemMessage(content=hashtag_system_prompt)
)

formatter_system_prompt = """You are a LinkedIn post formatting specialist.
Your role is to ensure the post follows LinkedIn best practices for formatting:
1. Use appropriate line breaks for readability
2. Ensure proper emoji placement
3. Format lists and bullet points effectively
4. Position hashtags optimally
5. Highlight key points using appropriate formatting techniques"""

post_formatter_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=SystemMessage(content=formatter_system_prompt)
)

@dataclass
class PostConfig:
    """Configuration for LinkedIn post generation"""
    tone: str
    keywords: List[str]
    company_details: Dict[str, str]
    target_audience: str
    post_length: str
    industry: str
    call_to_action: Optional[str] = None

def extract_config_from_query(query: str) -> PostConfig:
    """Extract post configuration from user query"""
    system_prompt = """You are a post configuration expert. Analyze the user's query and extract:
    1. Tone of the post
    2. Target audience
    3. Company name and details
    4. Industry (if mentioned)
    5. Keywords (extract from context)
    6. Post length (if specified)
    7. Call to action (if specified)
    
    For unspecified parameters, use these defaults:
    - tone: professional and engaging (if not specified)
    - keywords: [innovation, technology, growth]
    - industry: Technology
    - post_length: medium
    - call_to_action: Learn more on our website
    
    Return ONLY the Python dictionary with NO code block markers, quotes, or additional formatting. Format exactly like this:
    {
        "tone": "extracted_tone",
        "keywords": ["keyword1", "keyword2"],
        "company_details": {"name": "company_name", "industry": "industry", "value_proposition": "value_prop"},
        "target_audience": "extracted_audience",
        "post_length": "medium",
        "industry": "extracted_industry",
        "call_to_action": "extracted_cta"
    }"""
    
    config_agent = create_react_agent(
        llm,
        tools=[],
        state_modifier=SystemMessage(content=system_prompt)
    )
    
    result = config_agent.invoke({
        "messages": [HumanMessage(content=f"Extract configuration from: {query}")]
    })
    
    # Clean the response content
    content = result["messages"][-1].content
    
    # Remove any markdown code block markers and clean up the string
    content = content.replace("```python", "").replace("```", "").strip()
    
    try:
        # Safely evaluate the Python dictionary from the response
        config_dict = eval(content)
        return PostConfig(**config_dict)
    except (SyntaxError, ValueError) as e:
        # Provide default configuration if parsing fails
        default_config = {
            "tone": "professional and engaging",
            "keywords": ["innovation", "technology", "growth"],
            "company_details": {
                "name": "GOOGLE",
                "industry": "Technology",
                "value_proposition": "AI-powered digital assistants for sales"
            },
            "target_audience": "business professionals",
            "post_length": "medium",
            "industry": "Technology",
            "call_to_action": "Learn more on our website"
        }
        print(f"Warning: Failed to parse agent response. Using default configuration. Error: {e}")
        return PostConfig(**default_config)

def post_config_node(state: MessagesState):
    # Get the user query from the initial message
    user_query = state["messages"][0].content
    
    # Extract configuration
    config = extract_config_from_query(user_query)
    
    # Create the content generator agent with the extracted config
    global content_generator_agent
    content_generator_agent = create_content_generator_agent(llm, config)
    
    # Return next node to visit
    return {"goto": "content_generator_agent"}

def create_content_generator_agent(llm, config: PostConfig):
    system_prompt = f"""You are a professional LinkedIn content creator. Generate posts with the following specifications:
    - Tone: {config.tone}
    - Target Audience: {config.target_audience}
    - Industry: {config.industry}
    - Post Length: {config.post_length}
    - Keywords to include: {', '.join(config.keywords)}
    - Company Details: {config.company_details}
    
    Guidelines:
    1. Maintain consistent brand voice
    2. Include relevant hashtags
    3. Optimize for engagement
    4. Include a clear call-to-action: {config.call_to_action if config.call_to_action else 'Based on post context'}
    """
    
    return create_react_agent(
        llm,
        tools=[],
        state_modifier=SystemMessage(content=system_prompt)
    )

def content_generator_node(state: MessagesState):
    result = content_generator_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(
                content=result["messages"][-1].content,
                name="content_generator_agent"
            )
        ]
    }

def hashtag_optimizer_node(state: MessagesState):
    result = hashtag_optimizer_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(
                content=result["messages"][-1].content,
                name="hashtag_optimizer_agent"
            )
        ]
    }

def post_formatter_node(state: MessagesState):
    result = post_formatter_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(
                content=result["messages"][-1].content,
                name="post_formatter_agent"
            )
        ]
    }

def create_linkedin_graph(llm):
    # Build the graph
    linkedin_post_builder = StateGraph(MessagesState)

    # Add nodes
    linkedin_post_builder.add_node("post_config", post_config_node)
    linkedin_post_builder.add_node("content_generator_agent", content_generator_node)
    linkedin_post_builder.add_node("hashtag_optimizer_agent", hashtag_optimizer_node)
    linkedin_post_builder.add_node("post_formatter_agent", post_formatter_node)
    linkedin_post_builder.add_node("END", lambda state: state)

    # Add edges
    linkedin_post_builder.add_edge("post_config", "content_generator_agent")
    linkedin_post_builder.add_edge("content_generator_agent", "hashtag_optimizer_agent")
    linkedin_post_builder.add_edge("hashtag_optimizer_agent", "post_formatter_agent")
    linkedin_post_builder.add_edge("post_formatter_agent", "END")

    # Set the entry point
    linkedin_post_builder.set_entry_point("post_config")

    # Compile the graph
    linkedin_post_graph = linkedin_post_builder.compile()

    return linkedin_post_graph

def call_linkedin_post_assistant(state: MessagesState) -> Command[Literal["xyro_employee_base"]]:
    llm = ChatOpenAI(model="gpt-4o")
    result = create_linkedin_graph(llm).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="linkedin_post_assistant")
            ]
        },
        goto="xyro_employee_base",
    )