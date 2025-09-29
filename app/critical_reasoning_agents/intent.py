from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
from app.core.utils import build_messages_for_invoke
from langchain_core.messages import  HumanMessage
import logging
logger = logging.getLogger(__name__)

def classify_critical_reasoning_intent_node(state: CriticalAgentState):
    logger.info("Entered classify_critical_reasoning_intent_node")

    # Define the classification prompt
    critical_reasoning_intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert CAT VARC critical reasoning intent classifier. 
        Analyze the student's query and classify it into one of these categories:

        - Identify the conclusion 
        - Identify an entailment (implication)
        - Infer what is most strongly supported
        - Identify or infer an issue in dispute 
        - Identify the technique 
        - Identify the role 
        - Identify the principle 
        - Match the structure
        - Match principles
        - Identify a flaw
        - Match flaws
        - Necessary Assumptions
        - Sufficient Assumptions
        - Strengthen the argument
        - Weaken the argument
        - Identify what is most/least helpful to know
        - Explain
        - Resolve a conflict

        Passage: {passage}
        """),
        ("human", "{user_query}"),
    ])

    # Format prompt with passage + query
    messages = critical_reasoning_intent_prompt.format_messages(
        passage=state["passage"],
        user_query=state["user_query"]
    )
    logger.debug("Formatted messages: %s", messages)

    # Wrap with recent history
    all_messages = build_messages_for_invoke(state, messages, recent_n=4)
    logger.debug("All messages for invocation: %s", all_messages)

    # Structured output model
    structured_model = model.with_structured_output(CriticalAgentResponse)

    # Get structured classification
    structured_response = structured_model.invoke(all_messages)
    logger.info("Structured response: %s", structured_response)

    # Persist user query as HumanMessage (for chat history)
    original_user_msg = HumanMessage(content=state["user_query"])
    logger.debug("Original user message: %s", original_user_msg)

    return {
        "intent_metadata": structured_response,
        "original_user_message": original_user_msg
    }