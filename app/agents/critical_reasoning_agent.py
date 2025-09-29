from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from app.critical_reasoning_agents.cr_graph import workflow_critical
import logging
logger = logging.getLogger(__name__)

# def critical_reasoning_agent_node(state: CATAgentState):
#     """Run CR subgraph and return its synthesizer's result."""
#     intent_data: IntentAgentResponse = state['intent_metadata']
#     print(intent_data)
#     if intent_data.rc_question_type == None:
#         print(">>> Entering Critical Reasoning Subgraph")
#         result = workflow.invoke({
#             "passage": state["passage"],
#             "user_query": state["user_query"]
#         })
#         # CR workflow guarantees final_answer exists
#         targeted_agent = list(result.keys())[-1]
#         return {"critical_reasoning_response": result[targeted_agent]}


def critical_reasoning_agent_node(state: CATAgentState):
    """
    Handles critical reasoning queries by invoking the critical reasoning workflow.
    Always returns a structured response dictionary.
    """
    intent_data: IntentAgentResponse = state["intent_metadata"]
    logger.info(">>> Entering Critical Reasoning Subgraph for intent: %s", intent_data.intent)

    try:
        result = workflow_critical.invoke({
            "passage": state["passage"],
            "user_query": state["user_query"]
        })
        logger.debug("Critical reasoning workflow returned keys: %s", list(result.keys()))

        # Look for any *_response in the result
        for k, v in result.items():
            if k.endswith("_response"):
                logger.info("Critical reasoning agent produced response key: %s", k)
                return {"critical_reasoning_response": v}

    except Exception:
        logger.exception("Error in critical_reasoning_agent_node")
        return {"critical_reasoning_response": "Sorry, I had trouble analyzing this reasoning question."}

    else:
        logger.warning("No *_response key found in critical reasoning workflow result")
        return {"critical_reasoning_response": "No valid reasoning response was generated."}

    finally:
        logger.debug("critical_reasoning_agent_node finished execution")
