from app.models.schemas import CATAgentState
from langchain_core.messages import  HumanMessage
from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.core.utils import build_messages_for_invoke,append_human_ai_to_history
import logging
logger = logging.getLogger(__name__)
# def synthesizer_agent_node(state: CATAgentState):
#     print("Running synthesiser agent")
#     intent_metadata = state['intent_metadata']
#     rc_type = getattr(state['intent_metadata'], 'rc_question_type', None)
#     print(f"🎯 ROUTING DEBUG:")
#     print(f"   Intent: {intent_metadata}")
#     print(f"   RC Type: {rc_type}")
#     print(f"   Full metadata: {state['intent_metadata']}")
#      # Handle both schema cases
#     if hasattr(intent_metadata, "intent"):
#         intent = intent_metadata.intent   # general graph
#     elif hasattr(intent_metadata, "intent_critical"):
#         intent = intent_metadata.intent_critical   # critical reasoning graph
#     else:
#         intent = "unknown"
#     print(f" intnet in the synthesiser agent {intent}")
#     if intent == 'reading_comprehension':
#         final_response = state.get('rc_response', '')
       
#     elif intent == 'option_elimination':
#         final_response = state.get('option_elimination_response', '')
#     elif intent == 'exam_mind_simulator':
#         final_response = state.get('exam_mind_simulator_response', '')
#     elif intent == 'critical_reasoning':
#         final_response = state.get('critical_reasoning_response', '')
#     elif intent == 'general_help':
#         final_response = state.get('general_agent_response', '')
#     else:
#         final_response = state.get('general_agent_response', '')
#     messages = state.get("conversation_messages", [])
#     if final_response:
#         messages = messages + [AIMessage(content=final_response)]
#     return {
#         "final_answer": final_response,
#         "conversation_messages": messages
#     }


def synthesizer_agent_node(state: CATAgentState):
    """
    Synthesizes the final answer for the student by combining
    the intent-specific agent response with the user's query.
    """

    logger.info("Running synthesizer agent")
    intent_metadata = state["intent_metadata"]

    # Handle both schema cases
    if hasattr(intent_metadata, "intent"):
        intent = intent_metadata.intent
    elif hasattr(intent_metadata, "intent_critical"):
        intent = intent_metadata.intent_critical
    else:
        intent = "unknown"

    logger.debug("Detected intent in synthesizer: %s", intent)

    # Extract the appropriate response
    agent_response = ""
    if intent == "reading_comprehension":
        agent_response = state.get("rc_response", "")  # string
    elif intent == "option_elimination":
        agent_response = getattr(state.get("option_elimination_response", ""), "content", "")
    elif intent == "exam_mind_simulator":
        agent_response = getattr(state.get("exam_mind_simulator_response", ""), "content", "")
    elif intent == "critical_reasoning":
        agent_response = getattr(state.get("critical_reasoning_response", ""), "content", "")
    elif intent == "general_help":
        agent_response = getattr(state.get("general_agent_response", ""), "content", "")
    else:
        agent_response = getattr(state.get("general_agent_response", ""), "content", "")

    # Safety: ensure agent_response is always a string
    if not isinstance(agent_response, str):
        agent_response = str(agent_response)

    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a final answer synthesizer. Take the agent's response and create a clear, 
         well-formatted final answer for the student. Maintain the educational value while ensuring clarity."""),
        ("human", "Agent Response: {agent_response}\n\nUser Query: {user_query}")
    ])

    try:
        messages = synthesis_prompt.format_messages(
            agent_response=agent_response,
            user_query=state["user_query"]
        )
        logger.debug("Synthesis messages created: %s", messages)

        # Build conversation context but don’t include synthesis prompt in history
        all_messages = build_messages_for_invoke(state, messages, recent_n=20)

        # Risky LLM call
        response = model.invoke(all_messages)

    except Exception:
        logger.exception("Error in synthesizer agent while invoking model")
        return {
            "final_answer": "⚠️ Sorry, I had trouble synthesizing the final answer. Please try again.",
            "conversation_messages": state.get("conversation_messages", [])
        }

    else:
        logger.info("Synthesizer agent generated final response")

        # FIXED: Use the original user message stored during intent classification
        original_user_msg = state.get("original_user_message")
        if not original_user_msg:
            original_user_msg = HumanMessage(content=state["user_query"])

        # Add only the original user query and final synthesized response to history
        new_history = append_human_ai_to_history(state, original_user_msg, response)

        return {
            "final_answer": response.content,
            "conversation_messages": new_history
        }

    finally:
        logger.debug("synthesizer_agent_node finished execution")