from app.models.schemas import CATAgentState
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage

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
    """Already fixed in previous version"""
    print("Running synthesiser agent")
    intent_metadata = state['intent_metadata']
    
    # Handle both schema cases
    if hasattr(intent_metadata, "intent"):
        intent = intent_metadata.intent
    elif hasattr(intent_metadata, "intent_critical"):
        intent = intent_metadata.intent_critical
    else:
        intent = "unknown"
    
    print(f"Intent in synthesiser agent: {intent}")
    
    # Get the response based on intent
    if intent == 'reading_comprehension':
        raw_response = state.get('rc_response', '')  # This is a STRING
    elif intent == 'option_elimination':
        raw_response = state.get('option_elimination_response', '')  # This is an AIMessage
    elif intent == 'exam_mind_simulator':
        raw_response = state.get('exam_mind_simulator_response', '')  # Check what this is
    elif intent == 'critical_reasoning':
        raw_response = state.get('critical_reasoning_response', '')  # From CR subgraph
    elif intent == 'general_help':
        raw_response = state.get('general_agent_response', '')  # This is an AIMessage
    else:
        raw_response = state.get('general_agent_response', '')
    
    # ✅ UNIVERSAL CONTENT EXTRACTION
    if isinstance(raw_response, AIMessage):
        final_response = raw_response.content  # Extract content from AIMessage
        print(f"✅ Extracted content from AIMessage: {final_response[:100]}...")
    elif isinstance(raw_response, str):
        final_response = raw_response  # Already a string
        print(f"✅ Using string response: {final_response[:100]}...")
    else:
        final_response = str(raw_response) if raw_response else "No response generated"
        print(f"⚠️ Converted {type(raw_response)} to string")
    
    messages = state.get("conversation_messages", [])
    
    if final_response:
        messages = messages + [AIMessage(content=final_response)]
    
    return {
        "final_answer": final_response,  # ✅ Always string for FastAPI
        "conversation_messages": messages  # ✅ Full message history preserved
    }
