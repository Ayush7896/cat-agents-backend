from app.models.schemas import CATAgentState
from langchain_core.messages import  HumanMessage
from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.core.utils import build_messages_for_invoke,append_human_ai_to_history
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
    agent_response = ""
    # Get the response based on intent
    if intent == 'reading_comprehension':
        agent_response = state.get('rc_response', '')  # This is a STRING
    elif intent == 'option_elimination':
        agent_response = state.get('option_elimination_response', '')  # This is an AIMessage
    elif intent == 'exam_mind_simulator':
        agent_response = state.get('exam_mind_simulator_response', '')  # Check what this is
    elif intent == 'critical_reasoning':
        agent_response = state.get('critical_reasoning_response', '')  # From CR subgraph
    elif intent == 'general_help':
        agent_response = state.get('general_agent_response', '')  # This is an AIMessage
    else:
        agent_response = state.get('general_agent_response', '')
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a final answer synthesizer. Take the agent's response and create a clear, 
         well-formatted final answer for the student. Maintain the educational value while ensuring clarity."""),
        ("human", "Agent Response: {agent_response}\n\nUser Query: {user_query}")
    ])
    messages = synthesis_prompt.format_messages(
        agent_response=agent_response,
        user_query=state['user_query']
    )
    
    
    # Get conversation context but don't include the synthesis prompt in history
    all_messages = build_messages_for_invoke(state, messages, recent_n=20)
    response = model.invoke(all_messages)
    
    # FIXED: Use the original user message stored during intent classification
    original_user_msg = state.get('original_user_message')
    if not original_user_msg:
        original_user_msg = HumanMessage(content=state['user_query'])
    
    # FIXED: Add only the original user query and final synthesized response to history
    new_history = append_human_ai_to_history(state, original_user_msg, response)
    
    print(f"🔍 SYNTHESIZER: Added final Q&A pair to history")
    
    return {
        "final_answer": response.content,
        "conversation_messages": new_history
    }
