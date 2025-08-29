from app.models.schemas import CATAgentState, ToneResponse

def synthesizer_agent_node(state: CATAgentState):
    print("Running synthesiser agent")
    intent_metadata = state['intent_metadata']
    rc_type = getattr(state['intent_metadata'], 'rc_question_type', None)
    print(f"🎯 ROUTING DEBUG:")
    print(f"   Intent: {intent_metadata}")
    print(f"   RC Type: {rc_type}")
    print(f"   Full metadata: {state['intent_metadata']}")
     # Handle both schema cases
    if hasattr(intent_metadata, "intent"):
        intent = intent_metadata.intent   # general graph
    elif hasattr(intent_metadata, "intent_critical"):
        intent = intent_metadata.intent_critical   # critical reasoning graph
    else:
        intent = "unknown"
    print(f" intnet in the synthesiser agent {intent}")
    if intent == 'reading_comprehension':
        rc_response = state.get('rc_response','')
        if isinstance(rc_response, ToneResponse):
            final_response = (
                f"**Tone Analysis:**\n\n**Tone:** {rc_response.tone_of_passage}\n\n"
                f"**Explanation:** {rc_response.explanation}"
            )
        else:
            final_response = str(rc_response)
    elif intent == 'option_elimination':
        final_response = state.get('option_elimination_response', '')
    elif intent == 'exam_mind_simulator':
        final_response = state.get('exam_mind_simulator_response', '')
    elif intent == 'critical_reasoning':
        final_response = state.get('critical_reasoning_response', '')
    elif intent == 'general_help':
        final_response = state.get('general_agent_response', '')
    else:
        final_response = state.get('general_agent_response', '')
    return {"final_answer": final_response}
