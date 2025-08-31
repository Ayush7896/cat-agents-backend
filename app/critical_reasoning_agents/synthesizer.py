from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
# synthesiser agent
def synthesizer_agent_node(state: CriticalAgentState):
    print("Running synthesiser agent")
    intent = state['intent_metadata'].intent_critical
    print(f" intent in the synthesiser agent {intent}")
    if intent == 'Identify the conclusion':
        final_response = state.get('conclusion_response', '')

    elif intent == 'Identify an entailment (also known as implication)':
        final_response = state.get('implication_response', '')

    elif intent == 'Infer what is most strongly supported':
        final_response = state.get('infer_strongly_supported_response', '')

    elif intent == 'Identify or infer an issue in dispute':
        final_response = state.get('infer_dispute_response', '')

    elif intent == 'Identify the technique':
        final_response = state.get('identify_technique_response', '')

    elif intent == 'Identify the role':
        final_response = state.get('role_response', '')

    elif intent == 'Identify the principle':
        final_response = state.get('principle_response', '')

    elif intent == 'Match the structure':
        final_response = state.get('structure_response', '')

    elif intent == 'Identify a flaw':
        final_response = state.get('flaw_response', '')

    elif intent == 'Match flaws':
        final_response = state.get('match_flaws_response', '')

    elif intent == 'Necessary Assumptions':
        final_response = state.get('necessary_assumptions_response', '')

    elif intent == 'Sufficient Assumptions':
        final_response = state.get('sufficient_assumptions_response', '')

    elif intent == 'Strengthen the argument':
        final_response = state.get('strengthen_response', '')

    elif intent == 'Weaken the argument':
        final_response = state.get('weaken_response', '')

    elif intent == 'Identify what is most/least helpful to know':
        final_response = state.get('most_least_helpful_response', '')

    elif intent == 'Explain':
        final_response = state.get('explain_response', '')

    elif intent == 'Resolve a Conflict':
        final_response = state.get('resolve_conflict_response', '')

    messages = state.get("conversation_messages", [])

    return {
        "final_answer": final_response,
        "conversation_messages": messages
    }
    
    
    
    

    # return {"final_answer": final_response}



# def synthesizer_agent_node(state: CriticalAgentState):
#     print("Running synthesiser agent")
#     intent = state['intent_metadata'].intent_critical.strip().lower()
#     print(f" intent in the synthesiser agent {intent}")

#     if intent == 'identify the conclusion':
#         final_response = state.get('conclusion_response', '')

#     elif intent == 'identify an entailment (also known as implication)':
#         final_response = state.get('implication_response', '')

#     elif intent == 'infer what is most strongly supported':
#         final_response = state.get('infer_strongly_supported_response', '')

#     elif intent == 'identify or infer an issue in dispute':
#         final_response = state.get('infer_dispute_response', '')

#     elif intent == 'identify the technique':
#         final_response = state.get('identify_technique_response', '')

#     elif intent == 'identify the role':
#         final_response = state.get('role_response', '')

#     elif intent == 'identify the principle':
#         final_response = state.get('principle_response', '')

#     elif intent == 'match the structure':
#         final_response = state.get('structure_response', '')

#     elif intent == 'identify a flaw':
#         final_response = state.get('flaw_response', '')

#     elif intent == 'match flaws':
#         final_response = state.get('match_flaws_response', '')

#     elif intent == 'necessary assumptions':
#         final_response = state.get('necessary_assumptions_response', '')

#     elif intent == 'sufficient assumptions':
#         final_response = state.get('sufficient_assumptions_response', '')

#     elif intent == 'strengthen the argument':
#         final_response = state.get('strengthen_response', '')

#     elif intent == 'weaken the argument':
#         final_response = state.get('weaken_response', '')

#     elif intent == 'identify what is most/least helpful to know':
#         final_response = state.get('most_least_helpful_response', '')

#     elif intent == 'explain':
#         final_response = state.get('explain_response', '')

#     elif intent == 'resolve a conflict':
#         final_response = state.get('resolve_conflict_response', '')

#     else:
#         final_response = "⚠️ Intent not recognized"

#     return {"final_answer": final_response}
