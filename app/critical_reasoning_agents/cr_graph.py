from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.synthesizer import synthesizer_agent_node
from app.critical_reasoning_agents.conclusion import conclusion_agent_node
from app.critical_reasoning_agents.explain import explain_agent_node
from app.critical_reasoning_agents.flaw import flaw_agent_node
from app.critical_reasoning_agents.general import general_agent_node
from app.critical_reasoning_agents.identify_tecnhique import identify_technique_agent_node
from app.critical_reasoning_agents.implication import implication_agent_node
from app.critical_reasoning_agents.infer_dispute import infer_dispute_agent_node
from app.critical_reasoning_agents.infer_strongly_supported import infer_strongly_supported_agent_node
from app.critical_reasoning_agents.intent import classify_critical_reasoning_intent_node
from app.critical_reasoning_agents.match_flaws import match_flaws_agent_node
from app.critical_reasoning_agents.most_least_helpful import most_least_helpful_agent_node
from app.critical_reasoning_agents.necessary_assumptions import necessary_assumptions_agent_node
from app.critical_reasoning_agents.principle import principle_agent_node
from app.critical_reasoning_agents.resolve_conflict import resolve_conflict_agent_node
from app.critical_reasoning_agents.role import role_agent_node
from app.critical_reasoning_agents.strengthen import strengthen_agent_node
from app.critical_reasoning_agents.structure import structure_agent_node
from app.critical_reasoning_agents.sufficient_assumptions import sufficient_assumptions_agent_node
from app.critical_reasoning_agents.weaken import weaken_agent_node

def route_based_on_intent(state: CriticalAgentState):
    intent = state['intent_metadata'].intent_critical
    print(f"🎯 Intent in route based on intent function: {intent}")

    # Normalize verbose → short key
    mapping = {
        "Identify the conclusion": "conclusion",
        "Identify an entailment (also known as implication)": "implication",
        "Infer what is most strongly supported": "infer_strongly_supported",
        "Identify or infer an issue in dispute": "infer_dispute",
        "Identify the technique": "identify_technique",
        "Identify the role": "role",
        "Identify the principle": "principle",
        "Match the structure": "structure",
        "Identify a flaw": "flaw",
        "Match flaws": "match_flaws",
        "Necessary Assumptions": "necessary_assumptions",
        "Sufficient Assumptions": "sufficient_assumptions",
        "Strengthen the argument": "strengthen",
        "Weaken the argument": "weaken",
        "Identify what is most/least helpful to know": "most_least_helpful",
        "Explain": "explain",
        "Resolve a conflict": "resolve_conflict",
    }

    normalized_intent = mapping.get(intent, "general_help")

    intent_to_agent = {
        "conclusion": "conclusion_agent",
        "implication": "implication_agent",
        "infer_strongly_supported": "infer_strongly_supported_agent",
        "infer_dispute": "infer_dispute_agent",
        "identify_technique": "identify_technique_agent",
        "role": "role_agent",
        "principle": "principle_agent",
        "structure": "structure_agent",
        "flaw": "flaw_agent",
        "match_flaws": "match_flaws_agent",
        "necessary_assumptions": "necessary_assumptions_agent",
        "sufficient_assumptions": "sufficient_assumptions_agent",
        "strengthen": "strengthen_agent",
        "weaken": "weaken_agent",
        "most_least_helpful": "most_least_helpful_agent",
        "explain": "explain_agent",
        "resolve_conflict": "resolve_conflict_agent",
        "general_help": "general_agent",
    }

    selected_agent = intent_to_agent.get(normalized_intent, "general_help")
    print(f"🎯 Intent: {intent} → Normalized: {normalized_intent} → Routing to: {selected_agent}")
    # return selected_agent
    return normalized_intent

checkpointer = InMemorySaver()

graph = StateGraph(CriticalAgentState)

# nodes
graph.add_node("classify_intent", classify_critical_reasoning_intent_node)
graph.add_node("conclusion_agent",conclusion_agent_node)
graph.add_node("implication_agent",implication_agent_node)
graph.add_node("infer_strongly_supported_agent",infer_strongly_supported_agent_node)
graph.add_node("infer_dispute_agent",infer_dispute_agent_node)
graph.add_node("identify_technique_agent",identify_technique_agent_node)
graph.add_node("role_agent",role_agent_node)
graph.add_node("principle_agent",principle_agent_node)
graph.add_node("structure_agent",structure_agent_node)
graph.add_node("flaw_agent",flaw_agent_node)
graph.add_node("match_flaws_agent",match_flaws_agent_node)
graph.add_node("necessary_assumptions_agent",necessary_assumptions_agent_node)
graph.add_node("sufficient_assumptions_agent",sufficient_assumptions_agent_node)   
graph.add_node("strengthen_agent",strengthen_agent_node)
graph.add_node("weaken_agent",weaken_agent_node)
graph.add_node("most_least_helpful_agent",most_least_helpful_agent_node)
graph.add_node("explain_agent",explain_agent_node)
graph.add_node("resolve_conflict_agent",resolve_conflict_agent_node)
graph.add_node("general_agent",general_agent_node)
graph.add_node("synthesizer_agent", synthesizer_agent_node)



# edges
graph.add_edge(START, "classify_intent")


# conditional_edges
graph.add_conditional_edges(
    source = 'classify_intent',
    path = route_based_on_intent,
    path_map = {
        "conclusion": "conclusion_agent",
        "implication": "implication_agent",
        "infer_strongly_supported": "infer_strongly_supported_agent",
        "infer_dispute": "infer_dispute_agent",
        "identify_technique": "identify_technique_agent",
        "role": "role_agent",
        "principle": "principle_agent",
        "structure": "structure_agent",
        "flaw": "flaw_agent",
        "match_flaws": "match_flaws_agent",
        "necessary_assumptions": "necessary_assumptions_agent",
        "sufficient_assumptions": "sufficient_assumptions_agent",
        "strengthen": "strengthen_agent",
        "weaken": "weaken_agent",
        "most_least_helpful": "most_least_helpful_agent",
        "explain": "explain_agent",
        "resolve_conflict": "resolve_conflict_agent",
        "general_help": "general_agent"
    }
)

# synthesier agent
graph.add_edge("conclusion_agent", "synthesizer_agent")
graph.add_edge("implication_agent", "synthesizer_agent")
graph.add_edge("infer_strongly_supported_agent", "synthesizer_agent")
graph.add_edge("infer_dispute_agent", "synthesizer_agent")
graph.add_edge("identify_technique_agent", "synthesizer_agent")
graph.add_edge("role_agent", "synthesizer_agent")
graph.add_edge("principle_agent", "synthesizer_agent")
graph.add_edge("structure_agent", "synthesizer_agent")
graph.add_edge("flaw_agent", "synthesizer_agent")
graph.add_edge("match_flaws_agent", "synthesizer_agent")
graph.add_edge("necessary_assumptions_agent", "synthesizer_agent")
graph.add_edge("sufficient_assumptions_agent", "synthesizer_agent")
graph.add_edge("strengthen_agent", "synthesizer_agent")
graph.add_edge("weaken_agent", "synthesizer_agent")
graph.add_edge("most_least_helpful_agent", "synthesizer_agent")
graph.add_edge("explain_agent", "synthesizer_agent")
graph.add_edge("resolve_conflict_agent", "synthesizer_agent")
graph.add_edge("general_agent", "synthesizer_agent")

graph.add_edge("synthesizer_agent", END)


# compile
workflow = graph.compile(checkpointer = checkpointer)
# return graph.compile(checkpointer = checkpointer)