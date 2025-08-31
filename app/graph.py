from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.models.schemas import CATAgentState
from app.agents.intent import classify_intent_node
from app.agents.reading_comprehension import reading_comprehension_agent_node
from app.agents.option_elimination import option_elimination_agent_node
from app.agents.exam_mind_simulator import exam_mind_simulator_agent_node
from app.agents.general import general_agent_node
from app.agents.synthesizer import synthesizer_agent_node
from app.agents.critical_reasoning_agent import critical_reasoning_agent_node

def route_based_on_intent(state: CATAgentState):
    """
    Route the conversation based on the identified intent.
    """
    intent = state['intent_metadata'].intent
    print(f"🎯 Intent in route based on intent function: {intent}")

    # map intent to agent node name
    intent_to_agent = {
        "reading_comprehension": "reading_comprehension_agent",
        "option_elimination": "option_elimination_agent", 
        "exam_mind_simulator": "exam_mind_simulator_agent",
        "critical_reasoning": "critical_reasoning_agent",
        "general_help": "general_help_agent"
    }

    # add fallback criteria
    selected_agent = intent_to_agent.get(intent, "general_help")
    print(f"🎯 Intent: {intent} → Routing to: {selected_agent}")
    return selected_agent

checkpointer = InMemorySaver()

def build_workflow():
    graph = StateGraph(CATAgentState)

    # nodes
    graph.add_node('classify_intent', classify_intent_node)
    graph.add_node("reading_comprehension_agent", reading_comprehension_agent_node)
    graph.add_node("option_elimination_agent", option_elimination_agent_node)
    graph.add_node("exam_mind_simulator_agent", exam_mind_simulator_agent_node)
    graph.add_node("critical_reasoning_agent", critical_reasoning_agent_node)
    graph.add_node("general_help_agent", general_agent_node)
    graph.add_node("synthesizer_agent", synthesizer_agent_node)


    # start
    graph.add_edge(START, 'classify_intent')

    # conditional route
    graph.add_conditional_edges(
        source='classify_intent',
        path=route_based_on_intent,
        path_map={
            "reading_comprehension_agent": "reading_comprehension_agent",
            "option_elimination_agent": "option_elimination_agent",
            "exam_mind_simulator_agent": "exam_mind_simulator_agent",
            "critical_reasoning_agent": "critical_reasoning_agent",
            "general_help_agent": "general_help_agent",
        }
    )

    # to synthesizer
    graph.add_edge('reading_comprehension_agent', 'synthesizer_agent')
    graph.add_edge('option_elimination_agent', 'synthesizer_agent')
    graph.add_edge('exam_mind_simulator_agent', 'synthesizer_agent')
    graph.add_edge("critical_reasoning_agent", "synthesizer_agent")
    graph.add_edge('general_help_agent', 'synthesizer_agent')

    graph.add_edge('synthesizer_agent', END)

    return graph.compile(checkpointer = checkpointer)
