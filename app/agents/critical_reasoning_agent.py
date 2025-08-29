from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from app.critical_reasoning_agents.cr_graph import workflow


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
    intent_data: IntentAgentResponse = state['intent_metadata']
    print(f">>> Entering Critical Reasoning Subgraph for intent: {intent_data.intent}")
    
    # Always run CR subgraph when this node is called
    result = workflow.invoke({
        "passage": state["passage"],
        "user_query": state["user_query"]
    })
    print(f" result keys are{result.keys()}")
    print(f" result keys are{result.values()}")
    for k in result.keys():
        if k.endswith("_response"):
            return {"critical_reasoning_response": result[k]}
    # targeted_agent = list(result.keys())[-1]
    # print(f"targeted agent is {targeted_agent}")
    # return {"critical_reasoning_response": result[targeted_agent]}