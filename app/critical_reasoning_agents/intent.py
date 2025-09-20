from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
from app.core.utils import build_messages_for_invoke
from langchain_core.messages import  HumanMessage
# from dotenv import load_dotenv

# load_dotenv()

def classify_critical_reasoning_intent_node(state: CriticalAgentState):
    critical_reasoning_intent_prompt = ChatPromptTemplate.from_messages([
       ("system", """You are an expert CAT VARC critical reasoning intent classifier. 
            Analyze the student's query and classify it into one of these categories:
            Identify the conclusion 
            Identify an entailment (also known as implication)
        
            Infer what is most strongly supported,
            Identify or infer an issue in dispute 
            Identify the technique 
            Identify the role 
            Identify the principle 
            Match the structure
            Match principles
            Identify a flaw
            Match flaws
            Necessary Assumptions
            Sufficient Assumptions
            Strengthen the argument
            Weaken the argument
            Identify what is most/least helpful to know
            Explain
            Resolve a conflict
            {passage}
            """),
        ("human", "{user_query}"),
    ])

    messages = critical_reasoning_intent_prompt.format_messages(
        passage = state['passage'],
        user_query = state['user_query']
    )
    # build invocation messages without persisting system prompts
    all_messages = build_messages_for_invoke(state, messages, recent_n=4)
    
    structured_model = model.with_structured_output(CriticalAgentResponse)
    # raw_model = model  # Regular model for AIMessage

    structured_response = structured_model.invoke(all_messages)
    
    # Get AIMessage for conversation history
    # ai_message = raw_model.invoke(all_messages)
    # extract the human message from `messages` (usually the current user query)
    # human_msg = next((m for m in messages if isinstance(m, HumanMessage)), HumanMessage(content=state['user_query']))
    original_user_msg = HumanMessage(content=state['user_query'])

    # persist only human + ai into conversation history

    return {"intent_metadata": structured_response,
            "original_user_message": original_user_msg}