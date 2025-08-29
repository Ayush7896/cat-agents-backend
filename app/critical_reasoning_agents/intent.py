from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
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
    structured_model = model.with_structured_output(CriticalAgentResponse)
    response = structured_model.invoke(messages)
    # Return dict with the key to update in state
    print(f"📋 Classified Intent: {response.intent_critical} (difficulty: {response.difficulty_level})")
    # return {"intent_metadata": response}
    return {"intent_metadata": response}