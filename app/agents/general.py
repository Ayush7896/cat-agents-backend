from app.models.schemas import CATAgentState, ToneResponse
from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState

def general_agent_node(state: CATAgentState):
    print(" running the general agent")
    general_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a response synthesizer for a CAT VARC tutoring system.
            
            Your job is to:
            1. Combine insights from specialist agents
            2. Ensure response coherence and flow
            3. Add personalized recommendations
            4. Suggest next steps for continued learning
            5. Maintain encouraging and motivational tone
    Create a comprehensive, personalized response that helps the student improve
    """),
    ("human", "{query}")
    ])
    messages = general_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query']
    )
    response = model.invoke(messages).content
    return {"general_agent_response": response}
