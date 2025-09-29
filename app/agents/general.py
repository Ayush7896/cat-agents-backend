from app.models.schemas import CATAgentState
from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState
from app.core.utils import build_messages_for_invoke
import logging
logger = logging.getLogger(__name__)

def general_agent_node(state: CATAgentState):
    logger.info(f"running the general agent")
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
    try:
        messages = general_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query']
        )
        all_messages = build_messages_for_invoke(state, messages, recent_n=20)
        response = model.invoke(all_messages)
    except Exception as e:
        # log with traceback
        logger.exception("Error in general agent while invoking model")
        # return safe fallback
        return {"general_agent_response": "Sorry, I had trouble analyzing the passage. Please try again."}
    else:
        logger.info("general agent successfully generated response")
        return {"general_agent_response": response.content}
    finally:
        logger.debug("general agent node finished execution")
