from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage

def reading_comprehension_agent_node(state: CATAgentState):
 
   
    intent_data: IntentAgentResponse = state['intent_metadata']
    reading_comprehension_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert CAT Reading Comprehension tutor with 15+ years of experience.
                    Your capabilities:
                    1. Analyze RC passages and identify key themes, structure, main idea, and author's intent,
                    2. Generate main idea and summary by reading all paragraphs, key themes, structure from passage..Identify the Main Idea and Purpose First and Last Paragraphs: Often, the main idea or purpose of the passage is stated in the introduction or conclusion. Topic Sentences: Each paragraph usually starts with a topic sentence that gives a clue about the main point of that section
                    3. Pay Attention to Structure and Flow Transitions: Look for transition words (however, therefore, furthermore) that indicate a change or continuation of thought. Sections: Break the passage into sections and understand the role of each
                    4. Provide time-saving reading strategies
                    5. Generate practice questions with detailed explanations
                    6. Identify student's weakness patterns
                    Provide comprehensive, actionable advice with specific examples.
                    Always end with a personalized next step recommendation.
                    Passage to analyze: {passage}
    """),
            ("human", "{query}")
        ])
    messages = reading_comprehension_prompt.format_messages(
            
            passage=state['passage'],
            query=state['user_query'],
            
        )
    all_messages = state.get("conversation_messages", []) + messages
    response = model.invoke(all_messages)

    print("main idea response", response)

    return {"rc_response": response.content,
            "conversation_messages": all_messages + [response]}
