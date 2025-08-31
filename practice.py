from typing import Annotated, Optional, Literal, TypedDict, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage


class IntentAgentResponse(BaseModel):
    intent: Literal[
        "reading_comprehension", "verbal_ability", "critical_reasoning",
        "exam_mind_simulator", "option_elimination", "preparation_strategy",
        "general_help", "mock_test"
    ]
    difficulty_level: Literal["easy", "medium", "hard"]
    rc_question_type: Optional[Literal[
        "tone", "main_idea", "summary"
    ]] = None


class ToneResponse(BaseModel):
    tone_of_passage: Literal[
        "Formal", "Informal", "Optimistic", "Pessimistic", "Serious",
        "Humorous", "Sarcastic", "Critical", "Sympathetic", "Objective",
        "Subjective", "Encouraging", "Cautious", "Joyful", "Melancholic",
        "Detached", "Enthusiastic", "Ironic", "Neutral", "Admiring",
        "Indifferent", "Cynical", "Reflective", "Nostalgic", "Regretful",
        "Didactic", "Authoritative", "Defensive", "Skeptical", "Witty"
    ]
    explanation: str = Field(..., description="Why this tone was chosen.")

class CATAgentState(TypedDict, total=False):
    passage: str
    user_query: str
    intent_metadata: IntentAgentResponse

    rc_response: Union[str, ToneResponse]
    option_elimination_response: str
    exam_mind_simulator_response: str
    critical_reasoning_response: str
    general_agent_response: str
    final_answer: str
    
    messages: Annotated[list[BaseMessage], add_messages]


class CATRequest(BaseModel):
    passage: str
    user_query: str
    thread_id: str


class CATResponse(BaseModel):
    final_answer: str

def classify_intent_node(state: CATAgentState):
    intent_classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert CAT VARC intent classifier. 

**CLASSIFICATION RULES:**

**1. READING COMPREHENSION** - Select this ONLY for direct passage understanding questions:
   - Questions about tone, mood, author's attitude
   - Questions about main idea, central theme, primary purpose  
   - Questions asking for summary or overview
   
   **Key characteristic:** These questions can be answered by READING and UNDERSTANDING the passage content directly, without complex logical reasoning.

**2. CRITICAL REASONING** - Select this for logical analysis questions:
   - Questions about assumptions (stated/unstated)
   - Questions about strengthening or weakening arguments
   - Questions about logical flaws or fallacies
   - Questions about principles underlying arguments
   - Questions about disputes or disagreements
   - Questions about entailments or implications
   - Questions asking "what follows logically" or "most strongly supported"
   - Questions about argument structure or technique
   
   **Keywords to watch for:** "assumption", "strengthen", "weaken", "flaw", "principle", "dispute", "entailment", "follows logically", "most strongly supported", "conclude", "infer","except","NOT inconsistent"

**3. OTHER CATEGORIES:**
   - **verbal_ability**: Grammar, vocabulary, sentence correction, para jumbles
   - **exam_mind_simulator**: Examiner psychology, question design, test-taking psychology
   - **option_elimination**: Elimination strategies, decision-making between options
   - **preparation_strategy**: Study plans, time management, resources
   - **general_help**: General CAT advice, motivation, study guidance
   - **mock_test**: Practice tests, mock exams, timed practice sessions

**IMPORTANT INSTRUCTIONS:**
- **ONLY** set `rc_question_type` if intent is "reading_comprehension"
- For ALL other intents, `rc_question_type` must be null/None
- When intent is "reading_comprehension", choose from: "tone", "main_idea", "summary"
- Difficulty levels: "easy", "medium", "hard"

**DECISION FRAMEWORK:**
Ask yourself: "Does this question require logical reasoning about arguments, assumptions, or inference patterns?" 
- If YES → critical_reasoning
- If NO, and it's about understanding passage content → reading_comprehension
- Otherwise → appropriate category from the list above
{passage}
Based on the passage provided, classify the intent and set appropriate fields."""),
        ("human", "{user_query}"),
    ])
    
    print(f"passage is {state['passage'][:100]} and query is {state['user_query']}")
    messages = intent_classifier_prompt.format_messages(
        passage=state['passage'],
        user_query=state['user_query']
    )
    structured_model = model.with_structured_output(IntentAgentResponse)
    response = structured_model.invoke(messages)
    ai_message = AIMessage(content=f"Intent classified: {response.intent}, "
                                      f"RC Type: {response.rc_question_type}, "
                                      f"Difficulty: {response.difficulty_level}")
    print(f"📋 Classified Intent: {response.intent} (RC Type: {response.rc_question_type}) (difficulty: {response.difficulty_level})")
   #  return {"intent_metadata": response}
    return {
        "intent_metadata": response,
        # append user + AI messages into conversation state
        "messages": state["messages"] + [
        HumanMessage(content=state["user_query"]),
        ai_message
    ]
       
    }


