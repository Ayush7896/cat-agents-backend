from typing import Annotated, Optional, Literal, TypedDict, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class IntentAgentResponse(BaseModel):
    intent: Literal[
        "reading_comprehension", "verbal_ability", "critical_reasoning",
        "exam_mind_simulator", "option_elimination", "preparation_strategy",
        "general_help", "mock_test"
    ]
    # difficulty_level: Literal["easy", "medium", "hard"]
    # rc_question_type: Optional[Literal[
    #     "tone", "main_idea", "summary"
    # ]] = None


# class ToneResponse(BaseModel):
#     tone_of_passage: Literal[
#         "Formal", "Informal", "Optimistic", "Pessimistic", "Serious",
#         "Humorous", "Sarcastic", "Critical", "Sympathetic", "Objective",
#         "Subjective", "Encouraging", "Cautious", "Joyful", "Melancholic",
#         "Detached", "Enthusiastic", "Ironic", "Neutral", "Admiring",
#         "Indifferent", "Cynical", "Reflective", "Nostalgic", "Regretful",
#         "Didactic", "Authoritative", "Defensive", "Skeptical", "Witty"
#     ]
#     explanation: str = Field(..., description="Why this tone was chosen.")

class CATAgentState(TypedDict, total=False):
    passage: str
    user_query: str
    intent_metadata: IntentAgentResponse

    # rc_response: Union[str, ToneResponse]
    rc_response: str
    option_elimination_response: str
    exam_mind_simulator_response: str
    critical_reasoning_response: str
    general_agent_response: str
    final_answer: str
    
    conversation_messages: Annotated[list[BaseMessage], add_messages]

class CriticalAgentResponse(BaseModel):
    intent_critical: Literal["Identify the conclusion", 
    "Identify an entailment (also known as implication)", 
    "Infer what is most strongly supported", 
    "Identify or infer an issue in dispute", 
    "Identify the technique", 
    "Identify the role", 
    "Identify the principle", 
    "Match the structure",
    "Identify a flaw",
    "Match flaws",
    "Necessary Assumptions",
    "Sufficient Assumptions",
    "Strengthen the argument",
    "Weaken the argument",
    "Identify what is most/least helpful to know",
    "Explain",
    "Resolve a conflict"]
    difficulty_level: Literal["easy", "medium", "hard"]


class CriticalAgentState(TypedDict, total = False):
    passage: str
    user_query: str
    intent_metadata: CriticalAgentResponse

    conclusion_response: str
    implication_response: str
    infer_strongly_supported_response: str
    infer_dispute_response: str
    identify_technique_response: str
    role_response: str
    principle_response: str
    structure_response: str
    flaw_response: str
    match_flaws_response: str
    necessary_assumptions_response: str
    sufficient_assumptions_response: str
    strengthen_response: str
    weaken_response: str
    most_least_helpful_response: str
    explain_response: str
    resolve_conflict_response: str
    general_agent_response: str
    final_answer: str

    conversation_messages: Annotated[list[BaseMessage], add_messages]


class CATRequest(BaseModel):
    passage: str
    user_query: str
    thread_id: str


class CATResponse(BaseModel):
    final_answer: str


