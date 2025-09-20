from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from app.models.schemas import CATAgentState, CriticalAgentState

from typing import Union

def sanitize_conversation_history(history: list[BaseMessage] | None) -> list[BaseMessage]:
    """Return only human + assistant messages (drop system messages)."""
    history = history or []
    return [m for m in history if isinstance(m, (HumanMessage, AIMessage))]

def get_recent_history(state: Union[CATAgentState, CriticalAgentState], n: int = 10) -> list[BaseMessage]:
    """Return last n human/ai messages from state in chronological order."""
    hist = sanitize_conversation_history(state.get("conversation_messages", []))
    return hist[-n:] if hist else []

def build_messages_for_invoke(state: Union[CATAgentState, CriticalAgentState], 
                             prompt_messages: list[BaseMessage], 
                             recent_n: int = 10) -> list[BaseMessage]:
    """
    Build the final messages list to pass to model.invoke:
    [system_messages...] + [last N human/ai turns] + [current human msg(s) / other prompt non-system messages]
    """
    system_msgs = [m for m in prompt_messages if isinstance(m, SystemMessage)]
    non_system_msgs = [m for m in prompt_messages if not isinstance(m, SystemMessage)]
    recent = get_recent_history(state, n=recent_n)
    return system_msgs + recent + non_system_msgs

def append_human_ai_to_history(state: Union[CATAgentState, CriticalAgentState], 
                               human_msg: HumanMessage, 
                               ai_msg: AIMessage) -> list[BaseMessage]:
    """Append the current human+ai messages to the sanitized conversation history and return it."""
    hist = sanitize_conversation_history(state.get("conversation_messages", []))
    hist.extend([human_msg, ai_msg])
    return hist