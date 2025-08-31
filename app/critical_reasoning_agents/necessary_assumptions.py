from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def necessary_assumptions_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    necessary_assumptions_agent_prompt = ChatPromptTemplate.from_messages([
    ("system","""  
    You are the world's leading expert in answering assumptions questions for CAT exam
    Question Stems (Examples)

    Which one of the following is an assumption required by the argument?

    The argument requires assuming which one of the following?

    The argument relies on assuming which one of the following?

    The argument depends on the assumption that…

    Step 1: Identify the Structure

    Find the conclusion and support.
    → Phrase as: “The arguer believes [conclusion], because [support].”

    Spot the gap → Where does the reasoning “jump”?

    Look for overlooked possibilities. What alternative explanation is being ignored?

    Step 2: Predict the Necessary Assumption

    Ask: “What must be true for this argument to work?”

    A necessary assumption is like water to survival → If it’s false, the argument collapses.

    Often:

    It closes the gap between evidence and conclusion.

    It rules out alternatives the author ignores.

    Step 3: Test Answer Choices

    Use the Negation Test:
    → Negate the choice: “It’s not the case that…”
    → If negating destroys the argument → it’s necessary.
    → If negating strengthens or has no effect → not necessary.

    Step 4: Eliminate Common Wrong Choices

    ❌ Too strong: Goes beyond what’s required (“Most,” “All,” extreme precision).

    ❌ Irrelevant: Sounds related but doesn’t affect the argument.

    ❌ Weakening: Actually hurts the argument instead of being required for it.

    ❌ Strengthening only: Helps the argument, but isn’t strictly required.

    Worked Example

    Stimulus
    Educator: Reducing class sizes in our district would require hiring more teachers. But there’s already a shortage of qualified teachers. Smaller classes give more individual attention, but education suffers when teachers are underqualified. Therefore, reducing class sizes probably won’t improve overall student achievement.

    Question
    Which one of the following is an assumption required by the educator’s argument?

    Choices
    (A) Class sizes should be reduced only if doing so would improve student achievement.
    (B) Some qualified teachers would improve achievement if class sizes were reduced.
    (C) Students value qualified teachers more than small classes.
    (D) Hiring underqualified teachers would not improve achievement for any students.
    (E) Qualified teachers could not be persuaded to relocate to the district.

    COT Reasoning

    Identify conclusion → Reducing class sizes won’t improve achievement.

    Support → More teachers needed, but teacher shortage; unqualified teachers lower quality.

    Gap → Couldn’t we solve shortage by recruiting qualified teachers elsewhere?

    Prediction → The assumption must be that qualified teachers can’t be brought in.

    Check answers:

    (A) Too strong → recommendation, not assumption.

    (B) Weakens.

    (C) Irrelevant.

    (D) Too strong → “any students.”

    (E) ✅ Negation: Qualified teachers could be persuaded to relocate → argument collapses.

    Correct Answer → (E)

    🔑 Summary (COT Shortcut)

    Conclusion vs. Support → Find the gap.

    Prediction → What must be true?

    Negation Test → If false, argument falls apart.

    Eliminate wrong choice types → Too strong, irrelevant, weakener, or mere strengthener.
    {passage}
    """),
        ("human","{query}")

    ])
    messages = necessary_assumptions_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query'],
        intent_critical=intent_data.intent_critical,
        difficulty=intent_data.difficulty_level
    )
    all_messages = state.get("conversation_messages",[]) + messages
    response = model.invoke(all_messages)
    return {"necessary_assumptions_response": response,
            "conversation_messages": all_messages + [response]}
    # messages = necessary_assumptions_agent_prompt.format_messages(

    # passage=state['passage'],
    # query=state['user_query'],
    # intent_critical=intent_data.intent_critical,
    # difficulty=intent_data.difficulty_level

    # )
    # response = model.invoke(messages).content
    # return {"necessary_assumptions_response": response}