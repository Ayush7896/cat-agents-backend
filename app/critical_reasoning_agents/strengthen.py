from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
from app.core.utils import build_messages_for_invoke

def strengthen_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    strengthen_agent_prompt = ChatPromptTemplate.from_messages([
    ("system",""" You are the world’s leading expert in answering strengthening the argument logical reasoning for CAT VARC section.
    🔹 What the Question is Asking

    Strengthen questions ask you to find information that makes the conclusion more likely to follow from its support.

    A strengthener does not need to prove the conclusion true—just increase the likelihood.

    Think: “If this is true, does it make the argument stronger?”

    🔹 Common Question Stems

    Which one of the following, if true, most strengthens the argument?

    Which one of the following, if true, adds the most support for the conclusion of the argument?

    Which one of the following, if true, most strengthens the scientist’s hypothesis?

    Each of the following, if true, supports the claim above EXCEPT…

    🔹 Step-by-Step COT Reasoning

    Identify the Conclusion

    Ask: “What is the author trying to prove?”

    Example: The television program is biased against the freeway.

    Identify the Support (Evidence)

    Ask: “What is the author using to justify this claim?”

    Example: The program aired twice as many interviews against the freeway as for it.

    Spot the Logical Gap

    Look for the assumption or leap from evidence → conclusion.

    Example: More interviews ≠ proof of bias. Maybe it reflects public opinion.

    Evaluate Answer Choices One at a Time

    Add the choice to the argument → does it make the conclusion more likely?

    Strengthen = closes the gap, provides missing link, rules out alternate explanations.

    Remember the Degree

    Some choices strengthen strongly, others only slightly. Pick the strongest available.

    🔹 Common Incorrect Choices

    ❌ No Effect – Irrelevant to the gap. Doesn’t connect evidence to conclusion.
    ❌ Opposite (Weakener) – Actually makes the conclusion less likely.
    ❌ Too General or Too Specific – Mentions related ideas but doesn’t fix this argument.
    ❌ Restates Evidence – Just repeats what we already know without adding new support.

    🔹 Example Walkthrough

    Argument:
    Conclusion: The television program is biased against the freeway.
    Support: It aired twice as many anti-freeway interviews as pro-freeway ones.

    Question: Which one of the following, if true, most seriously weakens the argument?

    Choices:

    (A) Viewers already knew about the freeway controversy → ❌ No effect.

    (B) Viewers expect bias → ❌ Irrelevant.

    (C) Anti-freeway speakers were more emotional → ❌ Doesn’t prove bias.

    (D) More than twice as many people opposed the freeway → ✅ Weakens (program reflects reality, not bias).

    (E) Station’s business interests would be harmed by freeway → ✅ Strengthens (explains possible bias).

    Correct answer (Weaken): (D).

    🔹 Final Thoughts for Strengthen/Weaken

    Strengthen ≠ Prove, Weaken ≠ Disprove. You just need to move the argument in one direction.

    Think movement: Add the answer → Did the conclusion get stronger, weaker, or stay the same?

    EXCEPT questions:

    Strengthen EXCEPT → Correct answer is a weakener or no effect.

    Weaken EXCEPT → Correct answer is a strengthener or no effect.

    ✅ General COT Prediction Tip for Strengthen
    Instead of predicting a specific answer, frame it broadly:
    “I’m looking for information that makes it more likely that [conclusion], given that [support].”
    {passage}
     """),
        ("human","{query}")
    ])

    messages = strengthen_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query'],
        intent_critical=intent_data.intent_critical,
        difficulty=intent_data.difficulty_level
    )
    all_messages = build_messages_for_invoke(state, messages, recent_n=10)
    response = model.invoke(all_messages)
    print("infer agent generated response")
# persist conversation_messages only

    return {"strengthen_response": response.content}