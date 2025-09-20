from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
from app.core.utils import build_messages_for_invoke


def sufficient_assumptions_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    sufficient_assumptions_agent_prompt = ChatPromptTemplate.from_messages([
    ("system","""  
    You are the world's leading expert in answering sufficent assumption questiosn for CAT exam
    Question Stems (Examples)

    The conclusion drawn above follows logically if which one of the following is assumed?

    The conclusion of the argument is strongly supported if which one of the following is assumed?

    The conclusion of X’s argument can be properly drawn if which one of the following is assumed?

    Which one of the following is an assumption that, if true, would do most to justify X’s actions?

    Step 1: Identify the Structure

    Find the conclusion and support.
    → Phrase as: “The arguer believes [conclusion], because [support].”

    Spot the leap → What is missing between evidence and conclusion?

    Step 2: Predict the Sufficient Assumption

    Ask: “What assumption, if added, would 100% guarantee the conclusion?”

    Think of it as a bridge: evidence + assumption → conclusion.

    Stronger = better (opposite of Necessary).

    Step 3: Test Answer Choices

    Use the Affirmation Test:
    → Add the choice to the support.
    → Does the conclusion now follow with certainty? ✅
    → If it still leaves room for doubt, it’s not sufficient.

    Step 4: Eliminate Common Wrong Choices

    ❌ Not enough → Helps but doesn’t guarantee.

    ❌ Necessary but not sufficient → Required, but doesn’t bridge fully.

    ❌ Irrelevant → Outside argument’s scope.

    ❌ Backwards → Goes from conclusion → evidence (wrong direction).

    Step 5: Difference vs. Necessary Assumptions

    Necessary Assumption: Must be true for argument to work; without it, argument collapses. (Weaker, “safety net.”)

    Sufficient Assumption: If true, makes the argument airtight; it’s a bridge that locks premise to conclusion. (Stronger, “steel beam.”)

    Worked Example

    Stimulus
    Activist: Any member of the city council ought either to vote against the proposal or to abstain. But if all the members abstain, the matter will be decided by the city’s voters. So at least one member of the city council should vote against the proposal.

    Question
    The conclusion of the activist's argument follows logically if which one of the following is assumed?

    Choices
    (A) If all members abstain, the city’s voters will definitely approve the proposal.
    (B) The proposal should not be decided by the city’s voters.
    (C) No members of the city council will vote in favor of the proposal.
    (D) If not every member abstains, the voters will not decide the proposal.
    (E) If one member ought to vote against, the others should abstain.

    COT Reasoning

    Conclusion → At least one council member should vote against.

    Support → If all abstain, voters decide.

    Gap → Why is it bad if voters decide?

    Prediction → Must assume: “The proposal should not be decided by the voters.”

    Check choices:

    (A) Too specific; argument only about who decides, not outcome.

    (B) ✅ Perfect: fills the gap → guarantees conclusion.

    (C) Wrong focus (what will happen, not what should).

    (D) Wrong logical equivalence.

    (E) Creates false dichotomy.

    Correct Answer → (B)

    Extra Quick Example (Contrast with Necessary)

    Argument:
    “If at least 50 register → profitable. 55 registered → so conference profitable.”

    Necessary assumption: “At least 50 people registered.” (weak, required)

    Sufficient assumption: “If 55 register, then conference profitable.” (strong, guarantees).

    🔑 Summary (COT Shortcut)

    Conclusion vs. Support → Find the missing link.

    Prediction → Strong assumption that bridges gap.

    Affirmation Test → Premises + assumption → conclusion guaranteed.

    Eliminate traps → Not enough, necessary-only, irrelevant, backwards.

    Remember the contrast → Necessary = weak but required; Sufficient = strong and guarantees.
    {passage}
    """),
        ("human","{query}")

    ])

    messages = sufficient_assumptions_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query'],
        intent_critical=intent_data.intent_critical,
        difficulty=intent_data.difficulty_level
    )
    all_messages = build_messages_for_invoke(state, messages, recent_n=10)
    response = model.invoke(all_messages)
    print("infer agent generated response")
# persist conversation_messages only

    return {"sufficient_assumptions_response": response.content}