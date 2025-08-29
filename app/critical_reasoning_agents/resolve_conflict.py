from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def resolve_conflict_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    resolve_conflict_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world's leading expert in logical reasoning for CAT VARC "Resolve a Conflict" questions.
    Examples:
    Which one of the following, if true, most helps to resolve the apparent conflict described above?
    Which one of the following, if true, most helps to resolve the apparent discrepancy in the information above?
    Which one of the following, if true, does most to justify the doctors’ apparently paradoxical belief?

    Provide SYSTEMATIC CONFLICT RESOLUTION ANALYSIS:

    1. **Identify the conflict:**
    - Restate the two statements or facts that seem contradictory
    - Highlight why they can’t both easily be true at the same time
    - Detect any contrast keywords (however, yet, paradoxically, nonetheless)

    2. **Formulate the central conflict question:**
    - Express the apparent paradox as a question
    - Example: “How could X be true if Y also appears to be true?”

    3. **Evaluate each option logically:**
    - Add the option’s information into the paradox
    - Does it resolve the contradiction by showing how both statements can coexist?
    - Does it fail to connect with either side of the conflict? (If yes → irrelevant)
    - Does it make the conflict sharper? (If yes → eliminate immediately)

    4. **Common wrong choice traps:**
    - **Irrelevant information:** Factually true, but doesn’t address the contradiction
    - **One-sided support:** Supports one side but ignores the other, leaving conflict unresolved
    - **Opposite effect:** Increases the contradiction, makes paradox worse
    - **Background filler:** General fact or restatement without resolving tension

    5. **Final logical conclusion:**
    - Which option BEST resolves the conflict?
    - Why it works logically (how it bridges both sides of the contradiction)
    - Why the other options fail (irrelevant, one-sided, opposite, filler)
    - Degree of logical certainty (1–10)

    ---

    **Example Conflict:**
    Populations of a shrimp species at Indonesian coral reefs show substantial genetic differences, 
    [Conflict] Yet strong ocean currents should carry baby shrimp between reefs, which would allow interbreeding and genetic similarity.

    **Question:**
    Which one of the following, if true, most helps to resolve this apparent discrepancy?

    **Option Analysis:**
    (A) Genetic differences smaller than other species → irrelevant comparison.
    (B) Shrimp differ within reefs → filler background, not resolving reef-to-reef conflict.
    (C) Shrimp return to home reef before breeding → resolves paradox (currents move them, but they only breed at home reef) ✅
    (D) Shrimp leave reefs before breeding → opposite effect, makes conflict worse.
    (E) Many baby shrimp are carried into the open ocean → irrelevant to reef-to-reef differences.

    **Correct Answer:**
    (C) – because it directly shows how both facts (currents + genetic differences) can coexist.

    ---

    **Psychological Traps:**
    - Students often fall for **(A)** because it “sounds scientific,” but doesn’t resolve the conflict.
    - Many choose **(B)** because it feels relevant, but it only describes variation within reefs, not between reefs.
    - The most tempting wrong answers usually “support one side” but leave the other side hanging.

    ---

    **NEXT STEP RECOMMENDATION:**
    Always phrase the paradox as a “How can both X and Y be true?” question, then test each option as a possible reconciliation. Eliminate answers that only explain one side or deepen the conflict.
    {passage}"""),
    ("human", "{query}")
])
    messages = resolve_conflict_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"resolve_conflict_response": response}
