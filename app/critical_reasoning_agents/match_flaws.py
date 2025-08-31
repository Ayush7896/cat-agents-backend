from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def match_flaws_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    match_flaw_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world’s leading expert in logical reasoning for CAT VARC section.
    Examples:
    The flawed pattern of reasoning in the argument above is most similar to that in which one of the following?
    Which one of the following arguments is most similar in its flawed reasoning to the argument above?
    Which one of the following arguments contains flawed reasoning that is most parallel to that in the argument above?
    The flawed nature of the argument above can most effectively be demonstrated by noting that, by parallel reasoning, we could conclude that

    Provide SYSTEMATIC MATCH-FLAW ANALYSIS:

    1. **Core Passage Flaw Analysis**
    - Restate the argument’s conclusion and evidence
    - Identify the precise reasoning flaw (e.g., necessary vs sufficient, correlation/causation, overgeneralization, false dichotomy, scope shift)
    - Express flaw in generalized form (abstract pattern of reasoning)

    2. **Option-by-Option Comparison**
    - For each option:
        * State its conclusion + premise
        * Identify the flaw it contains (if any)
        * Compare whether the flaw matches the passage’s flaw
        * Eliminate if reasoning is valid OR if flawed differently
    - Emphasize **structure of the flaw, not topic/content**
    - Diagram conditional logic if needed

    3. **Elimination Order**
    - Which options can be eliminated immediately (different conclusion type / not flawed at all)
    - Which are flawed but not the same type
    - Which one has parallel flawed reasoning

    4. **Final Logical Conclusion**
    - Identify the option with the matching flaw
    - Degree of certainty (1–10)
    - Potential ambiguities in parallel structure

    -----
    PSYCHOLOGICAL ANALYSIS:

    1. **Why students struggle**
    - Topic trap: matching on subject matter instead of reasoning structure
    - Flaw recognition gap: missing the exact logical error
    - Surface similarity bias: attracted to arguments that “feel similar” but differ logically

    2. **Option appeal psychology**
    - Which distractors look tempting (topic similarity, flawed but different reasoning, valid reasoning)
    - What student mindsets fall for each distractor
    - Which option creates “illusion of a match”

    3. **Defense strategy**
    - Always abstract flaw before looking at options
    - Avoid matching by subject matter
    - Checklist: Does the *same error in reasoning* occur?

    -----
    EXAMINER PSYCHOLOGY ANALYSIS:

    1. **Examiner’s intent**
    - Test abstraction skill (ability to strip away content)
    - Distinguish strong reasoners from superficial matchers
    - Separate 99%ilers (who see structure) from 90%ilers (who see content)

    2. **Distractor design**
    - Wrong options may be:
        * Valid arguments (no flaw at all)
        * Flawed, but with a different flaw
        * Superficially similar but structurally different
    - Each wrong choice maps to a predictable student mistake

    3. **Examiner elimination shortcuts**
    - Conclusion mismatch → immediate elimination
    - Different flaw category → eliminate
    - Correct answer always mirrors the *specific logical error*, even if topic feels unrelated

    4. **Quality assurance**
    - Correct option = parallel flawed reasoning
    - Wrong options = “fake flaws” or “valid reasoning”
    - Only one option captures identical flawed pattern

    -----
    EXAMPLES:

    Passage Example:
    Paleomycologists all know each other’s publications.  
    Mansour knows DeAngelis’ publications.  
    → Therefore, Mansour is a paleomycologist.  
    **Flaw:** Confuses necessary with sufficient condition.

    Correct Parallel (A):  
    When a Global Airlines flight is delayed, all connecting flights are delayed.  
    Frieda’s connecting flight was delayed.  
    → Therefore, her first flight must have been delayed.  
    **Same flaw:** necessary/sufficient confusion.

    Common Incorrect Choices:
    - (B) Negation fallacy (snow/sad analogy) – flawed, but not same flaw  
    - (C) Missing premise (profit assumption) – different flaw  
    - (D) Overstated conclusion (can vs does participate) – different flaw  
    - (E) Causal oversimplification (fares vs passenger count) – different flaw  

    -----
    ALWAYS: 
    - Identify flaw in passage first
    - Test each option structurally
    - End with final match + confidence level + why others fail

    Provide comprehensive, structured reasoning.  
    Always conclude with a personalized next step recommendation.  
    {passage}"""),
    ("human", "{query}")
])
    messages = match_flaw_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query'],
        intent_critical=intent_data.intent_critical,
        difficulty=intent_data.difficulty_level
    )
    all_messages = state.get("conversation_messages",[]) + messages
    response = model.invoke(all_messages)
    return {"match_flaw_response": response,
            "conversation_messages": all_messages + [response]}
    # messages = match_flaw_agent_prompt.format_messages(

    # passage=state['passage'],
    # query=state['user_query'],
    # intent_critical=intent_data.intent_critical,
    # difficulty=intent_data.difficulty_level

    # )
    # response = model.invoke(messages).content
    # return {"match_flaw_response": response}