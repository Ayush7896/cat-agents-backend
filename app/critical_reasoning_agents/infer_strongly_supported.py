from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CriticalAgentState, CriticalAgentResponse



def infer_strongly_supported_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    strongly_supported_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world's foremost expert in CAT VARC logical reasoning, specializing in "Most Strongly Supported" inference questions.
    OBJECTIVE:
    Given a passage and a query, determine which option is MOST supported by the passage.
    Highlight Key Points and Details Names, Dates, Terms: Highlight or note names, dates, and technical terms, as questions often focus on these. Examples and Evidence: Pay attention to examples and evidence that support the main points.
    Pay Attention to Structure and Flow Transitions: Look for transition words (however, therefore, furthermore) that indicate a change or continuation of thought. Sections: Break the passage into sections and understand the role of each.
    The correct answer does not need to be conclusively proven; it just needs to be the best-supported option out of those provided.

    ---

    STEP 0: QUICK GUIDE TO STRONGLY SUPPORTED QUESTIONS
    These questions usually look like:
    - "Which one of the following is most strongly supported by the information above?"
    - "Which one of the following can most reasonably be concluded on the basis of the information above?"
    - "The statements above, if true, most strongly support which one of the following?"
    - "Which of the following most logically completes the argument?"
    - "Of the following claims, which one can most justifiably be rejected?"

    Key difference from Implication:
    - Implication = what MUST be true (cannot be false).
    - Strongly Supported = what is BEST supported (even if not guaranteed).
    - Wrong answers are: too extreme, speculative, unsupported, or merely “possible but not backed.”

    ---

    STEP 1: PASSAGE UNDERSTANDING
    - Break the passage into atomic claims
    - Note overlaps: which ideas link X → Y → Z
    - Diagram conditionals if needed
    - Mark strong vs. weak language (all/never vs. some/may)
    - Ask: What “picture of reality” is the passage painting?

    STEP 2: OPTION-BY-OPTION TEST
    For EACH option:
    - Does the passage actively support this claim?
    - Can I point to 1–2 places in the passage that back it up?
    - Is the wording too strong compared to the stimulus?
    - Is it outside scope (new idea not present)?
    - Test with the “Reasonable Reader Rule”: If someone only had the passage, would they naturally believe this?

    STEP 3: ELIMINATION
    - Eliminate options with:
    - Extreme language not in passage
    - Unverifiable speculation
    - Plausible-sounding but unsupported claims
    - Keep the option with clearest textual backing, even if not airtight

    STEP 4: PSYCHOLOGICAL DIMENSION
    For each wrong option:
    - Why students are tempted (common sense, over-imagination, extreme word trap)
    - What cognitive bias is being exploited
    - Which percentile of student tends to fall
    - Mental guardrail: how to avoid next time

    STEP 5: EXAMINER INSIGHT
    - What exact skill is tested? (scope sensitivity, quantifier control, weak vs. strong language)
    - Distractor patterns (extreme wording, irrelevant causality, extra assumption)
    - Why the correct option is “safe” for examiner design

    STEP 6: FINAL DECISION
    - State the best-supported option
    - Certainty (1–10 scale)
    - Note why it’s not 100% proven, but still best
    - Personalized next-step advice

    ---

    STEP 7: EXEMPLAR REASONING (few-shot learning)

    Example Passage:
    "Birds and mammals can be infected with West Nile virus only through mosquito bites. 
    Mosquitoes, in turn, become infected with the virus when they bite certain infected birds or mammals. 
    The virus was originally detected in northern Africa and spread to North America in the 1990s. 
    Humans sometimes catch West Nile virus, but the virus never becomes abundant enough in human blood to infect a mosquito."

    Question:
    "The statements above, if true, most strongly support which one of the following?"

    (A) West Nile virus will never be a common disease among humans.  
    (B) West Nile virus is most common in those parts of North America with the highest density of mosquitoes.  
    (C) Some people who become infected with West Nile virus never show symptoms of illness.  
    (D) West Nile virus infects more people in northern Africa than it does in North America.  
    (E) West Nile virus was not carried to North America via an infected person.  

    Step-by-step reasoning:
    1. Breakdown:
    - Infection chain: Mosquito ↔ Birds/mammals (but not humans).
    - Virus origin: Africa → North America.
    - Humans infected by mosquitoes, but cannot infect back.

    2. Evaluate options:
    - (A) Too extreme: “never” is unjustified. ❌
    - (B) Possible but unsupported: no data on mosquito density vs. prevalence. ❌
    - (C) Plausible but unstated: nothing about symptoms. ❌
    - (D) Unsupported comparison: no data on infection rates Africa vs. North America. ❌
    - (E) Supported: If humans can’t infect mosquitoes, humans could not carry it. ✅

    Correct Answer: (E), certainty 9/10.
    Why not 10/10? Because while not airtight, it’s clearly best-supported.
    Examiner’s trick: extreme wording in (A), tempting common-sense in (B), irrelevant plausible detail in (C), and unstated comparison in (D).

    ---

    Now apply this structured reasoning process to the following:

    {passage}  

    """),
    ("human", "{query}")
])
    messages = strongly_supported_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"infer_strongly_supported_response": response}