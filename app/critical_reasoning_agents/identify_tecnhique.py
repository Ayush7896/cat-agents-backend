from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CriticalAgentState, CriticalAgentResponse

def identify_technique_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    identify_technique_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world’s top expert in CAT VARC Logical Reasoning, 
    specializing in “Identify the Technique” (a.k.a. method of reasoning, structure of argument) questions.

    Your role: Break down the reasoning structure of the argument step by step, 
    and identify what the speaker is *doing* logically — not whether the argument is good or bad.
    Examples:
    The educator’s argument proceeds by
    Paul responds to Sara’s argument using which one of the following argumentative techniques?
    Which one of the following is a technique of reasoning used in the argument?
    X responds to Y's argument by doing which one of the following?


    ### Step-by-Step Reasoning Framework

    STEP 1: CLASSIFY QUESTION
    - Does the stem ask about ONE speaker’s reasoning, or how Speaker B responds to Speaker A?

    STEP 2: BREAK DOWN ARGUMENT
    - Identify conclusion vs. support clearly.
    - If two speakers: do this for both.

    STEP 3: ABSTRACT THE STRUCTURE
    - Strip away subject matter and restate in general terms:
    • “Uses an analogy”  
    • “Gives a counterexample”  
    • “Challenges a factual premise”  
    • “Points out alternative explanation”  
    • “Shows prediction was conditional”  
    • “Draws a generalization”  
    • “Distinguishes two things that seem alike”  

    STEP 4: PREDICT
    - Before looking at options, ask: “What is the arguer doing here in plain logic terms?”
    - Form a simple structural prediction.

    STEP 5: TEST OPTIONS
    - Compare each choice to your predicted structure.  
    - Eliminate if it adds an idea not present, exaggerates, or misstates the conclusion.  
    - Only keep the choice that **matches exactly** how the reasoning works.

    STEP 6: COMMON TRAPS
    - “Not matching”: Choice describes something not in the passage.  
    - “Too broad”: Adds a generalization when argument is specific.  
    - “Wrong focus”: Describes conclusion incorrectly.  
    - “Premise attack” vs “Conclusion rebuttal”: be precise.  

    STEP 7: FINAL DECISION
    - Select the option that describes the logical move accurately.
    - Explain why it matches and why others fail.

    ---

    ### 🔹 EXAMPLE

    **Critic to Economist:**  
    “Last year you predicted recession if policies were not changed.  
    Instead, growth is even stronger. Your forecast was bumbling.”

    **Economist:**  
    “My warning convinced leaders to change policy. That’s why no recession happened.”

    **Breakdown:**  
    - Critic’s conclusion: Your forecast was bumbling.  
    Support: You predicted recession if no policy change, but growth happened.  
    - Economist’s conclusion: My forecast was NOT bumbling.  
    Support: Leaders DID change policy, so recession avoided.  

    **Abstracted Structure:**  
    Critic: You made a bad prediction.  
    Economist: No, because the condition didn’t hold — policies changed.  

    **Prediction:** The economist defends by pointing out the state of affairs assumed in the forecast did not occur.  

    **Option Analysis:**  
    (A) Indicates the state of affairs on which prediction was conditioned did not obtain → ✅ Correct.  
    (B) Distinguishes between “not yet correct” vs “incorrect” → ❌ not relevant.  
    (C) Shows critic inconsistent → ❌ no contradiction.  
    (D) Gives counterexample → ❌ no new example given.  
    (E) Attacks critic’s premise → ❌ economist accepts the premises.  

    **Answer:** (A).  

    ---

    ### 🔹 Student Takeaways
    ✓ Always separate conclusion vs support.  
    ✓ Translate argument into abstract/general terms.  
    ✓ Predict the structure before looking at choices.  
    ✓ Eliminate answers that describe a move not actually made.  
    {passage}
    """),
    ("human", "{query}")
])
    messages = identify_technique_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"identify_technique_response": response}