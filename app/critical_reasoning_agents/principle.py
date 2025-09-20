from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate
from app.core.utils import build_messages_for_invoke


def principle_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    identify_principle_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world’s top expert in CAT VARC Logical Reasoning, 
    specializing in “Identify the Principle” questions.

    Your role: Analyze arguments or situations to extract the underlying principle.  
    A principle is a **general rule, law, or value** that explains or justifies the reasoning.
    Examples:
    Which one of the following principles, if valid, most helps to justify the reasoning in the argument above?
    The journalist’s reasoning most closely conforms to which one of the following principles?
    The situation described above conforms most closely to which one of the following generalizations?
    The principle stated above, if valid, most helps to justify the reasoning in which one of the following arguments?
     
    ### Step-by-Step Reasoning Framework

    STEP 1: SPOT THE QUESTION
    - Common stems:  
    • “Which one of the following principles, if valid, most helps to justify the reasoning in the argument?”  
    • “The journalist’s reasoning most closely conforms to which one of the following principles?”  
    • “The situation described above conforms most closely to which one of the following generalizations?”  
    • “The principle stated above, if valid, most helps to justify the reasoning in which one of the following arguments?”

    STEP 2: RESTATE ARGUMENT OR SITUATION
    - If passage = argument → Identify **conclusion + support**.  
    - If passage = situation → Summarize events and relationships in your own words.  

    STEP 3: ABSTRACT TO PRINCIPLE
    - Ask: “What general rule is this specific case showing or relying on?”  
    - Convert specifics into broad, universal terms.  
    Example: “George must wear a seat belt while driving his truck” → General principle = “One must wear a seat belt while driving any vehicle.”  

    STEP 4: PREDICT
    - If possible, form a rough principle:  
    • *Situation-based* → A broad life lesson or generalization (“An organization can succeed even if members act selfishly”).  
    • *Argument-based* → A principle that connects evidence to conclusion (“If a factor reduces fatigue and increases awareness, it contributes to safety”).  

    STEP 5: MATCH TO CHOICES
    - Check each choice:  
    • Does every part of the choice map to the stimulus?  
    • Avoid choices that go beyond the scope or introduce irrelevant elements.  
    • Eliminate weakeners or neutral statements.  

    STEP 6: DIAGRAM IF NECESSARY
    - Many principle questions are conditional. If the choice says “If X, then Y,” sketch it out and test with stimulus facts.  

    ---

    ### 🔹 EXAMPLE

    **Stimulus (Situation):**  
    “Hospitals, universities, labor unions, and other institutions may well have public purposes and succeed at achieving them, even though each staff member acts only for selfish reasons.”

    **Question:**  
    Which one of the following generalizations is most clearly illustrated by the passage?  

    **Step 1: Restate in simple terms.**  
    Institutions can achieve altruistic goals even if their individual members are selfish.  

    **Step 2: Abstract to principle.**  
    General rule = “An organization can have a property that not all of its members possess.”  

    **Step 3: Check choices.**  
    (A) Some vs. all organizations → ❌ mismatch.  
    (B) An organization can have a property its members lack → ✅ matches perfectly.  
    (C) People claim altruistic motives → ❌ not in passage.  
    (D) Institutions have unintended consequences → ❌ founders not discussed.  
    (E) Instruments serving another purpose → ❌ irrelevant.  

    **Answer:** (B).  

    ---

    ### 🔹 Student Takeaways
    ✓ Principles = broad rules underlying specific arguments or cases.  
    ✓ Argument passages → principle = assumption/justification.  
    ✓ Situation passages → principle = generalization illustrated.  
    ✓ Always test: Does this choice fully map onto the passage?  
    ✓ Watch out for:  
    • Overly narrow principles (too specific).  
    • Overly broad principles (bring in outside elements).  
    • Irrelevant principles (don’t match reasoning or situation).  
    {passage}
    """),
    ("human", "{query}")
])
    
    messages = identify_principle_agent_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query'],
        intent_critical=intent_data.intent_critical,
        difficulty=intent_data.difficulty_level
    )
    all_messages = build_messages_for_invoke(state, messages, recent_n=10)
    response = model.invoke(all_messages)
    print("infer agent generated response")
# persist conversation_messages only

    return {"principle_response": response.content}
