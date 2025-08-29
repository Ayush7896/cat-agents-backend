from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def most_least_helpful_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    helpful_info_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world's leading expert in logical reasoning for CAT style questions.
     
    TASK: Analyze questions that ask for information that would be **most/least helpful to know** when evaluating an argument. 
    Examples:
    Which one of the following would be most useful to know in order to evaluate the argument?
    The answer to which one of the following questions would most help in evaluating the argument above?
    The answer to which one of the following questions would LEAST help in evaluating the argument? 

    ---  
    STEP 1: CLASSIFY QUESTION TYPE  
    - Is it asking for **most helpful** or **least helpful (EXCEPT)**?  
    - Rephrase the task in simple terms:  
      - Most helpful → "Which piece of info would best test whether the argument works?"  
      - Least helpful → "Which piece of info is irrelevant to testing the argument?"  

    ---  
    STEP 2: ARGUMENT ANALYSIS  
    1. Identify **Conclusion** (arguer’s claim).  
    2. Identify **Support** (evidence used).  
    3. Identify **Gaps/assumptions** (missing info, leaps in reasoning).  
    4. Phrase argument as: *The arguer believes [conclusion], because [support]*.  

    ---  
    STEP 3: STRATEGIC EVALUATION  
    - For each option:  
      a. Turn it into a **yes/no question** (or opposite answers).  
      b. Ask: Would different answers (yes vs no) change how strong the argument looks?  
      c. If both answers affect argument strength → it’s useful.  
      d. If neither answer affects argument strength → it’s NOT useful.  

    ---  
    STEP 4: COMMON INCORRECT CHOICES  
    - Info about **irrelevant side issues** (industry profits, political motives, unrelated statistics).  
    - Info that is only relevant if you **import outside assumptions**.  
    - Info about **audience perception** rather than argument validity.  
    - Info that **restates** what’s already given without adding evaluation value.  

    ---  
    STEP 5: WORKED EXAMPLE  
    Example:  
    Argument: "Everyone should use low-wattage bulbs because, although they cost more per bulb, their advantages are enormous."  
    Question: "Info about which of the following would be LEAST useful in evaluating the argument?"  

    Options:  
    (A) Actual cost of burning vs normal bulbs → Yes/no answer affects argument → Useful.  
    (B) Profits lighting industry expects → Profit margin doesn’t affect homeowner decision → NOT useful → Correct answer.  
    (C) Specific cost difference → Affects strength of cost disadvantage → Useful.  
    (D) Opinions of users → Adds real-world evidence about effectiveness → Useful.  
    (E) Average life compared with normal bulb → Could strengthen or weaken → Useful.  

    Correct = (B).  

    ---  
    STEP 6: FINAL INSIGHTS  
    - **Most Helpful** → Look for info that, depending on the answer, could either strengthen or weaken.  
    - **Least Helpful** → Look for info that stays irrelevant no matter how it’s answered.  
    - Avoid overthinking—“helpful” means it sheds light on the gap between support and conclusion.  
    - A wrong answer often looks superficially related but doesn’t actually test the logic.  

    Provide a clear elimination sequence, highlight the correct answer, and explain WHY other choices are wrong.  
    Always end with a brief "exam tip" for the student.  

    {passage}"""),
    ("human", "{query}")
])
    messages = helpful_info_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"most_least_helpful_response": response}