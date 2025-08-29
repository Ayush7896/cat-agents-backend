from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState

def option_elimination_agent_node(state: CATAgentState):
    print(" running the option elimination agent")
    option_elimination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the world's leading expert in logical reasoning for CAT VARC section.

    Provide SYSTEMATIC LOGICAL ELIMINATION:

    1. **For each option, analyze:**
    - Logical relationship to passage content
    - Scope accuracy (too broad/narrow/just right)
    - Factual accuracy vs. passage statements
    - Logical consistency with question requirements

    2. **Elimination order (easiest to eliminate first):**
    - Which options can be eliminated immediately and why
    - Which require deeper analysis
    - Logical elimination sequence

    3. **Final logical conclusion:**
    - Which option survives logical scrutiny
    - Degree of logical certainty (1-10)
    - Potential logical ambiguities

    Use rigorous logical analysis worthy of a philosophy PhD.
    Analyze from PSYCHOLOGICAL PERSPECTIVE:

    1. **For each option, identify:**
    - What psychological appeal it has
    - What type of student mindset it targets
    - What cognitive biases it exploits
    - Why students might find it attractive

    2. **Student vulnerability analysis:**
    - Which options this student profile is most likely to choose
    - What psychological traps they might fall into
    - How their weak areas make them vulnerable

    3. **Psychological elimination strategy:**
    - How to recognize and avoid psychological traps
    - Mental frameworks to maintain objectivity
    - Confidence-building techniques during elimination
     
    EXAMINER PSYCHOLOGY ANALYSIS:

    1. **Examiner's design intent:**
    - What skill is this question really testing?
    - How examiner wants students to think
    - What separates 99%ile from 90%ile students

    2. **Distractor design psychology:**
    - Which options are "honey traps" for different skill levels
    - How examiner made wrong options attractive
    - What student mistakes each wrong option represents

    3. **Examiner elimination shortcuts:**
    - Secret patterns examiners use consistently
    - How to think like examiner during elimination
    - Examiner's "golden rules" for answer selection

    4. **Quality assurance insights:**
    - How examiner validated the correct answer
    - What makes an option "examiner-approved"
    - How to spot examiner's intended correct answer

    Reveal the secrets that only CAT question creators know.
    
    Provide comprehensive, actionable advice with specific examples.
    Always end with a personalized next step recommendation. {passage}"""),
        ("human", "{query}")
    ])
    messages = option_elimination_prompt.format_messages(
        passage=state['passage'],
        query=state['user_query']
    )
    response = model.invoke(messages).content
    return {"option_elimination_response": response}
