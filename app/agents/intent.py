from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage

def classify_intent_node(state: CATAgentState):
    intent_classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert CAT VARC intent classifier. 

    **CLASSIFICATION RULES:**

    **1. READING COMPREHENSION** - Select this ONLY for direct passage understanding questions:
    - Questions about tone, mood, author's attitude
    - Questions about main idea, central theme, primary purpose  
    - Questions asking for summary or overview
    
    **Key characteristic:** These questions can be answered by READING and UNDERSTANDING the passage content directly, without complex logical reasoning.

    **2. CRITICAL REASONING** - Select this for logical analysis questions:
    - Questions about assumptions (stated/unstated)
    - Questions about strengthening or weakening arguments
    - Questions about logical flaws or fallacies
    - Questions about principles underlying arguments
    - Questions about disputes or disagreements
    - Questions about entailments or implications
    - Questions asking "what follows logically" or "most strongly supported"
    - Questions about argument structure or technique
    
    **Keywords to watch for:** "assumption", "strengthen", "weaken", "flaw", "principle", "dispute", "entailment", "follows logically", "most strongly supported", "conclude", "infer","except","NOT inconsistent"

    **3. OTHER CATEGORIES:**
    - **verbal_ability**: Grammar, vocabulary, sentence correction, para jumbles
    - **exam_mind_simulator**: Examiner psychology, question design, test-taking psychology
    - **option_elimination**: Elimination strategies, decision-making between options
    - **preparation_strategy**: Study plans, time management, resources
    - **general_help**: General CAT advice, motivation, study guidance
    - **mock_test**: Practice tests, mock exams, timed practice sessions

    **IMPORTANT INSTRUCTIONS:**
    - **ONLY** set `rc_question_type` if intent is "reading_comprehension"
    - For ALL other intents, `rc_question_type` must be null/None
    - When intent is "reading_comprehension", choose from: "tone", "main_idea", "summary"
    - Difficulty levels: "easy", "medium", "hard"

    **DECISION FRAMEWORK:**
    Ask yourself: "Does this question require logical reasoning about arguments, assumptions, or inference patterns?" 
    - If YES → critical_reasoning
    - If NO, and it's about understanding passage content → reading_comprehension
    - Otherwise → appropriate category from the list above
    {passage}
    Based on the passage provided, classify the intent and set appropriate fields."""),
            ("human", "{user_query}"),
        ])
    
    print(f"passage is {state['passage'][:100]} and query is {state['user_query']}")
    messages = intent_classifier_prompt.format_messages(
        passage = state['passage'],
        user_query = state['user_query']
        
    )

    all_messages = state.get("conversation_messages", []) + messages 
    
    structured_model = model.with_structured_output(IntentAgentResponse)
    raw_model = model  # Regular model for AIMessage

    structured_response = structured_model.invoke(all_messages)
    
    # Get AIMessage for conversation history
    ai_message = raw_model.invoke(all_messages)
    # Return dict with the key to update in state
    print(f"📋 Classified Intent: {structured_response.intent}")
    # return {"intent_metadata": response}
    return {"intent_metadata": structured_response,
            "conversation_messages": all_messages + [ai_message]}

#  "messages": [HumanMessage(content=state["user_query"]), ai_message]










# def classify_intent_node(state: CATAgentState):
#     intent_classifier_prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are an expert CAT VARC intent classifier. 
#             Analyze the student's query and classify it into one of these categories:
#          IMPORTANT: Only set rc_question_type if intent is "reading_comprehension".
#         For all other intents, rc_question_type should be null/None.
#             - reading_comprehension: reading_comprehension: Direct questions about understanding or analyzing the passage itself 
#                 (tone, main idea, summary).  
#                 These questions do NOT involve abstract reasoning about arguments, flaws, assumptions, or logic.
#             - verbal_ability: Grammar, vocabulary, sentence correction, para jumbles
#             - critical_reasoning: Questions that ask about logical structure, assumptions,        strengthening/weaking, flaws,
#                 disputes, principles, or inference that requires reasoning BEYOND just reading comprehension.
#                 Keywords often include: "assumption", "strengthen", "weaken", "flaw", "principle", "dispute", "entailment",
#                 "what follows logically", "infer most strongly supported".
#             - exam_mind_simulator: Simulating examiner's mind, question design, psychological traps
#             - option_elimination: Strategies for eliminating options, decision-making, confusing options
#             - preparation_strategy: Study plans, time management, resources
#             - general_help: Study plans, motivation, general CAT advice
#             - mock_test: Practice tests, mock exams, timed practice
         
#             IF the intent is "reading_comprehension", also identify the specific RC question type:
#             - tone: Questions about author's attitude, mood, feeling, tone
#             - main_idea: Questions about central theme, primary purpose, main point
#             - summary: Questions asking to summarize or provide overview
#             from the {passage} provided, extract the intent category.
#             Also extract:
#             - current_topic: Specific topic within the category
#             - difficulty_level: 1 (beginner) to 5 (expert)
#         """),
#         ("human", "{user_query}"),
#     ])
#     print(f"passage is {state['passage'][:100]} and query is{state['user_query']}")
#     messages = intent_classifier_prompt.format_messages(
#         passage=state['passage'],
#         user_query=state['user_query']
#     )
#     structured_model = model.with_structured_output(IntentAgentResponse)
#     response = structured_model.invoke(messages)
#     print(f"📋 Classified Intent: {response.intent} (RC Type: {response.rc_question_type})  (difficulty: {response.difficulty_level})")
#     return {"intent_metadata": response}