from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CATAgentState, IntentAgentResponse
from langchain_core.messages import HumanMessage
import logging
logger = logging.getLogger(__name__)

def classify_intent_node(state: CATAgentState):
    """
    Classifies the user query into an intent (e.g., reading_comprehension, critical_reasoning).
    Returns structured metadata and stores the original user message for later use.
    """

    logger.info("Running the intent classification agent")

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
        
        **Keywords to watch for:** "assumption", "strengthen", "weaken", "flaw", "principle", "dispute", "entailment", 
        "follows logically", "most strongly supported", "conclude", "infer", "except", "NOT inconsistent"

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

        Passage:
        {passage}

        Based on the passage provided, classify the intent and set appropriate fields.
        """),
        ("human", "{user_query}"),
    ])

    try:
        messages = intent_classifier_prompt.format_messages(
            passage=state["passage"],
            user_query=state["user_query"]
        )
        logger.debug("Intent classification messages created: %s", messages)

        # Structured output ensures we get IntentAgentResponse
        structured_model = model.with_structured_output(IntentAgentResponse)
        structured_response = structured_model.invoke(messages)

    except Exception:
        logger.exception("Error while classifying intent")
        return {
            "intent_metadata": IntentAgentResponse(
                intent="unknown",
                rc_question_type=None,
                difficulty_level="unknown"
            ),
            "original_user_message": HumanMessage(content=state["user_query"])
        }

    else:
        logger.info(
            "📋 Classified Intent: %s (RC Type: %s, Difficulty: %s)",
            structured_response.intent,
            structured_response.rc_question_type,
            structured_response.difficulty_level,
        )

        # Store original user message for later use
        original_user_msg = HumanMessage(content=state["user_query"])

        return {
            "intent_metadata": structured_response,
            "original_user_message": original_user_msg,
        }

    finally:
        logger.debug("classify_intent_node finished execution")