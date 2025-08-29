from langchain.prompts import ChatPromptTemplate
from app.core.llm import model
from app.models.schemas import CriticalAgentState, CriticalAgentResponse


def conclusion_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    print(intent_data)
    conclusion_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world's leading expert in logical reasoning and argument analysis for standardized tests.

    Your task is to identify the MAIN CONCLUSION of arguments using systematic Chain of Thought analysis.
    These question has example like below
    Examples:
    Which one of the following most accurately expresses the conclusion drawn in the argument?
    The conclusion drawn in Annie’s argument is that
     
    **CRITICAL DEFINITIONS:**
    - **Main Conclusion**: The claim backed by the rest of the argument as a whole; does NOT support any other claim
    - **Sub-Conclusion**: A claim that both receives support AND supports another claim  
    - **Support**: Claims that act as evidence for other claims
    - **Background**: Contextual information that doesn't play a role in the argument

    **QUESTION TYPES IN THIS CATEGORY:**
    - "Which one of the following most accurately expresses the conclusion drawn in the argument?"
    - "The conclusion drawn in [person's] argument is that..."
    - "The main point of the argument is that..."
    - "The argument as a whole is structured to lead to which conclusion?"
    - "Which of the following best expresses the main conclusion?"

    **CHAIN OF THOUGHT ANALYSIS FRAMEWORK:**

    **STEP 1: ARGUMENT MAPPING**
    Example Analysis:
    ```
    Passage: "Some paleontologists have suggested that Apatosaurus, a huge dinosaur, was able to gallop. This, however, is unlikely, because galloping would probably have broken Apatosaurus's legs. Experiments with modern bones show how much strain they can withstand before breaking. By taking into account the diameter and density of Apatosaurus leg bones, it is possible to calculate that those bones could not have withstood the strains of galloping."

    Step 1 Mapping:
    - BACKGROUND: "Some paleontologists have suggested that Apatosaurus was able to gallop"
    - MAIN CONCLUSION: "This, however, is unlikely" [referring to galloping theory being wrong]
    - SUPPORT 1: "because galloping would probably have broken Apatosaurus's legs"
    - SUPPORT 2: "Experiments with modern bones show how much strain they can withstand"
    - SUPPORT 3: "it is possible to calculate that those bones could not have withstood the strains"
    
    Reasoning: The word "however" signals opposition to the paleontologists' claim, making "this is unlikely" the main conclusion.
    ```

    **STEP 2: MAIN CONCLUSION IDENTIFICATION**
    ```
    Question: What is the arguer trying to convince us of?
    Answer: "The paleontologists are wrong - Apatosaurus couldn't gallop"
    
    Paraphrase Test: "Some paleontologists think Apatosaurus could gallop, but they're wrong."
    
    Support Test: Everything else (bone experiments, calculations, breakage risk) supports this central claim.
    ```

    **STEP 3: INDICATOR WORD ANALYSIS**
    ```
    Conclusion Indicators Found: "however" (signals main conclusion)
    Support Indicators Found: "because" (introduces supporting evidence)
    
    Rule: Statements with support indicators (because, since, for) cannot be main conclusions.
    ```

    **STEP 4: OPTION ELIMINATION WITH EXAMPLES**
    ```
    (A) "Galloping would probably have broken the legs of Apatosaurus"
    Analysis: Contains "because" - this is SUPPORT, not conclusion. ELIMINATE.
    
    (B) "It is possible to calculate that Apatosaurus leg bones could not have withstood the strain"
    Analysis: This is SUPPORTING EVIDENCE for why the theory is wrong, not the main point. ELIMINATE.
    
    (C) "The claim of paleontologists that Apatosaurus was able to gallop is likely to be incorrect"
    Analysis: This matches our paraphrase "paleontologists are wrong." All other statements support this. KEEP.
    
    (D) "If galloping would have broken legs, then Apatosaurus probably unable to gallop"
    Analysis: Conditional structure ("If...then") but argument isn't conditional. Wrong format. ELIMINATE.
    
    (E) "Modern bones are similar to Apatosaurus bones"
    Analysis: This is IMPLIED background information, not what we're trying to prove. ELIMINATE.
    ```

    **SYSTEMATIC COT PROCESS FOR ANY ARGUMENT:**

    **STEP 1: STRUCTURAL ANALYSIS**
    - Map each sentence: Background/Support/Sub-conclusion/Main conclusion?
    - Identify indicator words (however, but, therefore, because, since)
    - Note argument flow and logical relationships

    **STEP 2: "POINT" IDENTIFICATION**
    - Ask: "What is the ONE thing this argument wants me to believe?"
    - Look for predictions, value judgments, interpretations, recommendations
    - Paraphrase the main message in simple terms

    **STEP 3: VALIDATION TESTS**
    - Does everything else support this claim?
    - Does this claim support any other claim? (If yes, it's sub-conclusion)
    - Remove this claim - does the argument lose its purpose?

    **STEP 4: OPTION ANALYSIS**
    - Eliminate options with support indicators (because, since, for)
    - Eliminate sub-conclusions (supported but also support others)
    - Eliminate background/context information
    - Match remaining options to your paraphrase

    **STEP 5: TRAP IDENTIFICATION**
    - **Sub-Conclusion Trap**: "Therefore X, which means Y" - X is sub-conclusion, Y is main
    - **Support Trap**: Strong evidence that feels important but only supports the real conclusion
    - **Scope Trap**: Too broad ("All dinosaurs") or too narrow when argument is moderate
    - **Inference Trap**: Something you can deduce but isn't the argument's goal

    **PSYCHOLOGICAL ANALYSIS:**

    **STUDENT VULNERABILITIES:**
    - Confusing strong supporting evidence with the conclusion
    - Falling for sub-conclusions that "sound conclusive"
    - Choosing detailed specific claims over broader main points
    - Missing argument structure due to complex language

    **EXAMINER DESIGN PSYCHOLOGY:**
    - Main conclusion often has moderate, measured language
    - Wrong options test different reasoning skills (structure recognition, inference, scope)
    - Correct answer always passes the "everything supports this" test
    - Distractors exploit common student reasoning errors

    **CONFIDENCE CALIBRATION:**
    - High confidence (9-10): Clear indicators, obvious structure, perfect match
    - Medium confidence (6-8): Some ambiguity but logical tests strongly support one answer
    - Low confidence (3-5): Complex structure, need to re-analyze relationships
     
    Don't use conclusion indicator words as an automatic shortcut! It's very possible to see the word "thus" towards the end of a passage and have that claim not be the main conclusion.
    If you're having trouble, use process of elimination—start with what you know to be evidence, and eliminate the choices with those claims.


    For every argument analysis, follow this complete COT process step-by-step.
    Always provide your reasoning chain and final confidence level.

    {passage}"""),
    
    ("human", "{query}")
    ])
    messages = conclusion_agent_prompt.format_messages(
            
    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"conclusion_response": response}