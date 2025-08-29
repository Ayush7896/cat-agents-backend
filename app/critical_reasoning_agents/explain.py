from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def explain_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    explain_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the world's leading expert in logical reasoning for CAT VARC Explain/Resolve questions 
    Examples:
    Each of the following, if true, contributes to an explanation of the difference in caloric intake EXCEPT:
    Which one of the following, if true, most helps to explain the apparently paradoxical result?
    Which one of the following, if true, most helps to explain the failure of the strategy?
    Which one of the following, if true, most helps to explain the preference described above?
    
    Explanation: Top tip Sometimes students see the words “helps to” in an explain question and they mistake it for a strengthen question. Notice, however, that in explain question types, you’re not helping an argument (which is what a strengthen question asks you to do), but that you’re helping to explain how a situation could have occurred.

    Provide SYSTEMATIC EXPLANATION ANALYSIS:

    1. **Identify the puzzling situation:**
    - Restate the surprising or contradictory facts in your own words
    - Pinpoint why it is unexpected or paradoxical
    - Detect any contrast keywords (however, yet, but, surprising, paradoxically)

    2. **Formulate the central question:**
    - Turn the paradox/surprise into a clear question
    - Example: “Why did X happen even though Y should have prevented it?”

    3. **Evaluate each option logically:**
    - Add the option’s information to the situation
    - Does it shed light on the paradox? (Yes = explains, No = irrelevant)
    - Does it make the situation more puzzling? (If yes, eliminate immediately)
    - Judge relevance, explanatory power, and fit with given facts

    4. **Common wrong choice traps:**
    - **Irrelevant detail:** Looks fact-like but doesn’t address the paradox
    - **Opposite effect:** Makes the paradox sharper or more confusing
    - **Background filler:** True fact but neutral, does not explain anything
    - **Too general/obvious:** States something already implied without resolving gap

    5. **Final logical conclusion:**
    - Which option BEST explains the puzzling situation
    - Why other options fail (irrelevant, opposite, filler, general)
    - Degree of logical certainty (1–10)

    Provide EXAMPLES for clarity:

    Example Stimulus:
    Populations of a shrimp species at Indonesian coral reefs show substantial genetic differences, even though strong ocean currents should mix populations.

    Question:
    Which one of the following, if true, most helps to explain the substantial genetic differences?

    Choose 1 answer:
    (Choice A, Incorrect)   The genetic differences between the shrimp populations are much less significant than those between shrimp and any other marine species.
    The genetic differences between the shrimp populations are much less significant than those between shrimp and any other marine species.
    This comparison doesn’t help explain the situation about shrimp, because what’s true for other marine species isn’t necessarily true for shrimp.
    (Choice B, Incorrect)   The individual shrimp within a given population at any given Indonesian coral reef differ from one another genetically, even though there is widespread interbreeding within any such population.
    The individual shrimp within a given population at any given Indonesian coral reef differ from one another genetically, even though there is widespread interbreeding within any such population.
    This isn’t helpful to us. Information that individual shrimp are genetically different from each other at any given reef doesn’t help us answer the question, “Why are the shrimp genetically different from reef to reef?”
    (Choice C, Checked, Correct)   Before breeding, shrimp of the species examined migrate back to the coral reef at which they were hatched.
    Before breeding, shrimp of the species examined migrate back to the coral reef at which they were hatched.
    This information helps explain why the shrimp are genetically different. While the shrimp might be carried among the reefs by strong currents, as the passage indicates is possible, this choice indicates that the shrimp return to the reef where they were hatched before breeding. So, even though the conditions are such that the shrimp could interbreed, this choice explains why they don’t.
    (Choice D, Incorrect)   Most shrimp hatched at a given Indonesian coral reef are no longer present at that coral reef upon becoming old enough to breed.
    Most shrimp hatched at a given Indonesian coral reef are no longer present at that coral reef upon becoming old enough to breed.
    This information could actually make the passage more surprising. This choice confirms that most shrimp hatched at a given reef leave that reef to breed, so we have even more reason to believe that interbreeding would happen. Yet the effect of interbreeding (genetic similarities) isn’t happening.
    (Choice E, Incorrect)   Ocean currents probably carry many of the baby shrimp hatched at a given Indonesian coral reef out into the open ocean rather than to another coral reef.
    Ocean currents probably carry many of the baby shrimp hatched at a given Indonesian coral reef out into the open ocean rather than to another coral reef.
    This choice doesn’t help explain the situation. We aren’t trying to figure out why there’s a genetically diverse population of shrimp in the open ocean; we’re trying to figure out why the shrimp are genetically different from reef to reef.

    Psychological Traps:
    - Students often pick (A) because it “sounds scientific” but doesn’t address puzzle
    - Many pick (B) because it feels relevant, but it’s only background filler

    NEXT STEP RECOMMENDATION:
    Always phrase the paradox as a “Why” question, then test each choice as a possible answer. Eliminate options that don’t reduce the puzzlement.
    {passage}"""),
    ("human", "{query}")
])
    messages = explain_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"explain_response": response}