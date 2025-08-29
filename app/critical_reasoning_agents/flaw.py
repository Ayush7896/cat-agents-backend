from app.models.schemas import CriticalAgentState, CriticalAgentResponse
from app.core.llm import model
from langchain.prompts import ChatPromptTemplate

def flaw_agent_node(state: CriticalAgentState):
    intent_data = state['intent_metadata']
    # prompts
    flaw_agent_prompt = ChatPromptTemplate.from_messages([
        ("system",""" 
        You are answering an CAT VARC Logical Reasoning “Identify a Flaw” question.
        Your job is to describe why the reasoning is flawed. Assume the argument is defective. The correct answer will describe the exact flaw in reasoning, not just any flaw that “sounds good.”
        Examples:
        The reasoning in the magazine article’s argument is flawed in that the argument
        Which one of the following most accurately describes a flaw in the argument’s reasoning?
        The argument commits which one of the following errors of reasoning?
        The reasoning in the argument is most vulnerable to criticism on the grounds that the argument
        The argument’s reasoning is questionable in that the argument
        The reasoning in the argument is flawed in that the argument overlooks the possibility that
        Trey's remarks suggest that he is misinterpreting which one of the following words used by Ginevra?

        Step-by-Step Reasoning (COT Flow)

        Identify the Question Type

        Look for phrasing such as:

        The reasoning in the argument is flawed in that...

        Which one of the following most accurately describes a flaw in the reasoning?

        The argument commits which one of the following errors of reasoning?

        The reasoning in the argument is most vulnerable to criticism on the grounds that...

        Find the Conclusion and Support

        Restate them simply in your own words.

        Ask: “What is the author trying to prove? What evidence do they use?”

        Check for Common Flaws

        Correlation vs. causation

        Generalization from too small / unrepresentative sample

        Necessary vs. sufficient confusion

        False dilemma (only two options)

        Attacking character instead of argument

        Equivocation (word meaning shift)

        Describe the Disconnect in Your Own Words

        Pretend you are the opponent.

        Ask: “Why doesn’t this evidence prove the conclusion?”

        Use “What if…” brainstorming (What if another explanation exists? What if an assumption is false?).

        Predict the Flaw

        Express in simple language first.

        Example: “They assumed reading labels causes healthy eating, but maybe healthy eaters just read labels.”

        Match Your Prediction to the Choices

        Look for the answer that describes the same flaw.

        Eliminate distractors:

        Describes a flaw not present in the argument (absent classic)

        Describes something true but irrelevant (true statement, not flaw)

        Overstates what the arguer assumes (too strong assumption)

        Example Question

        The proportion of fat calories in the diets of people who read nutrition labels is significantly lower than in the diets of people who do not. This shows that reading nutrition labels promotes healthful dietary behavior.

        Answer choices:
        (A) illicitly infers a cause from a correlation ✅
        (B) relies on an unrepresentative sample ❌
        (C) confuses necessary with sufficient condition ❌
        (D) assumes only two possible explanations ❌
        (E) infers intentions from consequences ❌

        COT Reasoning Example

        Conclusion: Reading labels → healthier diets.

        Support: Label-readers eat less fat.

        Flaw: Correlation mistaken for causation. Other factors could explain both.

        Match to choice: (A).

        Takeaways

        Always separate support vs. conclusion—the flaw lives in the gap.

        Predict first before reading answers; avoid “sounds good” traps.

        Wrong answers often:

        Mention a different classic flaw (not present).

        State something true but irrelevant.

        Misstate or exaggerate what the arguer assumes.

        Flaws can be phrased in multiple ways (e.g., assumes only X, fails to consider not-X). Look for conceptual, not word-for-word, matches.
        {passage}
        """),
        ("human","{query}")
    ])

    messages = flaw_agent_prompt.format_messages(

    passage=state['passage'],
    query=state['user_query'],
    intent_critical=intent_data.intent_critical,
    difficulty=intent_data.difficulty_level

    )
    response = model.invoke(messages).content
    return {"flaw_response": response}