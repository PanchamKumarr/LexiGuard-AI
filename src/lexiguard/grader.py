"""
LexiGuard AI: Hallucination Grader
Ensures LLM generations are strictly grounded in retrieved legal documents.
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class GradeHallucination(BaseModel):
    """Binary score for hallucination check in LLM generation."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def create_hallucination_grader(model: str = "gpt-4o", temperature: float = 0):
    """
    Creates a hallucination grader node.
    
    Args:
        model: The LLM model to use.
        temperature: LLM temperature.
        
    Returns:
        A callable that grades the LLM generation.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    structured_llm_grader = llm.with_structured_output(GradeHallucination)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader

# Example usage in a graph node:
# grader = create_hallucination_grader()
# score = grader.invoke({"documents": docs, "generation": answer})
