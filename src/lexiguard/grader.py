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

class GradeRetrieval(BaseModel):
    """Binary score for retrieval check."""
    binary_score: str = Field(
        description="Retrieved documents are relevant and sufficient to answer the query, 'yes' or 'no'"
    )

def create_retrieval_grader(model: str = "gpt-4o", temperature: float = 0):
    """
    Creates a retrieval grader to check if the retrieved context is enough.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    structured_llm_grader = llm.with_structured_output(GradeRetrieval)

    system = """You are a grader assessing whether the retrieved legal documents are relevant and sufficient to answer the user's query. \n
         If the query mentions specific Indian Acts, Sections, or requires external legal verification not found in the facts, score it 'no'. \n
         Give a binary score 'yes' or 'no'. 'Yes' means the facts are sufficient to answer the query."""
    
    retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved facts: \n\n {documents} \n\n User query: {query}"),
        ]
    )

    retrieval_grader = retrieval_prompt | structured_llm_grader
    return retrieval_grader
