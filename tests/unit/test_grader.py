import pytest
from unittest.mock import MagicMock, patch
from lexiguard.grader import create_hallucination_grader, GradeHallucination

@pytest.fixture
def mock_llm():
    with patch("lexiguard.grader.ChatOpenAI") as mocked:
        yield mocked

def test_grader_grounded_answer(mock_llm):
    """Test grader when generation is grounded in facts."""
    # Setup mock structured output
    mock_instance = mock_llm.return_value
    mock_structured_llm = MagicMock()
    mock_instance.with_structured_output.return_value = mock_structured_llm
    
    # Mock return value for 'grounded' case
    mock_structured_llm.invoke.return_value = GradeHallucination(binary_score="yes")
    
    grader = create_hallucination_grader()
    score = grader.invoke({
        "documents": "The capital of France is Paris.",
        "generation": "Paris is the capital of France."
    })
    
    assert score.binary_score == "yes"

def test_grader_hallucinated_answer(mock_llm):
    """Test grader when generation is NOT grounded in facts."""
    # Setup mock structured output
    mock_instance = mock_llm.return_value
    mock_structured_llm = MagicMock()
    mock_instance.with_structured_output.return_value = mock_structured_llm
    
    # Mock return value for 'hallucinated' case
    mock_structured_llm.invoke.return_value = GradeHallucination(binary_score="no")
    
    grader = create_hallucination_grader()
    score = grader.invoke({
        "documents": "The capital of France is Paris.",
        "generation": "Lyon is the capital of France."
    })
    
    assert score.binary_score == "no"
