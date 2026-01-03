import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from lexiguard.app import app
from langchain_core.messages import HumanMessage

client = TestClient(app)

@patch("lexiguard.app.legal_agent")
def test_chat_endpoint_success(mock_graph):
    """Test /chat endpoint returns a successful answer."""
    # Mock graph response
    mock_graph.invoke.return_value = {
        "messages": [HumanMessage(content="This is a grounded answer.")]
    }
    
    response = client.post("/chat", json={"query": "test query", "history": []})
    
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a grounded answer."
    assert response.json()["status"] == "success"

@patch("lexiguard.app.legal_agent")
def test_chat_endpoint_max_retries(mock_graph):
    """Test /chat endpoint handles ungrounded exit gracefully (Max Retries)."""
    # Mock graph response for ungrounded exit
    mock_graph.invoke.return_value = {
        "messages": [HumanMessage(content="I'm sorry, I could not find a grounded answer...")]
    }
    
    response = client.post("/chat", json={"query": "unanswerable query", "history": []})
    
    assert response.status_code == 200
    assert "could not find a grounded answer" in response.json()["answer"]

def test_root_endpoint():
    """Test server root is operational."""
    response = client.get("/")
    assert response.status_code == 200
    assert "operational" in response.json()["message"]
