import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from lexiguard.agent import legal_agent

app = FastAPI(
    title="LexiGuard AI API",
    description="High-Precision Legal & Compliance Research Microservice",
    version="1.0.0"
)

# --- Models ---

class ChatRequest(BaseModel):
    query: str = Field(..., example="What are the compliance requirements for 2024?")
    history: Optional[List[dict]] = Field(default=[], description="Previous message history")

class ChatResponse(BaseModel):
    answer: str
    status: str = "success"

# --- Utils ---

def convert_history_to_messages(history: List[dict]) -> List[BaseMessage]:
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content")))
    return messages

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "LexiGuard AI API is operational."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint for legal research queries.
    Traverses the LexiGuard Graph (Retrieve -> Grade -> Correct -> Answer).
    """
    try:
        # Prepare state
        history_msgs = convert_history_to_messages(request.history)
        current_msg = HumanMessage(content=request.query)
        
        initial_state = {
            "messages": history_msgs + [current_msg],
            "loop_count": 0,
            "documents": []
        }

        # Invoke Graph Asynchronously (LangGraph supports sync/async, 
        # using aio_wrapper-like behavior if necessary, but 
        # legal_agent.invoke is standard)
        result = legal_agent.invoke(initial_state)
        
        # Extract last message content
        final_answer = result["messages"][-1].content
        
        return ChatResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
