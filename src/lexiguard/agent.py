from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from lexiguard.grader import create_hallucination_grader

load_dotenv()

# --- Configuration & Modular LLM Setup ---
DEFAULT_MODEL = "gpt-4o"
LLM_TEMP = 0

def get_llm(model: str = DEFAULT_MODEL, temperature: float = LLM_TEMP):
    """Factory for LLM instances, allowing easy swap."""
    return ChatOpenAI(model=model, temperature=temperature)

llm = get_llm()
hallucination_grader = create_hallucination_grader(model=DEFAULT_MODEL, temperature=LLM_TEMP)

# --- Vector Store Setup (Domain Pivot) ---
# Note: In production, this would be replaced by Pinecone as per the plan.
# For this refactor, we maintain the Chroma logic but update the context.
pdf_path = os.path.join(os.getcwd(), "data", "sample_compliance.pdf")

if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        collection_name="legal_compliance"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    print(f"Warning: {pdf_path} not found. Retrieval will fail.")
    retriever = None

@tool
def legal_research_tool(query: str) -> str:
    """
    Searches the compliance database for legal statutes, internal policies, and regulatory requirements.
    """
    if not retriever:
        return "Legal database is currently unavailable."
    
    docs = retriever.invoke(query)
    return "\n\n".join([f"Source Match {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

tools = [legal_research_tool]
llm_with_tools = llm.bind_tools(tools)

# --- Graph State & Logic ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[str]
    loop_count: int  # Tracking attempts to ground answer

# ... [LLM and Tool setups remain same] ...

def call_model(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    docs = []
    for t in tool_calls:
        print(f"Executing: {t['name']}")
        res = legal_research_tool.invoke(t['args'].get('query', ''))
        docs.append(res)
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=res))
    return {"messages": results, "documents": docs}

def exit_ungrounded(state: AgentState):
    """Graceful exit when no grounded answer is found."""
    return {
        "messages": [HumanMessage(content="I'm sorry, I could not find a grounded answer in the provided legal documents after multiple attempts.")]
    }

def check_hallucination(state: AgentState):
    """
    Grader node to check for hallucinations and manage retries.
    """
    last_message = state['messages'][-1].content
    docs = "\n\n".join(state.get('documents', []))
    
    # Check for hallucination
    score = hallucination_grader.invoke({"documents": docs, "generation": last_message})
    
    current_loops = state.get('loop_count', 0)
    
    if score.binary_score == "yes":
        print("[LexiGuard] Response validated as grounded in source material.")
        return "useful"
    
    # If hallucinating, check if we hit max retries
    if current_loops >= 2: # 0, 1, 2 = 3 attempts total
        print("[LexiGuard] Maximum grounding attempts exceeded. Transferring to fallout handler.")
        return "max_retries"
    
    print(f"[LexiGuard] Hallucination detected. Re-attempting grounding (Attempt {current_loops + 1}/3)...")
    return "hallucination"

def increment_loop(state: AgentState):
    """Utility node to increment loop counter."""
    return {"loop_count": state.get('loop_count', 0) + 1}

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tools"
    return "grade"

# --- Graph Definition ---

def build_legal_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tool)
    builder.add_node("increment_loop", increment_loop)
    builder.add_node("exit_ungrounded", exit_ungrounded)
    
    builder.set_entry_point("agent")
    
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "grade": "grade_check_logic"
        }
    )
    
    builder.add_node("grade_check_logic", lambda state: state) # Router node
    
    builder.add_conditional_edges(
        "grade_check_logic",
        check_hallucination,
        {
            "useful": END,
            "hallucination": "increment_loop",
            "max_retries": "exit_ungrounded"
        }
    )
    
    builder.add_edge("increment_loop", "agent")
    builder.add_edge("tools", "agent")
    builder.add_edge("exit_ungrounded", END)
    
    return builder.compile()

legal_agent = build_legal_graph()

if __name__ == "__main__":
    print("LexiGuard AI: Legal Compliance Agent initialized with iterative self-correction.")
