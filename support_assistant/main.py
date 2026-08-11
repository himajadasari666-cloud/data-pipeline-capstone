import os
from typing import List, TypedDict, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

# --- Environment Toggle ---
MOCK_LLM = os.getenv("MOCK_LLM", "1")

# --- FastAPI App ---
app = FastAPI(title="Zepto Support Assistant")

# --- Schemas ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

# --- Vector DB Setup ---
chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    embedding_function=sentence_transformer_ef
)

# Load Corpus
DOCS = {
    "doc_01": "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "doc_02": "Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
    "doc_03": "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.",
    "doc_04": "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
    "doc_05": "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
    "doc_06": "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.",
    "doc_07": "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.",
    "doc_08": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."
}

if collection.count() == 0:
    collection.add(
        documents=list(DOCS.values()),
        ids=list(DOCS.keys()),
        metadatas=[{"source": doc_id} for doc_id in DOCS.keys()]
    )

# --- Prompt Template ---
PROMPT_TEMPLATE = """
Role: You are an official Zepto Customer Support Assistant.
Context:
{context}

Task: Answer the user query using only the provided context.
Format: Return a concise, direct answer based strictly on the context.
Length: Under 100 words.

Negative Constraint: Do not answer using information not present in the provided context.

Few-shot Example:
User: What is the delivery fee for orders below 149?
Assistant: Standard delivery incurs a flat INR 25 delivery fee for orders below INR 149.

User Query: {query}
"""

# --- LangGraph State ---
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    response: QueryResponse

# --- Nodes ---
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    
    if MOCK_LLM != "0":
        state["intent"] = "policy_question" if any(kw in query for kw in keywords) else "general_question"
    else:
        state["intent"] = "policy_question" if any(kw in query for kw in keywords) else "general_question"
    return state

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []
    
    state["retrieved_chunks"] = retrieved_chunks
    state["retrieved_ids"] = retrieved_ids
    
    top_snippet = retrieved_chunks[0][:200] if retrieved_chunks else ""
    answer_text = f"Based on the retrieved context: {top_snippet}"
    
    state["response"] = QueryResponse(
        answer=answer_text,
        sources=retrieved_ids,
        confidence=1.0
    )
    return state

def direct_answer(state: GraphState) -> GraphState:
    state["response"] = QueryResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return state

# --- Routing ---
def route_intent(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

# --- Graph Assembly ---
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()

# --- API Endpoint ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state: GraphState = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "retrieved_ids": [],
        "response": QueryResponse(answer="", sources=[], confidence=0.0)
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
