# Module 3 — Zepto Support Assistant (`/support_assistant`)

This module implements a complete GenAI Customer Support Assistant for Zepto using a local vector store (ChromaDB), local embeddings (`sentence-transformers`), a LangGraph flow graph, and a FastAPI wrapper.

---

## 1. RAG Pipeline Architecture Description

The pipeline consists of 4 distinct stages operating in sequence:

1. **Ingestion**: The 8 Zepto policy text documents (`doc_01.txt` through `doc_08.txt`) located in `support_assistant/docs/` are loaded into memory on application startup.
2. **Embedding**: Each document is converted into dense vector embeddings using the open-source `sentence-transformers` model (`all-MiniLM-L6-v2`) via ChromaDB's native embedding integration.
3. **Retrieval**: The vector embeddings are indexed and stored in an in-memory ChromaDB collection named `zepto_policies`. When a query is classified as a policy question, ChromaDB retrieves the top 3 most similar document chunks using cosine distance.
4. **Generation**:
   - **Default / Mock Baseline (`MOCK_LLM=1`)**: Returns a deterministic response formatted as `Based on the retrieved context: <top_chunk_snippet>` for policy questions, and a canned string (`I can only answer questions about Zepto policies right now.`) for general questions.
   - **Real LLM Extension (`MOCK_LLM=0`)**: Formats the retrieved context into the structured prompt template and sends it to the LLM backend.

---

## 2. Structured Prompt Skeleton

The prompt template is designed around the **Role – Context – Task – Format – Length** skeleton, featuring an explicit **Negative Constraint** and a **Few-shot Example**:

```text
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
```

---

## 3. Example Execution & API Responses (Default MOCK_LLM Baseline)

Below are the raw JSON responses from calling the `POST /ask` FastAPI endpoint with `MOCK_LLM=1` (the default graded baseline):

### Example 1: Policy Question (Triggers Intent Routing & Retrieval)
* **Request**: `POST /ask`
```json
{
  "query": "What is the delivery fee for orders below 149?"
}
```
* **Response**:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01",
    "doc_06",
    "doc_03"
  ],
  "confidence": 1.0
}
```

### Example 2: General Question (Direct Answer, No Retrieval)
* **Request**: `POST /ask`
```json
{
  "query": "Tell me a joke"
}
```
* **Response**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 4. Local Build & Run Instructions

### Running via Uvicorn
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Building & Running with Docker
```bash
# Build the image
docker build -t zepto-support-assistant .

# Run the container locally
docker run -p 7860:7860 zepto-support-assistant
```
