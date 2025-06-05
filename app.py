from fastapi import FastAPI
from pydantic import BaseModel
from rag import GeminiRAG
import uvicorn

app = FastAPI()
rag_system = GeminiRAG()  

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    answer = rag_system.query(request.question)
    return ChatResponse(answer=answer)

@app.get("/ping")
async def ping():
    """Health check endpoint to verify the API is running"""
    return {"status": "ok", "message": "RAG Chatbot API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)