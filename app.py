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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)