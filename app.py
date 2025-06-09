from fastapi import FastAPI
from pydantic import BaseModel
from rag import GeminiRAG
import uvicorn
import requests

app = FastAPI()
rag_system = GeminiRAG()

class ChatRequest(BaseModel):
    question: str
    user_id: str
    role: str
    use_general_knowledge: bool = False

class ChatResponse(BaseModel):
    answer: str
    source_type: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if request.use_general_knowledge:
        answer = call_general_knowledge_llm(request.question, request.role)
        return ChatResponse(answer=answer, source_type="general")
    else:
        try:
            answer = rag_system.query(
                question=request.question,
                user_id=request.user_id,
                role=request.role
            )
            
            if "tidak ada data" in answer.lower() or "tidak ditemukan" in answer.lower():
                answer = call_general_knowledge_llm(request.question, request.role)
                return ChatResponse(answer=answer, source_type="general")
                
            return ChatResponse(answer=answer, source_type="rag")
        except Exception as e:
            answer = call_general_knowledge_llm(request.question, request.role)
            return ChatResponse(answer=answer, source_type="general")

def call_general_knowledge_llm(question: str, role: str) -> str:
    """Call Gemini API for general knowledge questions"""
    prompt = f"""
    Anda adalah asisten AI yang membantu {role}.
    
    Pertanyaan: {question}
    
    Mohon jawab dengan pengetahuan umum yang Anda miliki. Berikan jawaban yang informatif dan bermanfaat.
    Jika pertanyaan tidak sesuai atau tidak dapat dijawab, beri tahu pengguna dengan sopan.
    """
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyDXVqrq4Yd-vw4MCC--Qs6UNdckA9F1x_Y",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Maaf, saya tidak dapat menjawab pertanyaan tersebut saat ini. Error: {str(e)}"

@app.get("/ping")
async def ping():
    """Health check endpoint to verify the API is running"""
    return {"status": "ok", "message": "RAG Chatbot API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)