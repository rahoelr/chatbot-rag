from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import requests
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from database import fetch_documents_by_role


class GeminiRAG:
    def __init__(self):
        # Inisialisasi Embedding Model
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )

        # Template Prompt
        self.template = """
        Anda adalah asisten AI pintar yang bertugas membantu pengguna berdasarkan data yang tersedia.

        Role pengguna: {role}
        Nama pengguna: {name} (ID: {user_id})

        Berikut adalah informasi terkait pengguna:
        {context}

        Pertanyaan pengguna:
        {question}

        Jawablah dengan nada yang sesuai untuk role {role}. Jika data terbatas, jawab dengan sopan dan beri saran yang membangun.
        """

        self.prompt = ChatPromptTemplate.from_template(self.template)
    
    def _invoke_gemini(self, prompt: str) -> str:
        print("=== Prompt yang dikirim ke Gemini ===")
        print(prompt)
        prompt_str = prompt.to_string()

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_str}
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
            return f"Error calling Gemini API: {str(e)}"

            
    def query(self, question: str, user_id: str, role: str, name: str = "Pengguna") -> str:
        """Melakukan query berdasarkan user_id, role, dan nama"""
        documents = fetch_documents_by_role(user_id, role)

        if not documents:
            return "Maaf, tidak ada data yang ditemukan untuk pengguna ini."

        vector_store = FAISS.from_texts(
            texts=[doc['content'] for doc in documents],
            embedding=self.embeddings,
            metadatas=[{'title': doc['title'], 'id': doc['id']} for doc in documents]
        )

        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "role": lambda _: role,
                "name": lambda _: name,
                "user_id": lambda _: user_id
            }
            | self.prompt
            | RunnableLambda(self._invoke_gemini)
            | StrOutputParser()
        )

        return rag_chain.invoke(question)