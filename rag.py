from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # Embedding alternatif
from database import fetch_documents
import os
import requests
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv


class DeepSeekRAG:
    def __init__(self):
        # Inisialisasi Embedding Model (DeepSeek belum menyediakan embedding resmi)
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load dokumen dari database
        self.documents = fetch_documents()
        
        # Buat vector store
        self.vector_store = FAISS.from_texts(
            texts=[doc['content'] for doc in self.documents],
            embedding=self.embeddings,
            metadatas=[{'title': doc['title'], 'id': doc['id']} for doc in self.documents]
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
        
        # Template prompt khusus DeepSeek
        self.template = """[INST] Anda adalah asisten AI yang membantu menjawab pertanyaan. 
        Gunakan informasi berikut untuk menjawab dengan tepat:

        Konteks: {context}

        Pertanyaan: {question} [/INST]"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
        # RAG Chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | RunnableLambda(self._invoke_deepseek)
            | StrOutputParser()
        )
    
    def _invoke_deepseek(self, prompt: str) -> str:
        print("=== Prompt yang dikirim ke Gemini ===")
        print(prompt)
        prompt_str = prompt.to_string()
        """Mengirim permintaan ke API Gemini"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt_str
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyARQmOSicjZONPlpVM0P6n0-S3N6f1J634",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

            
    def query(self, question: str) -> str:
        """Antarmuka utama untuk query RAG"""
        return self.rag_chain.invoke(question)