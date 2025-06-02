FROM python:3.10-slim

WORKDIR /app

# Copy application code
COPY . /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic psycopg2-binary python-dotenv requests
RUN pip install --no-cache-dir langchain langchain-community langchain-core
RUN pip install --no-cache-dir faiss-cpu transformers sentence-transformers

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD uvicorn app:app --host=0.0.0.0 --port=8000