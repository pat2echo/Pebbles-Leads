#!/bin/bash

echo "Starting RAG Microservices..."

# Create data directories
mkdir -p data/chroma_db
mkdir -p data/memory
mkdir -p data/feedback
mkdir -p data/documents
mkdir -p data/ollama

# Start services
docker-compose up -d

echo "Waiting for Ollama to start..."
sleep 10

# Pull required models
docker-compose exec ollama ollama pull llama3
docker-compose exec ollama ollama pull nomic-embed-text

echo "RAG System is ready!"
echo "API Gateway: http://localhost:8000"
echo "Health Check: http://localhost:8000/health"