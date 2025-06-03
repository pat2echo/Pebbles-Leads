"""
Core RAG Service - Handles query processing, classification, and response generation
Port: 8001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import your existing classes (simplified for core functionality)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    OPINION = "opinion"
    CONTEXTUAL = "contextual"

# Core RAG processing logic
class CoreRAGProcessor:
    def __init__(self):
        self.llm = Ollama(model="llama3", temperature=0.2, 
            base_url="http://ollama:11434")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", 
            base_url="http://ollama:11434")
        self.vectorstore = Chroma(
            persist_directory="/data/chroma_db",
            embedding_function=self.embeddings
        )
        self.memory_service_url = os.getenv('MEMORY_SERVICE_URL', 'http://memory-service:8002')
        self.feedback_service_url = os.getenv('FEEDBACK_SERVICE_URL', 'http://feedback-service:8003')

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify the query type"""
        classification_prompt = f"""
        Classify the following query into one of these categories:
        1. FACTUAL: Asking for specific facts, definitions, or direct information
        2. ANALYTICAL: Requiring analysis, comparison, or synthesis of information
        3. OPINION: Seeking subjective views, recommendations, or interpretations
        4. CONTEXTUAL: Needing understanding of relationships, context, or broader implications

        Query: {query}

        Respond in this exact format:
        Category: [FACTUAL/ANALYTICAL/OPINION/CONTEXTUAL]
        Confidence: [0.0-1.0]
        Reasoning: [Brief explanation]
        """
        
        try:
            response = self.llm.invoke(classification_prompt).strip()
            # Parse response (simplified)
            return {
                "query_type": "factual",  # Default
                "confidence": 0.8,
                "reasoning": "Automated classification"
            }
        except Exception as e:
            return {
                "query_type": "factual",
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}"
            }

    def retrieve_documents(self, query: str, query_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        try:
            search_type = "similarity" if query_type in ["factual", "opinion"] else "mmr"
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": top_k}
            )
            
            docs = retriever.get_relevant_documents(query)
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get('source', 'unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', f"chunk_{i}")
                }
                for i, doc in enumerate(docs[:top_k])
            ]
        except Exception as e:
            return []

    def generate_response(self, query: str, documents: List[Dict], query_type: str, 
                         conversation_context: str = "") -> str:
        """Generate response using LLM"""
        if not documents:
            return "I don't have enough information to answer this question."
        
        context = "\n\n".join([doc["content"] for doc in documents[:3]])
        
        prompt = f"""
        Based on the following context and conversation history, provide a helpful answer.
        
        Previous conversation: {conversation_context}
        Context: {context}
        Question: {query}
        
        Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

processor = CoreRAGProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "core-rag"})

@app.route('/process', methods=['POST'])
def process_query():
    """Main endpoint for processing queries"""
    data = request.get_json()
    query = data.get('query', '').strip()
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Get conversation context from memory service
        conversation_context = ""
        if session_id:
            try:
                memory_response = requests.get(
                    f"{processor.memory_service_url}/context/{session_id}",
                    timeout=5
                )
                if memory_response.status_code == 200:
                    context_data = memory_response.json()
                    conversation_context = context_data.get('conversation_context', '')
            except Exception as e:
                print(f"Memory service error: {e}")
        
        # Process the query
        classification = processor.classify_query(query)
        documents = processor.retrieve_documents(query, classification['query_type'])
        answer = processor.generate_response(
            query, documents, classification['query_type'], conversation_context
        )
        
        # Generate turn ID
        turn_id = str(uuid.uuid4())
        
        response_data = {
            "answer": answer,
            "query_type": classification['query_type'],
            "confidence": classification['confidence'],
            "turn_id": turn_id,
            "session_id": session_id,
            "sources": documents[:3],
            "timestamp": datetime.now().isoformat()
        }
        
        # Store turn in memory service (async)
        if session_id:
            try:
                requests.post(
                    f"{processor.memory_service_url}/turns",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "query": query,
                        "response": answer,
                        "query_type": classification['query_type'],
                        "confidence": classification['confidence'],
                        "sources": documents
                    },
                    timeout=5
                )
            except Exception as e:
                print(f"Error storing turn: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)