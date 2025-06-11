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

# Enhanced RAG processing logic
class EnhancedCoreRAGProcessor:
    def __init__(self):
        self.llm = Ollama(model="llama3", temperature=0.2, base_url="http://ollama:11434")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
        self.vectorstore = Chroma(
            persist_directory="/data/chroma_db",
            embedding_function=self.embeddings
        )
        self.memory_service_url = os.getenv('MEMORY_SERVICE_URL', 'http://memory-service:8002')
        self.feedback_service_url = os.getenv('FEEDBACK_SERVICE_URL', 'http://feedback-service:8003')

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query classification"""
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
            lines = response.split('\n')
            
            category = "factual"  # Default
            confidence = 0.5
            reasoning = "Default classification"
            
            for line in lines:
                if line.startswith('Category:'):
                    cat_str = line.split(':', 1)[1].strip().upper()
                    try:
                        category = cat_str.lower()
                    except:
                        category = "factual"
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return {
                "query_type": category,
                "confidence": confidence,
                "reasoning": reasoning
            }
        except Exception as e:
            return {
                "query_type": "factual",
                "confidence": 0.5,
                "reasoning": f"Error in classification: {str(e)}"
            }

    def retrieve_documents(self, query: str, query_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced document retrieval with strategy selection"""
        try:
            # Select search strategy based on query type
            if query_type in ["factual", "opinion"]:
                search_type = "similarity"
                search_kwargs = {"k": top_k * 2}
            else:  # analytical, contextual
                search_type = "mmr"
                search_kwargs = {"k": top_k, "fetch_k": top_k * 2, "lambda_mult": 0.7}
            
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            docs = retriever.get_relevant_documents(query)
            
            # Deduplicate documents
            seen_sources = set()
            unique_docs = []
            for doc in docs:
                source = f"{doc.metadata.get('source', 'unknown')}"
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get('source', 'unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', f"chunk_{i}")
                }
                for i, doc in enumerate(unique_docs)
            ]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, query: str, documents: List[Dict], query_type: str, 
                         conversation_context: str = "", current_topic: str = "general") -> str:
        """Enhanced response generation with memory awareness"""
        if not documents:
            return "I don't have enough information to answer this question."
        
        context = "\n\n".join([doc["content"] for doc in documents[:5]])
        
        # Enhanced templates based on query type
        templates = {
            "factual": """
Based on the following context and our previous conversation, provide a direct, factual answer.

Previous conversation context: {conversation_context}
Current topic: {current_topic}
Context: {context}
Question: {query}

Answer:""",
            "analytical": """
Considering our ongoing conversation and the context provided, analyze and synthesize the information.

Previous conversation: {conversation_context}
Current topic: {current_topic}
Context: {context}
Question: {query}

Analysis:""",
            "opinion": """
Based on our conversation history and the context, provide a balanced perspective.

Conversation context: {conversation_context}
Current topic: {current_topic}
Context: {context}
Question: {query}

Perspective:""",
            "contextual": """
Using our conversation history and comprehensive context, explain the relationships and implications.

Previous discussion: {conversation_context}
Current topic: {current_topic}
Context: {context}
Question: {query}

Contextual Answer:"""
        }
        
        template = templates.get(query_type, templates["factual"])
        
        prompt = template.format(
            context=context,
            query=query,
            conversation_context=conversation_context,
            current_topic=current_topic
        )
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

processor = EnhancedCoreRAGProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "core-rag"})

@app.route('/test/index', methods=['GET'])
def get_first_five_entries():
    """Get the first 5 entries in the index"""
    result = processor.retrieve_documents(query="Mental", query_type="factual")
    return jsonify(result)
    
@app.route('/process', methods=['POST'])
def process_query():
    """Enhanced main endpoint for processing queries"""
    data = request.get_json()
    query = data.get('query', '').strip()
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Get enhanced conversation context from memory service
        conversation_context = ""
        current_topic = "general"
        context_data = {}
        
        if session_id:
            try:
                memory_response = requests.get(
                    f"{processor.memory_service_url}/context/{session_id}",
                    timeout=10
                )
                if memory_response.status_code == 200:
                    context_data = memory_response.json()
                    conversation_context = context_data.get('conversation_context', '')
                    current_topic = context_data.get('current_topic', 'general')
            except Exception as e:
                print(f"Memory service error: {e}")
        
        # Process the query with enhanced classification
        classification = processor.classify_query(query)
        documents = processor.retrieve_documents(query, classification['query_type'])
        answer = processor.generate_response(
            query, documents, classification['query_type'], 
            conversation_context, current_topic
        )
        
        # Generate turn ID
        turn_id = str(uuid.uuid4())
        
        # Determine strategy used
        strategy_used = "similarity" if classification['query_type'] in ["factual", "opinion"] else "mmr"
        
        response_data = {
            "answer": answer,
            "query_type": classification['query_type'],
            "confidence": classification['confidence'],
            "turn_id": turn_id,
            "session_id": session_id,
            "sources": documents[:3],
            "timestamp": datetime.now().isoformat(),
            "strategy_used": strategy_used,
            "topic_changed": False,  # Will be updated by memory service
            "current_topic": current_topic
        }
        
        # Store turn in memory service with enhanced data
        if session_id:
            try:
                memory_result = requests.post(
                    f"{processor.memory_service_url}/turns",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "query": query,
                        "response": answer,
                        "query_type": classification['query_type'],
                        "confidence": classification['confidence'],
                        "strategy_used": strategy_used,
                        "sources": documents
                    },
                    timeout=10
                )
                
                if memory_result.status_code == 200:
                    memory_data = memory_result.json()
                    response_data["topic_changed"] = memory_data.get("topic_changed", False)
                    response_data["current_topic"] = memory_data.get("current_topic", current_topic)
                    
            except Exception as e:
                print(f"Error storing turn in memory: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)