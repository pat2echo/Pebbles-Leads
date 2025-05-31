#!/usr/bin/env python3
"""
Modular RAG Backend System with ChromaDB and Llama3
Requires: pip install chromadb ollama langchain langchain-community langchain-chroma flask flask-cors
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader,
    UnstructuredMarkdownLoader, UnstructuredHTMLLoader
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# GPU/CPU Configuration
def configure_hardware():
    """
    Check if a GPU is available and configure the system to use GPU or CPU.
    Returns a dictionary with device information and configuration status.
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_info = {
                "device": "GPU",
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "status": "success",
                "message": f"Using GPU: {torch.cuda.get_device_name(0)}"
            }
        else:
            device = torch.device("cpu")
            device_info = {
                "device": "CPU",
                "device_name": "N/A",
                "cuda_version": "N/A",
                "status": "success",
                "message": "No GPU detected, falling back to CPU"
            }
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device.type == "cuda" else ""
        return device_info
    except Exception as e:
        device_info = {
            "device": "CPU",
            "device_name": "N/A",
            "cuda_version": "N/A",
            "status": "error",
            "message": f"Error detecting hardware: {str(e)}"
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return device_info

# Core data structures
class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    OPINION = "opinion"
    CONTEXTUAL = "contextual"

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_id: Optional[str] = None

@dataclass
class RetrievalResult:
    documents: List[Document]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any] = None

@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    reasoning: str

@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query_type: QueryType
    strategy_used: str
    confidence: float
    metadata: Dict[str, Any] = None

# Abstract base classes
class QueryClassifier(ABC):
    @abstractmethod
    def classify(self, query: str) -> QueryClassification:
        pass

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        pass

class ResponseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieval_result: RetrievalResult,
                query_type: QueryType) -> str:
        pass

class ResponseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, query: str, response: str, sources: List[Document]) -> Dict[str, Any]:
        pass

# Concrete implementations
class LLMQueryClassifier(QueryClassifier):
    def __init__(self, llm):
        self.llm = llm

    def classify(self, query: str) -> QueryClassification:
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
            category = QueryType.FACTUAL
            confidence = 0.5
            reasoning = "Default classification"

            for line in lines:
                if line.startswith('Category:'):
                    cat_str = line.split(':', 1)[1].strip().upper()
                    try:
                        category = QueryType(cat_str.lower())
                    except ValueError:
                        category = QueryType.FACTUAL
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()

            return QueryClassification(category, confidence, reasoning)
        except Exception as e:
            return QueryClassification(QueryType.FACTUAL, 0.5, f"Error in classification: {str(e)}")

class FactualRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k * 2}
            )

            docs = retriever.get_relevant_documents(query)
            seen_sources = set()
            unique_docs = []
            for doc in docs:
                source = f"{doc.metadata.get('row', 'unknown')}, {doc.metadata.get('source', 'unknown')}"
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break

            documents = [Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source=source,
                chunk_id=doc.metadata.get('chunk_id', f"{source}_{i}")
            ) for i, doc in enumerate(unique_docs)]
            scores = [1.0] * len(documents)

            return RetrievalResult(
                documents=documents,
                scores=scores,
                strategy_used="factual",
                metadata={"search_type": "similarity", "deduplicated": True}
            )
        except Exception as e:
            return RetrievalResult([], [], "factual", {"error": str(e)})

class AnalyticalRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 8) -> RetrievalResult:
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": top_k * 2, "lambda_mult": 0.7}
            )

            docs = retriever.get_relevant_documents(query)
            seen_sources = set()
            unique_docs = []
            for doc in docs:
                source = f"{doc.metadata.get('row', 'unknown')}, {doc.metadata.get('source', 'unknown')}"
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break

            documents = [Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source=source,
                chunk_id=doc.metadata.get('chunk_id', f"{source}_{i}")
            ) for i, doc in enumerate(unique_docs)]
            scores = [1.0] * len(documents)

            return RetrievalResult(
                documents=documents,
                scores=scores,
                strategy_used="analytical",
                metadata={"search_type": "mmr", "deduplicated": True}
            )
        except Exception as e:
            return RetrievalResult([], [], "analytical", {"error": str(e)})

class OpinionRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 6) -> RetrievalResult:
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k * 2}
            )

            docs = retriever.get_relevant_documents(query)
            seen_sources = set()
            unique_docs = []
            for doc in docs:
                source = f"{doc.metadata.get('row', 'unknown')}, {doc.metadata.get('source', 'unknown')}"
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break

            documents = [Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source=source,
                chunk_id=doc.metadata.get('chunk_id', f"{source}_{i}")
            ) for i, doc in enumerate(unique_docs)]
            scores = [1.0] * len(documents)

            return RetrievalResult(
                documents=documents,
                scores=scores,
                strategy_used="opinion",
                metadata={"search_type": "similarity_opinion", "deduplicated": True}
            )
        except Exception as e:
            return RetrievalResult([], [], "opinion", {"error": str(e)})

class ContextualRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": top_k * 2, "lambda_mult": 0.7}
            )

            docs = retriever.get_relevant_documents(query)
            seen_sources = set()
            unique_docs = []
            for doc in docs:
                source = f"{doc.metadata.get('row', 'unknown')}, {doc.metadata.get('source', 'unknown')}"
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break

            documents = [Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source=source,
                chunk_id=doc.metadata.get('chunk_id', f"{source}_{i}")
            ) for i, doc in enumerate(unique_docs)]
            scores = [1.0] * len(documents)

            return RetrievalResult(
                documents=documents,
                scores=scores,
                strategy_used="contextual",
                metadata={"search_type": "mmr_contextual", "deduplicated": True}
            )
        except Exception as e:
            return RetrievalResult([], [], "contextual", {"error": str(e)})

class AdaptiveResponseGenerator(ResponseGenerator):
    def __init__(self, llm):
        self.llm = llm
        self.templates = {
            QueryType.FACTUAL: """
Based on the following context, provide a direct, factual answer to the question.
Be precise and cite specific information from the sources.

Context: {context}

Question: {query}

Answer:""",
            QueryType.ANALYTICAL: """
Using the provided context, analyze and synthesize the information to answer the question.
Consider multiple perspectives and provide reasoning for your conclusions.

Context: {context}

Question: {query}

Analysis:""",
            QueryType.OPINION: """
Based on the context provided, offer a balanced perspective on the question.
Present different viewpoints when available and acknowledge subjective elements.

Context: {context}

Question: {query}

Perspective:""",
            QueryType.CONTEXTUAL: """
Using the comprehensive context provided, explain the relationships and broader implications
relevant to the question. Provide background and situate the answer within the larger context.

Context: {context}

Question: {query}

Contextual Answer:"""
        }

    def generate(self, query: str, retrieval_result: RetrievalResult,
                query_type: QueryType) -> str:
        try:
            if not retrieval_result.documents:
                return "I don't have enough information to answer this question."
            
            context = "\n\n".join([doc.content for doc in retrieval_result.documents[:5]])
            template = self.templates.get(query_type, self.templates[QueryType.FACTUAL])
            
            prompt = template.format(context=context, query=query)
            response = self.llm.invoke(prompt)
            
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class SimpleResponseEvaluator(ResponseEvaluator):
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, query: str, response: str, sources: List[Document]) -> Dict[str, Any]:
        try:
            eval_prompt = f"""
Evaluate the quality of this response on a scale of 1-10:

Query: {query}
Response: {response}

Consider:
- Relevance to the query
- Accuracy based on sources
- Completeness
- Clarity

Provide only a score from 1-10 and brief explanation.
Format: Score: X
Explanation: [brief explanation]
"""
            
            eval_response = self.llm.invoke(eval_prompt).strip()
            
            score = 5.0
            explanation = "Default evaluation"
            
            lines = eval_response.split('\n')
            for line in lines:
                if line.startswith('Score:'):
                    try:
                        score_str = line.split(':', 1)[1].strip()
                        score = float(score_str)
                    except:
                        pass
                elif line.startswith('Explanation:'):
                    explanation = line.split(':', 1)[1].strip()
            
            return {
                "score": score,
                "explanation": explanation,
                "source_count": len(sources),
                "response_length": len(response)
            }
        except Exception as e:
            return {
                "score": 5.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "source_count": len(sources),
                "response_length": len(response)
            }

# Main RAG System
class ModularRAGSystem:
    def __init__(self,
                 vectorstore,
                 llm,
                 query_classifier: Optional[QueryClassifier] = None,
                 response_generator: Optional[ResponseGenerator] = None,
                 response_evaluator: Optional[ResponseEvaluator] = None):
        
        self.vectorstore = vectorstore
        self.llm = llm
        
        self.query_classifier = query_classifier or LLMQueryClassifier(llm)
        self.response_generator = response_generator or AdaptiveResponseGenerator(llm)
        self.response_evaluator = response_evaluator or SimpleResponseEvaluator(llm)
        
        self.retrieval_strategies = {
            QueryType.FACTUAL: FactualRetrievalStrategy(vectorstore),
            QueryType.ANALYTICAL: AnalyticalRetrievalStrategy(vectorstore),
            QueryType.OPINION: OpinionRetrievalStrategy(vectorstore),
            QueryType.CONTEXTUAL: ContextualRetrievalStrategy(vectorstore)
        }

    def process_query(self, query: str) -> RAGResponse:
        try:
            classification = self.query_classifier.classify(query)
            retrieval_strategy = self.retrieval_strategies[classification.query_type]
            retrieval_result = retrieval_strategy.retrieve(query)
            answer = self.response_generator.generate(
                query, retrieval_result, classification.query_type
            )
            evaluation = self.response_evaluator.evaluate(
                query, answer, retrieval_result.documents
            )
            
            return RAGResponse(
                answer=answer,
                sources=retrieval_result.documents,
                query_type=classification.query_type,
                strategy_used=retrieval_result.strategy_used,
                confidence=classification.confidence,
                metadata={
                    "classification": {
                        "confidence": classification.confidence,
                        "reasoning": classification.reasoning
                    },
                    "retrieval": retrieval_result.metadata,
                    "evaluation": evaluation
                }
            )
        except Exception as e:
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query_type=QueryType.FACTUAL,
                strategy_used="error",
                confidence=0.0,
                metadata={"error": str(e)}
            )

# Document indexing system
class DocumentIndexer:
    def __init__(self, vectorstore, text_splitter=None):
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )

    def clear_vectorstore(self):
        """Clear all documents from the vectorstore."""
        try:
            self.vectorstore.reset_collection()
            return {"status": "success", "message": "Cleared existing vectorstore collection."}
        except Exception as e:
            return {"status": "error", "message": f"Error clearing vectorstore: {str(e)}"}

    def get_loader_for_file(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return TextLoader(str(file_path))
        elif suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix == ".csv":
            return CSVLoader(str(file_path))
        elif suffix in [".md", ".markdown"]:
            return UnstructuredMarkdownLoader(str(file_path))
        elif suffix in [".html", ".htm"]:
            return UnstructuredHTMLLoader(str(file_path))
        return None

    def index_directory(self, data_dir: str):
        data_path = Path(data_dir)
        if not data_path.exists():
            return {"status": "error", "message": f"Directory {data_dir} does not exist"}
        
        self.clear_vectorstore()
        indexed_files = []
        errors = []
        file_hashes = set()

        for file_path in data_path.glob("**/*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                if file_hash in file_hashes:
                    errors.append({
                        "file": str(file_path),
                        "error": "Duplicate file skipped"
                    })
                    continue
                file_hashes.add(file_hash)
                try:
                    loader = self.get_loader_for_file(file_path)
                    if loader:
                        print("not loader", loader)
                        documents = loader.load()
                        chunks = self.text_splitter.split_documents(documents)
                        for i, chunk in enumerate(chunks):
                            chunk.metadata['chunk_id'] = f"{file_path}_{i}"
                        if chunks:
                            self.vectorstore.add_documents(chunks)
                            indexed_files.append({
                                "file": str(file_path),
                                "chunks": len(chunks)
                            })
                        else:
                            errors.append({
                                "file": str(file_path),
                                "error": "No content extracted"
                            })
                    else:
                        errors.append({
                            "file": str(file_path),
                            "error": "Unsupported file type"
                        })
                except Exception as e:
                    errors.append({
                        "file": str(file_path),
                        "error": str(e)
                    })
        print(errors)
        return {
            "status": "success",
            "indexed_files": indexed_files,
            "errors": errors
        }

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Global RAG system and indexer
rag_system = None
indexer = None

def setup_system(data_dir: str = "./data", persist_dir: str = "./chroma_db"):
    global rag_system, indexer
    try:
        hardware_info = configure_hardware()
        print(f"Hardware Configuration: {hardware_info['message']}")
        
        llm = Ollama(model="llama3", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        indexer = DocumentIndexer(vectorstore)
        rag_system = ModularRAGSystem(vectorstore, llm)
        return {
            "status": "success",
            "message": "RAG system initialized",
            "hardware": hardware_info
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to initialize RAG system: {str(e)}",
            "hardware": configure_hardware()
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        test_llm = Ollama(model="llama3")
        test_llm.invoke("Hello")
        hardware_info = configure_hardware()
        return jsonify({
            "status": "success",
            "message": "Ollama connection successful",
            "hardware": hardware_info
        })
    except Exception as e:
        hardware_info = configure_hardware()
        return jsonify({
            "status": "error",
            "message": f"Ollama connection failed: {str(e)}. Ensure Ollama is running and llama3 and nomic-embed-text models are installed",
            "hardware": hardware_info
        }), 500

@app.route('/api/index', methods=['POST'])
def index_documents():
    global indexer
    if not indexer:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    data = request.get_json()
    data_dir = data.get('data_dir', './data')
    
    result = indexer.index_directory(data_dir)
    return jsonify(result)

@app.route('/api/query', methods=['POST'])
def process_query():
    global rag_system
    if not rag_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"status": "error", "message": "Query is required"}), 400
    
    start_time = time.time()
    response = rag_system.process_query(query)
    end_time = time.time()
    
    response_data = {
        "status": "success",
        "answer": response.answer,
        "query_type": response.query_type.value,
        "strategy_used": response.strategy_used,
        "confidence": response.confidence,
        "sources": [
            {
                "file": doc.source.split('/')[-1] if '/' in doc.source else doc.source,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "chunk_id": doc.chunk_id
            } for doc in response.sources[:3]
        ],
        "metadata": response.metadata,
        "processing_time": end_time - start_time
    }
    
    return jsonify(response_data)

if __name__ == "__main__":
    setup_result = setup_system()
    if setup_result["status"] == "error":
        print(f"Failed to start: {setup_result['message']}")
        exit(1)
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)