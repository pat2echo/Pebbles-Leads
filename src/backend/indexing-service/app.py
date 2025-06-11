from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid

# Import your existing langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class IndexingTask:
    task_id: str
    status: TaskStatus
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

class EnhancedIndexingService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url="http://ollama:11434"
        )
        self.vectorstore = Chroma(
            persist_directory="/data/chroma_db",
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        self.tasks: Dict[str, IndexingTask] = {}
        
    def get_loader_for_file(self, file_path: Path):
        """Get appropriate document loader for file type"""
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return TextLoader(str(file_path))
        elif suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix == ".csv":
            return CSVLoader(str(file_path))
        return None

    def start_async_indexing(self, data_dir: str) -> str:
        """Start asynchronous indexing and return task ID"""
        task_id = str(uuid.uuid4())
        task = IndexingTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="Task created",
            created_at=datetime.now()
        )
        self.tasks[task_id] = task
        
        # Start indexing in background thread
        thread = threading.Thread(
            target=self._index_directory_async,
            args=(task_id, data_dir)
        )
        thread.daemon = True
        thread.start()
        
        return task_id

    def _index_directory_async(self, task_id: str, data_dir: str):
        """Background indexing process"""
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.message = "Starting indexing process"
        
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                task.status = TaskStatus.FAILED
                task.error = f"Directory {data_dir} does not exist"
                return
            
            # Clear existing data
            task.message = "Clearing existing vectorstore"
            task.progress = 5.0
            try:
                self.vectorstore.reset_collection()
            except Exception as e:
                print(f"Error clearing vectorstore: {e}")
            
            # Get all files first to calculate progress
            all_files = [
                f for f in data_path.glob("**/*") 
                if f.is_file() and not f.name.startswith(".")
            ]
            
            if not all_files:
                task.status = TaskStatus.COMPLETED
                task.message = "No files found to index"
                task.progress = 100.0
                task.result = {
                    "status": "success",
                    "indexed_files": [],
                    "errors": [],
                    "total_files": 0,
                    "total_errors": 0
                }
                task.completed_at = datetime.now()
                return
            
            indexed_files = []
            errors = []
            file_hashes = set()
            
            for i, file_path in enumerate(all_files):
                try:
                    # Update progress
                    progress = 10 + (i / len(all_files)) * 85  # 10-95% for processing
                    task.progress = progress
                    task.message = f"Processing {file_path.name} ({i+1}/{len(all_files)})"
                    
                    # Check for duplicates
                    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                    if file_hash in file_hashes:
                        errors.append({
                            "file": str(file_path),
                            "error": "Duplicate file skipped"
                        })
                        continue
                    file_hashes.add(file_hash)

                    # Load and process document
                    loader = self.get_loader_for_file(file_path)
                    if loader:
                        task.message = f"Loading {file_path.name}"
                        documents = loader.load()
                        
                        task.message = f"Splitting {file_path.name} into chunks"
                        chunks = self.text_splitter.split_documents(documents)
                        
                        # Add metadata to chunks
                        for j, chunk in enumerate(chunks):
                            chunk.metadata['chunk_id'] = f"{file_path.name}_{j}"
                            chunk.metadata['source'] = str(file_path)
                        
                        if chunks:
                            task.message = f"Creating embeddings for {file_path.name} ({len(chunks)} chunks)"
                            
                            # Process chunks in smaller batches to avoid timeout
                            batch_size = 10 if file_path.suffix.lower() == '.csv' else 20
                            for batch_start in range(0, len(chunks), batch_size):
                                batch_end = min(batch_start + batch_size, len(chunks))
                                chunk_batch = chunks[batch_start:batch_end]
                                
                                # Add batch to vectorstore
                                self.vectorstore.add_documents(chunk_batch)
                                
                                # Small delay to prevent overwhelming the embedding service
                                time.sleep(0.1)
                            
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
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            task.message = "Indexing completed successfully"
            task.result = {
                "status": "success",
                "indexed_files": indexed_files,
                "errors": errors,
                "total_files": len(indexed_files),
                "total_errors": len(errors)
            }
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.message = f"Indexing failed: {str(e)}"
            task.completed_at = datetime.now()

    def get_task_status(self, task_id: str) -> Optional[IndexingTask]:
        """Get task status"""
        return self.tasks.get(task_id)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "status": "success",
                "document_count": count,
                "collection_name": collection.name
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def clear_collection(self) -> Dict[str, Any]:
        """Clear the vector collection"""
        try:
            self.vectorstore.reset_collection()
            return {"status": "success", "message": "Collection cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def get_first_five_index_entries(self) -> Dict[str, Any]:
        """Get the first 5 entries in the index"""
        try:
            collection = self.vectorstore._collection
            results = collection.get(limit=5)
            return {
                "status": "success",
                "documents": [
                    {
                        "id": doc_id,
                        "metadata": metadata,
                        "content": content
                    } for doc_id, metadata, content in zip(
                        results.get('ids', []),
                        results.get('metadatas', []),
                        results.get('documents', [])
                    )
                ],
                "total_returned": len(results.get('ids', []))
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

# Initialize service
indexing_service = EnhancedIndexingService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "indexing"})

@app.route('/index', methods=['POST'])
def start_indexing():
    """Start asynchronous document indexing"""
    data = request.get_json()
    data_dir = data.get('data_dir', '/data/documents')
    
    if not os.path.exists(data_dir):
        return jsonify({"error": "Directory does not exist"}), 400
    
    task_id = indexing_service.start_async_indexing(data_dir)
    
    return jsonify({
        "status": "accepted",
        "task_id": task_id,
        "message": "Indexing started in background",
        "status_url": f"/index/status/{task_id}"
    }), 202

@app.route('/index/status/<task_id>', methods=['GET'])
def get_indexing_status(task_id: str):
    """Get indexing task status"""
    task = indexing_service.get_task_status(task_id)
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    response = {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }
    
    if task.status == TaskStatus.COMPLETED and task.result:
        response["result"] = task.result
    elif task.status == TaskStatus.FAILED and task.error:
        response["error"] = task.error
    
    return jsonify(response)

@app.route('/collection/info', methods=['GET'])
def get_collection_info():
    """Get collection information"""
    info = indexing_service.get_collection_info()
    return jsonify(info)

@app.route('/collection/clear', methods=['POST'])
def clear_collection():
    """Clear the vector collection"""
    result = indexing_service.clear_collection()
    return jsonify(result)

@app.route('/tasks', methods=['GET'])
def list_tasks():
    """List all indexing tasks"""
    tasks = []
    for task in indexing_service.tasks.values():
        tasks.append({
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        })
    
    return jsonify({"tasks": tasks})

@app.route('/index/first-five', methods=['GET'])
def get_first_five_entries():
    """Get the first 5 entries in the index"""
    result = indexing_service.get_first_five_index_entries()
    return jsonify(result)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=True)