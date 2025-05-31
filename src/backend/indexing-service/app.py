"""
Document Indexing Service - Handles document processing and vector storage
Port: 8004
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hashlib
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)

class IndexingService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(
            persist_directory="/data/chroma_db",
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )

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

    def index_directory(self, data_dir: str) -> Dict[str, Any]:
        """Index all documents in a directory"""
        data_path = Path(data_dir)
        if not data_path.exists():
            return {"status": "error", "message": f"Directory {data_dir} does not exist"}
        
        # Clear existing data
        try:
            self.vectorstore.reset_collection()
        except Exception as e:
            print(f"Error clearing vectorstore: {e}")
        
        indexed_files = []
        errors = []
        file_hashes = set()

        for file_path in data_path.glob("**/*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
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
                        documents = loader.load()
                        chunks = self.text_splitter.split_documents(documents)
                        
                        for i, chunk in enumerate(chunks):
                            chunk.metadata['chunk_id'] = f"{file_path.name}_{i}"
                            chunk.metadata['source'] = str(file_path)
                        
                        if chunks:
                            self.vectorstore.add_documents(chunks)
                            indexed_files.append({
                                "file": str(file_path),
                                "chunks": len(chunks)
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

        return {
            "status": "success",
            "indexed_files": indexed_files,
            "errors": errors,
            "total_files": len(indexed_files),
            "total_errors": len(errors)
        }

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

indexing_service = IndexingService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "indexing"})

@app.route('/index', methods=['POST'])
def index_documents():
    """Index documents from a directory"""
    data = request.get_json()
    data_dir = data.get('data_dir', '/data/documents')
    
    if not os.path.exists(data_dir):
        return jsonify({"error": "Directory does not exist"}), 400
    
    result = indexing_service.index_directory(data_dir)
    return jsonify(result)

@app.route('/collection/info', methods=['GET'])
def get_collection_info():
    """Get collection information"""
    info = indexing_service.get_collection_info()
    return jsonify(info)

@app.route('/collection/clear', methods=['POST'])
def clear_collection():
    """Clear the vector collection"""
    try:
        indexing_service.vectorstore.reset_collection()
        return jsonify({"status": "success", "message": "Collection cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=True)