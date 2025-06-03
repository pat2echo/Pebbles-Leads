"""
API Gateway - Routes requests to appropriate microservices
Port: 8000 (main entry point)
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
from typing import Dict, Any
import uuid

app = Flask(__name__)
CORS(app)

# Service URLs
SERVICES = {
    'core_rag': os.getenv('CORE_RAG_SERVICE_URL', 'http://core-rag-service:8001'),
    'memory': os.getenv('MEMORY_SERVICE_URL', 'http://memory-service:8002'),
    'feedback': os.getenv('FEEDBACK_SERVICE_URL', 'http://feedback-service:8003'),
    'indexing': os.getenv('INDEXING_SERVICE_URL', 'http://indexing-service:8004')
}

def proxy_request(service_url: str, path: str = '', method: str = 'GET', 
                 data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Response:
    """Proxy request to a microservice"""
    try:
        url = f"{service_url}{path}"
        
        if method == 'GET':
            response = requests.get(url, params=params, timeout=300)
        elif method == 'POST':
            response = requests.post(url, json=data, params=params, timeout=300)
        elif method == 'PUT':
            response = requests.put(url, json=data, params=params, timeout=300)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, timeout=300)
        else:
            return jsonify({"error": "Unsupported method"}), 400
        
        return Response(
            response.content,
            status=response.status_code,
            headers={'Content-Type': 'application/json'}
        )
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Service timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Service unavailable"}), 503
    except Exception as e:
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for all services"""
    health_status = {}
    
    for service_name, service_url in SERVICES.items():
        try:
            response = requests.get(f"{service_url}/health", timeout=30)
            health_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    overall_status = "healthy" if all(
        service["status"] == "healthy" for service in health_status.values()
    ) else "unhealthy"
    
    return jsonify({
        "overall_status": overall_status,
        "services": health_status
    })

# ==============================================
# RAG QUERY ENDPOINTS
# ==============================================

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a RAG query"""
    data = request.get_json()
    
    # Generate session ID if not provided
    if not data.get('session_id'):
        data['session_id'] = str(uuid.uuid4())
    
    # Route to core RAG service
    return proxy_request(
        SERVICES['core_rag'],
        '/process',
        method='POST',
        data=data
    )

# ==============================================
# MEMORY SERVICE ENDPOINTS
# ==============================================

@app.route('/api/conversation/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history"""
    return proxy_request(
        SERVICES['memory'],
        f'/sessions/{session_id}',
        method='GET',
        params=request.args
    )

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all sessions"""
    return proxy_request(
        SERVICES['memory'],
        '/sessions',
        method='GET'
    )

# ==============================================
# FEEDBACK SERVICE ENDPOINTS
# ==============================================

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    return proxy_request(
        SERVICES['feedback'],
        '/feedback',
        method='POST',
        data=request.get_json()
    )

@app.route('/api/feedback/summary/global', methods=['GET'])
def get_global_feedback():
    """Get global feedback summary"""
    return proxy_request(
        SERVICES['feedback'],
        '/summary/global',
        method='GET'
    )

@app.route('/api/feedback/improvements', methods=['GET'])
def get_improvements():
    """Get improvement suggestions"""
    return proxy_request(
        SERVICES['feedback'],
        '/improvements',
        method='GET'
    )

@app.route('/api/analytics/<session_id>', methods=['GET'])
def get_session_analytics(session_id):
    """Get session analytics"""
    return proxy_request(
        SERVICES['feedback'],
        f'/analytics/{session_id}',
        method='GET'
    )

# ==============================================
# INDEXING SERVICE ENDPOINTS
# ==============================================

@app.route('/api/index', methods=['POST'])
def index_documents():
    """Index documents"""
    return proxy_request(
        SERVICES['indexing'],
        '/index',
        method='POST',
        data=request.get_json()
    )

@app.route('/api/collection/info', methods=['GET'])
def get_collection_info():
    """Get collection info"""
    return proxy_request(
        SERVICES['indexing'],
        '/collection/info',
        method='GET'
    )

@app.route('/api/collection/clear', methods=['POST'])
def clear_collection():
    """Clear collection"""
    return proxy_request(
        SERVICES['indexing'],
        '/collection/clear',
        method='POST'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
