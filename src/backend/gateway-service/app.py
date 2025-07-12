"""
API Gateway - Routes requests to appropriate microservices
Port: 8000 (main entry point)
"""

import os
import logging
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
from typing import Dict, Any
import uuid
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

app = Flask(__name__)
CORS(app)

# Production configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production'),
    DEBUG=False,
    TESTING=False,
    ENV='production'
)

# Service URLs
SERVICES = {
    'core_rag': os.getenv('CORE_RAG_SERVICE_URL', 'http://core-rag-service:8000'),
    'memory': os.getenv('MEMORY_SERVICE_URL', 'http://memory-service:8000'),
    'feedback': os.getenv('FEEDBACK_SERVICE_URL', 'http://feedback-service:8000'),
    'indexing': os.getenv('INDEXING_SERVICE_URL', 'http://indexing-service:8000')
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

# Middleware for request tracking
@app.before_request
def before_request():
    """Track request metrics"""
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()

@app.after_request
def after_request(response):
    """Log requests in production"""
    logger.info(f"{request.method} {request.path} - {response.status_code}")
    return response

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for API Gateway and all downstream services"""
    try:
        # Check own service health
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        gateway_health = {
            'hi': 'hip',
            'status': 'healthy',
            'service': 'api-gateway',
            'memory_usage_percent': memory_usage,
            'cpu_usage_percent': cpu_usage,
            'timestamp': int(time.time())
        }
        
        # Check if gateway itself is unhealthy
        if memory_usage > 90 or cpu_usage > 90:
            gateway_health['status'] = 'unhealthy'
        
        # Check downstream services
        services_health = {}
        for service_name, service_url in SERVICES.items():
            try:
                response = requests.get(f"{service_url}/health", timeout=20)
                services_health[service_name] = {
                    "service_url": service_url,
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "note": "" if response.status_code == 200 else "possibly due to ollama not running, test it via http://localhost:11434/api/tags",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                services_health[service_name] = {
                    "status": "unhealthy",
                    "note": "possibly due to ollama not running, test it via http://localhost:11434/api/tags",
                    "error": str(e)
                }
        
        # Determine overall status
        overall_status = "healthy"
        if gateway_health['status'] == 'unhealthy':
            overall_status = "unhealthy"
        elif any(service["status"] == "unhealthy" for service in services_health.values()):
            overall_status = "degraded"  # Gateway is healthy but some services aren't
        
        response_data = {
            "overall_status": overall_status,
            "gateway": gateway_health,
            "services": services_health
        }
        
        # Return 503 if gateway itself is unhealthy, 200 otherwise
        status_code = 503 if gateway_health['status'] == 'unhealthy' else 200
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': int(time.time())
        }), 503

# Metrics endpoint for Prometheus monitoring
@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

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

@app.route('/api/feedback/insights', methods=['GET'])
def get_feedback_insights():
    """Get learning insights from feedback"""
    return proxy_request(
        SERVICES['feedback'],
        '/insights',
        method='GET',
        params=request.args
    )

@app.route('/api/topics/<session_id>', methods=['GET'])
def get_topic_transitions(session_id):
    """Get topic transitions for a session"""
    return proxy_request(
        SERVICES['memory'],
        f'/topics/{session_id}',
        method='GET'
    )

@app.route('/api/feedback/session/<session_id>', methods=['GET'])
def get_session_feedback_summary(session_id):
    """Get feedback summary for a session"""
    return proxy_request(
        SERVICES['feedback'],
        f'/summary/session/{session_id}',
        method='GET'
    )

if __name__ == '__main__':
    # This will only run in development
    # In production, Gunicorn will handle the app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)