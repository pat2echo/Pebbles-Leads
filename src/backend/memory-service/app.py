"""
Memory Service - Handles conversation memory, sessions, and topic detection
Port: 8002
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)

class MemoryService:
    def __init__(self, db_path: str = "/data/memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for memory storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    current_topic TEXT,
                    user_preferences TEXT DEFAULT '{}'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    query_type TEXT,
                    confidence REAL,
                    sources TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    topic_labels TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    from_topic TEXT,
                    to_topic TEXT,
                    transition_type TEXT,
                    similarity_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT id, current_topic, user_preferences FROM sessions WHERE id = ?',
                (session_id,)
            )
            session = cursor.fetchone()
            
            if not session:
                conn.execute(
                    'INSERT INTO sessions (id) VALUES (?)',
                    (session_id,)
                )
                return {
                    "session_id": session_id,
                    "current_topic": None,
                    "user_preferences": {}
                }
            
            # Update last activity
            conn.execute(
                'UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?',
                (session_id,)
            )
            
            return {
                "session_id": session[0],
                "current_topic": session[1],
                "user_preferences": json.loads(session[2] or '{}')
            }

    def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """Add a conversation turn"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Detect topic (simplified)
                current_topic = self.extract_topic(turn_data['query'])
                
                conn.execute('''
                    INSERT INTO conversation_turns 
                    (id, session_id, query, response, query_type, confidence, sources, topic_labels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    turn_data['turn_id'],
                    session_id,
                    turn_data['query'],
                    turn_data['response'],
                    turn_data['query_type'],
                    turn_data['confidence'],
                    json.dumps(turn_data.get('sources', [])),
                    json.dumps([current_topic])
                ))
                
                # Update session topic
                conn.execute(
                    'UPDATE sessions SET current_topic = ? WHERE id = ?',
                    (current_topic, session_id)
                )
                
            return True
        except Exception as e:
            print(f"Error adding turn: {e}")
            return False

    def extract_topic(self, text: str) -> str:
        """Simple topic extraction"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {'the', 'and', 'you', 'are', 'how', 'what', 'can', 'tell', 'about', 'this', 'that'}
        meaningful = [w for w in words if w not in stopwords]
        return meaningful[0] if meaningful else "general"

    def get_conversation_context(self, session_id: str, max_turns: int = 5) -> Dict[str, Any]:
        """Get recent conversation context"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT query, response FROM conversation_turns 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, max_turns))
            
            turns = cursor.fetchall()
            
            if not turns:
                return {"conversation_context": ""}
            
            context_parts = []
            for i, (query, response) in enumerate(reversed(turns)):
                context_parts.append(f"Q{i+1}: {query}\nA{i+1}: {response[:200]}...")
            
            return {"conversation_context": "\n".join(context_parts)}

    def get_session_history(self, session_id: str, max_turns: int = 20) -> Dict[str, Any]:
        """Get full session history"""
        with sqlite3.connect(self.db_path) as conn:
            # Get session info
            cursor = conn.execute(
                'SELECT current_topic, user_preferences FROM sessions WHERE id = ?',
                (session_id,)
            )
            session_info = cursor.fetchone()
            
            if not session_info:
                return {"error": "Session not found"}
            
            # Get turns
            cursor = conn.execute('''
                SELECT id, query, response, query_type, confidence, timestamp, topic_labels
                FROM conversation_turns 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, max_turns))
            
            turns = [
                {
                    "turn_id": row[0],
                    "query": row[1],
                    "response": row[2],
                    "query_type": row[3],
                    "confidence": row[4],
                    "timestamp": row[5],
                    "topic_labels": json.loads(row[6] or '[]')
                }
                for row in cursor.fetchall()
            ]
            
            return {
                "session_id": session_id,
                "current_topic": session_info[0],
                "user_preferences": json.loads(session_info[1] or '{}'),
                "turns": list(reversed(turns))
            }

memory_service = MemoryService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "memory"})

@app.route('/context/<session_id>', methods=['GET'])
def get_context(session_id):
    """Get conversation context for a session"""
    max_turns = request.args.get('max_turns', 5, type=int)
    context = memory_service.get_conversation_context(session_id, max_turns)
    return jsonify(context)

@app.route('/turns', methods=['POST'])
def add_turn():
    """Add a conversation turn"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    success = memory_service.add_turn(session_id, data)
    if success:
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Failed to add turn"}), 500

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get full session history"""
    max_turns = request.args.get('max_turns', 20, type=int)
    history = memory_service.get_session_history(session_id, max_turns)
    return jsonify(history)

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all sessions"""
    with sqlite3.connect(memory_service.db_path) as conn:
        cursor = conn.execute('''
            SELECT id, created_at, last_activity, current_topic,
                   (SELECT COUNT(*) FROM conversation_turns WHERE session_id = sessions.id) as turn_count
            FROM sessions 
            ORDER BY last_activity DESC
        ''')
        
        sessions = [
            {
                "session_id": row[0],
                "created_at": row[1],
                "last_activity": row[2],
                "current_topic": row[3],
                "turn_count": row[4]
            }
            for row in cursor.fetchall()
        ]
        
    return jsonify({"sessions": sessions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)