from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)

class TopicDetector:
    """Enhanced topic detection with better similarity calculation"""
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold

    def extract_topic(self, text: str, context_turns: List[Dict] = None) -> str:
        """Extract main topic from text with context"""
        try:
            # Combine current text with recent context for better topic detection
            full_text = text
            if context_turns:
                recent_queries = [turn.get('query', '') for turn in context_turns[-3:]]
                full_text = " ".join(recent_queries + [text])
            
            # Extract keywords
            words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # Get most frequent meaningful words
            stopwords = {'the', 'and', 'you', 'are', 'how', 'what', 'can', 'tell', 'about', 'this', 'that', 'with', 'for', 'get', 'help', 'need', 'want'}
            meaningful_words = {w: f for w, f in word_freq.items() if w not in stopwords and f >= 1 and len(w) > 3}
            
            if meaningful_words:
                # Return top 2 keywords as topic
                top_words = sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:2]
                topic = "_".join([w[0] for w in top_words])
            else:
                # Fallback to first meaningful word
                meaningful = [w for w in words if w not in stopwords and len(w) > 3]
                topic = meaningful[0] if meaningful else "general"
            
            return topic
            
        except Exception as e:
            print(f"Error extracting topic: {e}")
            return "general"

    def detect_topic_change(self, current_query: str, session_data: Dict) -> Tuple[bool, str, float]:
        """Detect if topic has changed from previous conversation"""
        try:
            # Get recent turns for context
            recent_turns = session_data.get('recent_turns', [])
            previous_topic = session_data.get('current_topic', 'general')
            
            if not recent_turns:
                new_topic = self.extract_topic(current_query)
                return True, new_topic, 1.0
            
            # Extract current topic with context
            current_topic = self.extract_topic(current_query, recent_turns)
            
            # Calculate similarity between topics
            if previous_topic == current_topic:
                return False, current_topic, 1.0
            
            # Simple keyword-based similarity
            prev_words = set(previous_topic.split('_')) if previous_topic else set()
            curr_words = set(current_topic.split('_'))
            
            if prev_words & curr_words:  # If there's any overlap
                similarity = len(prev_words & curr_words) / len(prev_words | curr_words)
            else:
                similarity = 0.0
            
            topic_changed = similarity < self.similarity_threshold
            
            return topic_changed, current_topic, similarity
            
        except Exception as e:
            print(f"Error detecting topic change: {e}")
            return True, current_topic if 'current_topic' in locals() else "general", 0.0

class EnhancedMemoryService:
    def __init__(self, db_path: str = "/data/memory.db"):
        self.db_path = db_path
        self.topic_detector = TopicDetector()
        self.init_database()

    def init_database(self):
        """Initialize comprehensive SQLite database for memory storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    current_topic TEXT,
                    user_preferences TEXT DEFAULT '{}',
                    topic_history TEXT DEFAULT '[]'
                )
            ''')
            
            # Conversation turns table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    query_type TEXT,
                    confidence REAL,
                    strategy_used TEXT,
                    sources TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    topic_labels TEXT DEFAULT '[]',
                    feedback_score REAL,
                    feedback_comments TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # Topic transitions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS topic_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    from_turn_id TEXT,
                    to_turn_id TEXT,
                    from_topic TEXT,
                    to_topic TEXT,
                    similarity_score REAL,
                    transition_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create a session with comprehensive data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT id, current_topic, user_preferences, topic_history FROM sessions WHERE id = ?',
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
                    "user_preferences": {},
                    "topic_history": []
                }
            
            # Update last activity
            conn.execute(
                'UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?',
                (session_id,)
            )
            
            return {
                "session_id": session[0],
                "current_topic": session[1],
                "user_preferences": json.loads(session[2] or '{}'),
                "topic_history": json.loads(session[3] or '[]')
            }

    def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a conversation turn with topic detection"""
        try:
            # Get session data for topic detection
            session_data = self.get_or_create_session(session_id)
            
            # Get recent turns for context
            recent_turns = self.get_recent_turns(session_id, 3)
            session_data['recent_turns'] = recent_turns
            
            # Detect topic change
            topic_changed, current_topic, similarity = self.topic_detector.detect_topic_change(
                turn_data['query'], session_data
            )
            
            with sqlite3.connect(self.db_path) as conn:
                # Insert the turn
                conn.execute('''
                    INSERT INTO conversation_turns 
                    (id, session_id, query, response, query_type, confidence, strategy_used, sources, topic_labels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    turn_data['turn_id'],
                    session_id,
                    turn_data['query'],
                    turn_data['response'],
                    turn_data['query_type'],
                    turn_data['confidence'],
                    turn_data.get('strategy_used', 'unknown'),
                    json.dumps(turn_data.get('sources', [])),
                    json.dumps([current_topic])
                ))
                
                # Handle topic transition
                if topic_changed and session_data['current_topic']:
                    # Classify transition type
                    transition_type = self._classify_transition_type(
                        session_data['current_topic'], 
                        current_topic, 
                        session_data['topic_history']
                    )
                    
                    # Record topic transition
                    conn.execute('''
                        INSERT INTO topic_transitions 
                        (session_id, from_turn_id, to_turn_id, from_topic, to_topic, 
                         similarity_score, transition_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        recent_turns[0]['turn_id'] if recent_turns else '',
                        turn_data['turn_id'],
                        session_data['current_topic'],
                        current_topic,
                        similarity,
                        transition_type
                    ))
                
                # Update session topic and history
                if topic_changed:
                    # Add old topic to history
                    if session_data['current_topic']:
                        topic_history = session_data['topic_history']
                        if session_data['current_topic'] not in topic_history:
                            topic_history.append(session_data['current_topic'])
                            # Keep only last 10 topics in history
                            if len(topic_history) > 10:
                                topic_history = topic_history[-10:]
                    else:
                        topic_history = session_data['topic_history']
                    
                    # Update session with new topic
                    conn.execute(
                        'UPDATE sessions SET current_topic = ?, topic_history = ? WHERE id = ?',
                        (current_topic, json.dumps(topic_history), session_id)
                    )
            
            return {
                "status": "success",
                "topic_changed": topic_changed,
                "current_topic": current_topic,
                "similarity_score": similarity
            }
            
        except Exception as e:
            print(f"Error adding turn: {e}")
            return {
                "status": "error",
                "message": str(e),
                "topic_changed": False,
                "current_topic": "general"
            }

    def _classify_transition_type(self, from_topic: str, to_topic: str, topic_history: List[str]) -> str:
        """Classify the type of topic transition"""
        # Check if returning to a previous topic
        if to_topic in topic_history[-5:]:
            return "return"
        
        # Check if topics share keywords (gradual transition)
        from_words = set(from_topic.split('_')) if from_topic else set()
        to_words = set(to_topic.split('_'))
        
        if from_words & to_words:
            return "gradual"
        else:
            return "abrupt"

    def get_recent_turns(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent turns for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, query, response, query_type, confidence, strategy_used, 
                           timestamp, topic_labels, feedback_score
                    FROM conversation_turns 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (session_id, limit))
                
                return [
                    {
                        "turn_id": row[0],
                        "query": row[1],
                        "response": row[2],
                        "query_type": row[3],
                        "confidence": row[4],
                        "strategy_used": row[5],
                        "timestamp": row[6],
                        "topic_labels": json.loads(row[7] or '[]'),
                        "feedback_score": row[8]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"Error getting recent turns: {e}")
            return []

    def get_conversation_context(self, session_id: str, max_turns: int = 5) -> Dict[str, Any]:
        """Get enhanced conversation context"""
        try:
            recent_turns = self.get_recent_turns(session_id, max_turns)
            
            if not recent_turns:
                return {"conversation_context": ""}
            
            # Build context string
            context_parts = []
            for i, turn in enumerate(reversed(recent_turns)):
                context_parts.append(f"Q{i+1}: {turn['query']}\nA{i+1}: {turn['response'][:200]}...")
            
            # Get session info
            session_data = self.get_or_create_session(session_id)
            
            return {
                "conversation_context": "\n".join(context_parts),
                "current_topic": session_data['current_topic'],
                "topic_history": session_data['topic_history'],
                "recent_turns": recent_turns,
                "user_preferences": session_data['user_preferences']
            }
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return {"conversation_context": ""}

    def get_session_history(self, session_id: str, max_turns: int = 20) -> Dict[str, Any]:
        """Get comprehensive session history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session info
                cursor = conn.execute(
                    'SELECT current_topic, user_preferences, topic_history, created_at, last_activity FROM sessions WHERE id = ?',
                    (session_id,)
                )
                session_info = cursor.fetchone()
                
                if not session_info:
                    return {"error": "Session not found"}
                
                # Get turns
                cursor = conn.execute('''
                    SELECT id, query, response, query_type, confidence, strategy_used,
                           timestamp, topic_labels, feedback_score, feedback_comments
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
                        "strategy_used": row[5],
                        "timestamp": row[6],
                        "topic_labels": json.loads(row[7] or '[]'),
                        "feedback_score": row[8],
                        "feedback_comments": row[9]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get topic transitions
                cursor = conn.execute('''
                    SELECT from_topic, to_topic, similarity_score, transition_type, timestamp
                    FROM topic_transitions 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                ''', (session_id,))
                
                transitions = [
                    {
                        "from_topic": row[0],
                        "to_topic": row[1],
                        "similarity_score": row[2],
                        "transition_type": row[3],
                        "timestamp": row[4]
                    }
                    for row in cursor.fetchall()
                ]
                
                return {
                    "session_id": session_id,
                    "current_topic": session_info[0],
                    "user_preferences": json.loads(session_info[1] or '{}'),
                    "topic_history": json.loads(session_info[2] or '[]'),
                    "created_at": session_info[3],
                    "last_activity": session_info[4],
                    "turns": list(reversed(turns)),
                    "topic_transitions": transitions,
                    "total_turns": len(turns)
                }
        except Exception as e:
            print(f"Error getting session history: {e}")
            return {"error": str(e)}

    def update_turn_feedback(self, session_id: str, turn_id: str, feedback_score: float, feedback_comments: str = None) -> bool:
        """Update feedback for a specific turn"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE conversation_turns 
                    SET feedback_score = ?, feedback_comments = ?
                    WHERE id = ? AND session_id = ?
                ''', (feedback_score, feedback_comments, turn_id, session_id))
                
                return conn.total_changes > 0
        except Exception as e:
            print(f"Error updating turn feedback: {e}")
            return False

# Initialize the enhanced memory service
memory_service = EnhancedMemoryService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "memory"})

@app.route('/context/<session_id>', methods=['GET'])
def get_context(session_id):
    """Get enhanced conversation context for a session"""
    max_turns = request.args.get('max_turns', 5, type=int)
    context = memory_service.get_conversation_context(session_id, max_turns)
    return jsonify(context)

@app.route('/turns', methods=['POST'])
def add_turn():
    """Add a conversation turn with topic detection"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    result = memory_service.add_turn(session_id, data)
    return jsonify(result)

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get comprehensive session history"""
    max_turns = request.args.get('max_turns', 20, type=int)
    history = memory_service.get_session_history(session_id, max_turns)
    return jsonify(history)

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all sessions with enhanced info"""
    try:
        with sqlite3.connect(memory_service.db_path) as conn:
            cursor = conn.execute('''
                SELECT s.id, s.created_at, s.last_activity, s.current_topic,
                       COUNT(ct.id) as turn_count,
                       MAX(ct.timestamp) as last_turn_time
                FROM sessions s
                LEFT JOIN conversation_turns ct ON s.id = ct.session_id
                GROUP BY s.id, s.created_at, s.last_activity, s.current_topic
                ORDER BY s.last_activity DESC
            ''')
            
            sessions = [
                {
                    "session_id": row[0],
                    "created_at": row[1],
                    "last_activity": row[2],
                    "current_topic": row[3],
                    "turn_count": row[4],
                    "last_turn_time": row[5],
                    "is_active": (datetime.now() - datetime.fromisoformat(row[2].replace('Z', '+00:00') if row[2] else datetime.now().isoformat())).total_seconds() < 3600
                }
                for row in cursor.fetchall()
            ]
            
        return jsonify({"sessions": sessions, "total_sessions": len(sessions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/topics/<session_id>', methods=['GET'])
def get_topic_transitions(session_id):
    """Get topic transition history for a session"""
    try:
        with sqlite3.connect(memory_service.db_path) as conn:
            # Get session info
            cursor = conn.execute(
                'SELECT current_topic, topic_history FROM sessions WHERE id = ?',
                (session_id,)
            )
            session_info = cursor.fetchone()
            
            if not session_info:
                return jsonify({"error": "Session not found"}), 404
            
            # Get transitions
            cursor = conn.execute('''
                SELECT from_turn_id, to_turn_id, from_topic, to_topic, 
                       similarity_score, transition_type, timestamp
                FROM topic_transitions 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', (session_id,))
            
            transitions = [
                {
                    'from_turn_id': row[0],
                    'to_turn_id': row[1],
                    'from_topic': row[2],
                    'to_topic': row[3],
                    'similarity_score': row[4],
                    'transition_type': row[5],
                    'timestamp': row[6]
                }
                for row in cursor.fetchall()
            ]
            
            return jsonify({
                'session_id': session_id,
                'current_topic': session_info[0],
                'topic_history': json.loads(session_info[1] or '[]'),
                'transitions': transitions
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback/update', methods=['POST'])
def update_feedback():
    """Update feedback for a turn"""
    data = request.get_json()
    session_id = data.get('session_id')
    turn_id = data.get('turn_id')
    feedback_score = data.get('feedback_score')
    feedback_comments = data.get('feedback_comments')
    
    if not all([session_id, turn_id, feedback_score is not None]):
        return jsonify({"error": "session_id, turn_id, and feedback_score required"}), 400
    
    success = memory_service.update_turn_feedback(session_id, turn_id, feedback_score, feedback_comments)
    
    if success:
        return jsonify({"status": "success", "message": "Feedback updated"})
    else:
        return jsonify({"error": "Failed to update feedback"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)