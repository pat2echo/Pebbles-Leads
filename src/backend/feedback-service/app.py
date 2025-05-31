"""
Feedback Service - Handles user feedback, analytics, and learning insights
Port: 8003
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)

class FeedbackService:
    def __init__(self, db_path: str = "/data/feedback.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    session_id TEXT,
                    score REAL NOT NULL,
                    comments TEXT,
                    query_text TEXT,
                    response_text TEXT,
                    query_type TEXT,
                    strategy_used TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    score_impact REAL NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit feedback for a turn"""
        try:
            feedback_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO feedback 
                    (id, turn_id, session_id, score, comments, query_text, response_text, 
                     query_type, strategy_used, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback_id,
                    feedback_data['turn_id'],
                    feedback_data.get('session_id'),
                    feedback_data['score'],
                    feedback_data.get('comments'),
                    feedback_data.get('query_text'),
                    feedback_data.get('response_text'),
                    feedback_data.get('query_type'),
                    feedback_data.get('strategy_used'),
                    feedback_data.get('confidence')
                ))
                
                # Update patterns
                self._update_patterns(
                    feedback_data['score'],
                    feedback_data.get('comments'),
                    feedback_data.get('query_type')
                )
            
            return {
                "status": "success",
                "feedback_id": feedback_id,
                "message": "Feedback recorded successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _update_patterns(self, score: float, comments: str, query_type: str):
        """Update feedback patterns"""
        patterns_to_update = []
        
        if comments:
            keywords = re.findall(r'\b\w+\b', comments.lower())
            for keyword in keywords:
                if keyword in ['good', 'great', 'excellent', 'helpful']:
                    patterns_to_update.append((f"positive_{keyword}", score))
                elif keyword in ['bad', 'poor', 'unhelpful', 'wrong']:
                    patterns_to_update.append((f"negative_{keyword}", 1 - score))
        
        if query_type:
            patterns_to_update.append((f"query_type_{query_type}", score))
        
        with sqlite3.connect(self.db_path) as conn:
            for pattern, impact in patterns_to_update:
                conn.execute('''
                    INSERT OR REPLACE INTO feedback_patterns 
                    (pattern_type, pattern_value, score_impact, frequency, last_updated)
                    VALUES (?, ?, ?, 
                            COALESCE((SELECT frequency FROM feedback_patterns WHERE pattern_value = ?) + 1, 1),
                            CURRENT_TIMESTAMP)
                ''', ("feedback", pattern, impact, pattern))

    def get_global_summary(self) -> Dict[str, Any]:
        """Get global feedback summary"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT score FROM feedback')
            scores = [row[0] for row in cursor.fetchall()]
            
            if not scores:
                return {
                    "total_feedback": 0,
                    "average_score": 0,
                    "score_distribution": {"high": 0, "medium": 0, "low": 0}
                }
            
            return {
                "total_feedback": len(scores),
                "average_score": sum(scores) / len(scores),
                "score_distribution": {
                    "high": sum(1 for s in scores if s >= 0.8),
                    "medium": sum(1 for s in scores if 0.4 <= s < 0.8),
                    "low": sum(1 for s in scores if s < 0.4)
                }
            }

    def get_improvements(self) -> Dict[str, Any]:
        """Get system improvement suggestions"""
        with sqlite3.connect(self.db_path) as conn:
            # Query type performance
            cursor = conn.execute('''
                SELECT AVG(score), query_type, COUNT(*)
                FROM feedback 
                WHERE query_type IS NOT NULL
                GROUP BY query_type
                ORDER BY AVG(score) DESC
            ''')
            
            query_performance = {
                row[1]: {"avg_score": row[0], "count": row[2]}
                for row in cursor.fetchall()
            }
            
            # Common issues
            cursor = conn.execute('''
                SELECT comments, COUNT(*) as frequency
                FROM feedback 
                WHERE score < 0.4 AND comments IS NOT NULL
                GROUP BY comments
                ORDER BY frequency DESC
                LIMIT 5
            ''')
            
            common_issues = [
                {"issue": row[0], "frequency": row[1]}
                for row in cursor.fetchall()
            ]
            
            return {
                "query_type_performance": query_performance,
                "common_issues": common_issues,
                "improvement_suggestions": self._generate_suggestions(query_performance, common_issues)
            }

    def _generate_suggestions(self, query_perf: Dict, issues: List) -> List[Dict]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if query_perf:
            low_performing = [
                qt for qt, perf in query_perf.items() 
                if perf["avg_score"] < 0.6
            ]
            if low_performing:
                suggestions.append({
                    "type": "query_improvement",
                    "suggestion": f"Focus on improving {', '.join(low_performing)} query types"
                })
        
        if issues:
            suggestions.append({
                "type": "content_improvement", 
                "suggestion": f"Address common concerns: {'; '.join([i['issue'] for i in issues[:3]])}"
            })
        
        return suggestions

feedback_service = FeedbackService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "feedback"})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    data = request.get_json()
    
    required_fields = ['turn_id', 'score']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "turn_id and score are required"}), 400
    
    if not 0.0 <= data['score'] <= 1.0:
        return jsonify({"error": "Score must be between 0.0 and 1.0"}), 400
    
    result = feedback_service.submit_feedback(data)
    return jsonify(result)

@app.route('/summary/global', methods=['GET'])
def get_global_summary():
    """Get global feedback summary"""
    summary = feedback_service.get_global_summary()
    return jsonify(summary)

@app.route('/improvements', methods=['GET'])
def get_improvements():
    """Get improvement suggestions"""
    improvements = feedback_service.get_improvements()
    return jsonify(improvements)

@app.route('/analytics/<session_id>', methods=['GET'])
def get_session_analytics(session_id):
    """Get analytics for a specific session"""
    with sqlite3.connect(feedback_service.db_path) as conn:
        cursor = conn.execute('''
            SELECT score, query_type, timestamp 
            FROM feedback 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        ''', (session_id,))
        
        feedback_data = cursor.fetchall()
        
        if not feedback_data:
            return jsonify({
                "session_id": session_id,
                "total_feedback": 0,
                "message": "No feedback data available"
            })
        
        scores = [row[0] for row in feedback_data]
        query_types = defaultdict(int)
        for row in feedback_data:
            if row[1]:
                query_types[row[1]] += 1
        
        analytics = {
            "session_id": session_id,
            "total_feedback": len(scores),
            "average_score": sum(scores) / len(scores),
            "query_type_distribution": dict(query_types),
            "latest_feedback": feedback_data[0][2] if feedback_data else None
        }
        
    return jsonify(analytics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True)