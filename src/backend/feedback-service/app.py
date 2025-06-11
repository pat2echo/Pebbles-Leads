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

class EnhancedFeedbackService:
    def __init__(self, db_path: str = "/data/feedback.db"):
        self.db_path = db_path
        self.feedback_storage = defaultdict(list)  # In-memory cache
        self.feedback_patterns = defaultdict(float)
        self.learning_enabled = True
        self.init_database()
        self.load_existing_feedback()

    def init_database(self):
        """Initialize comprehensive SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Enhanced feedback table
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Feedback patterns table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    score_impact REAL NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern_value)
                )
            ''')
            
            # System learning insights table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learning_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    effectiveness_score REAL,
                    usage_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_used DATETIME
                )
            ''')

    def load_existing_feedback(self):
        """Load existing feedback from database into memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load feedback into memory cache
                cursor = conn.execute('SELECT turn_id, score, comments FROM feedback')
                for turn_id, score, comments in cursor.fetchall():
                    self.feedback_storage[turn_id].append({
                        "score": score,
                        "comments": comments
                    })
                
                # Load patterns
                cursor = conn.execute('SELECT pattern_value, score_impact FROM feedback_patterns')
                for pattern_value, score_impact in cursor.fetchall():
                    self.feedback_patterns[pattern_value] = score_impact
        except Exception as e:
            print(f"Error loading existing feedback: {e}")

    def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced feedback submission with full context"""
        try:
            feedback_id = str(uuid.uuid4())
            
            # Store in memory cache
            feedback = {
                "turn_id": feedback_data['turn_id'],
                "score": feedback_data['score'],
                "comments": feedback_data.get('comments'),
                "timestamp": datetime.now()
            }
            self.feedback_storage[feedback_data['turn_id']].append(feedback)
            
            # Store in database
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
                
                # Update patterns and learning
                self._update_patterns(
                    feedback_data['score'],
                    feedback_data.get('comments'),
                    feedback_data.get('query_type'),
                    feedback_data.get('strategy_used')
                )
                
                # Extract learning insights
                if self.learning_enabled:
                    self._extract_learning_insights(
                        feedback_data['score'],
                        feedback_data.get('comments'),
                        feedback_data.get('query_text'),
                        feedback_data.get('response_text')
                    )
            
            return {
                "status": "success",
                "feedback_id": feedback_id,
                "message": "Feedback recorded and analyzed",
                "learning_applied": self.learning_enabled
            }
            
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _update_patterns(self, score: float, comments: str, query_type: str = None, strategy_used: str = None):
        """Enhanced pattern analysis"""
        patterns_to_update = []
        
        if comments:
            # Extract keywords from comments
            keywords = re.findall(r'\b\w+\b', comments.lower())
            for keyword in keywords:
                if keyword in ['good', 'great', 'excellent', 'helpful', 'clear', 'accurate']:
                    impact = score
                    patterns_to_update.append((f"positive_{keyword}", impact))
                elif keyword in ['bad', 'poor', 'unhelpful', 'wrong', 'unclear', 'confusing']:
                    impact = 1 - score
                    patterns_to_update.append((f"negative_{keyword}", impact))
        
        # Strategy effectiveness
        if strategy_used:
            patterns_to_update.append((f"strategy_{strategy_used}", score))
        
        # Query type effectiveness
        if query_type:
            patterns_to_update.append((f"query_type_{query_type}", score))
        
        # Update in memory and database
        try:
            with sqlite3.connect(self.db_path) as conn:
                for pattern, impact in patterns_to_update:
                    self.feedback_patterns[pattern] += impact
                    
                    # Update database with proper UPSERT
                    conn.execute('''
                        INSERT INTO feedback_patterns (pattern_type, pattern_value, score_impact, frequency, last_updated)
                        VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                        ON CONFLICT(pattern_value) DO UPDATE SET
                            score_impact = score_impact + excluded.score_impact,
                            frequency = frequency + 1,
                            last_updated = CURRENT_TIMESTAMP
                    ''', ("feedback", pattern, impact))
        except Exception as e:
            print(f"Error updating patterns: {e}")

    def _extract_learning_insights(self, score: float, comments: str, query_text: str, response_text: str):
        """Extract learning insights for system improvement"""
        insights = []
        
        if score >= 0.8 and comments:
            # High-quality responses - learn what worked
            insights.append({
                "type": "successful_response_pattern",
                "content": json.dumps({
                    "query_pattern": query_text[:100] if query_text else "",
                    "response_pattern": response_text[:200] if response_text else "",
                    "success_keywords": re.findall(r'\b\w+\b', comments.lower())
                }),
                "effectiveness_score": score
            })
        
        elif score <= 0.3 and comments:
            # Poor responses - learn what to avoid
            insights.append({
                "type": "failed_response_pattern", 
                "content": json.dumps({
                    "query_pattern": query_text[:100] if query_text else "",
                    "failure_keywords": re.findall(r'\b\w+\b', comments.lower()),
                    "issues": comments
                }),
                "effectiveness_score": 1 - score
            })
        
        # Store insights in database
        if insights:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for insight in insights:
                        conn.execute('''
                            INSERT INTO system_learning 
                            (learning_type, content, effectiveness_score)
                            VALUES (?, ?, ?)
                        ''', (insight["type"], insight["content"], insight["effectiveness_score"]))
            except Exception as e:
                print(f"Error storing learning insights: {e}")

    def get_learning_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get learning insights for system improvement"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT learning_type, content, effectiveness_score, usage_count
                    FROM system_learning 
                    ORDER BY effectiveness_score DESC, created_at DESC
                    LIMIT ?
                ''', (limit,))
                
                return [
                    {
                        "type": row[0],
                        "content": json.loads(row[1]) if row[1] else {},
                        "effectiveness": row[2],
                        "usage_count": row[3]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"Error getting learning insights: {e}")
            return []

    def get_system_improvements(self) -> Dict[str, Any]:
        """Get comprehensive system improvement suggestions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Strategy effectiveness
                cursor = conn.execute('''
                    SELECT AVG(score), strategy_used, COUNT(*)
                    FROM feedback 
                    WHERE strategy_used IS NOT NULL
                    GROUP BY strategy_used
                    ORDER BY AVG(score) DESC
                ''')
                strategy_performance = {
                    row[1]: {"avg_score": row[0], "count": row[2]}
                    for row in cursor.fetchall()
                }
                
                # Query type effectiveness
                cursor = conn.execute('''
                    SELECT AVG(score), query_type, COUNT(*)
                    FROM feedback 
                    WHERE query_type IS NOT NULL
                    GROUP BY query_type
                    ORDER BY AVG(score) DESC
                ''')
                query_type_performance = {
                    row[1]: {"avg_score": row[0], "count": row[2]}
                    for row in cursor.fetchall()
                }
                
                # Common issues
                cursor = conn.execute('''
                    SELECT comments, COUNT(*) as frequency
                    FROM feedback 
                    WHERE score < 0.4 AND comments IS NOT NULL AND comments != ''
                    GROUP BY comments
                    ORDER BY frequency DESC
                    LIMIT 5
                ''')
                common_issues = [
                    {"issue": row[0], "frequency": row[1]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    "strategy_performance": strategy_performance,
                    "query_type_performance": query_type_performance,
                    "common_issues": common_issues,
                    "improvement_suggestions": self._generate_improvement_suggestions(
                        strategy_performance, query_type_performance, common_issues
                    )
                }
        except Exception as e:
            print(f"Error getting system improvements: {e}")
            return {
                "strategy_performance": {},
                "query_type_performance": {},
                "common_issues": [],
                "improvement_suggestions": []
            }

    def _generate_improvement_suggestions(self, strategy_perf: Dict, query_perf: Dict, issues: List) -> List[Dict]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Strategy suggestions
        if strategy_perf and len(strategy_perf) > 1:
            best_strategy = max(strategy_perf.items(), key=lambda x: x[1]["avg_score"])
            worst_strategy = min(strategy_perf.items(), key=lambda x: x[1]["avg_score"])
            
            if best_strategy[1]["avg_score"] - worst_strategy[1]["avg_score"] > 0.2:
                suggestions.append({
                    "type": "strategy_optimization",
                    "suggestion": f"Consider using '{best_strategy[0]}' strategy more often (avg score: {best_strategy[1]['avg_score']:.2f}) and improving '{worst_strategy[0]}' strategy (avg score: {worst_strategy[1]['avg_score']:.2f})"
                })
        
        # Query type suggestions
        if query_perf:
            low_performing_types = [
                qt for qt, perf in query_perf.items() 
                if perf["avg_score"] < 0.6
            ]
            if low_performing_types:
                suggestions.append({
                    "type": "query_type_improvement",
                    "suggestion": f"Focus on improving handling of {', '.join(low_performing_types)} query types"
                })
        
        # Issue-based suggestions
        if issues:
            frequent_issues = [issue["issue"] for issue in issues[:3]]
            suggestions.append({
                "type": "content_improvement",
                "suggestion": f"Address these common user concerns: {'; '.join(frequent_issues)}"
            })
        
        return suggestions

    def get_global_summary(self) -> Dict[str, Any]:
        """Get comprehensive global feedback summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT score FROM feedback')
                all_scores = [row[0] for row in cursor.fetchall()]
                
                if not all_scores:
                    return {
                        "total_feedback": 0,
                        "average_score": 0,
                        "score_distribution": {"high": 0, "medium": 0, "low": 0},
                        "patterns": {},
                        "learning_insights_count": 0
                    }
                
                return {
                    "total_feedback": len(all_scores),
                    "average_score": sum(all_scores) / len(all_scores),
                    "score_distribution": {
                        "high": sum(1 for s in all_scores if s >= 0.8),
                        "medium": sum(1 for s in all_scores if 0.4 <= s < 0.8),
                        "low": sum(1 for s in all_scores if s < 0.4)
                    },
                    "patterns": dict(self.feedback_patterns),
                    "learning_insights_count": len(self.get_learning_insights())
                }
        except Exception as e:
            print(f"Error getting global summary: {e}")
            return {
                "total_feedback": 0,
                "average_score": 0,
                "score_distribution": {"high": 0, "medium": 0, "low": 0},
                "patterns": {},
                "learning_insights_count": 0
            }

    def get_feedback_summary(self, turn_ids: List[str] = None, include_all_time: bool = False) -> Dict[str, Any]:
        """Enhanced feedback summary with persistent data"""
        if include_all_time:
            return self.get_global_summary()
        else:
            # Use existing logic for specific turns
            if turn_ids:
                relevant_feedback = []
                for turn_id in turn_ids:
                    relevant_feedback.extend(self.feedback_storage.get(turn_id, []))
            else:
                relevant_feedback = []
                for feedbacks in self.feedback_storage.values():
                    relevant_feedback.extend(feedbacks)
            
            if not relevant_feedback:
                return {"average_score": 0, "total_feedback": 0, "patterns": {}}
            
            scores = [f["score"] for f in relevant_feedback]
            return {
                "average_score": sum(scores) / len(scores),
                "total_feedback": len(relevant_feedback),
                "score_distribution": {
                    "high": sum(1 for s in scores if s >= 0.8),
                    "medium": sum(1 for s in scores if 0.4 <= s < 0.8),
                    "low": sum(1 for s in scores if s < 0.4)
                },
                "patterns": dict(self.feedback_patterns)
            }

# Initialize the enhanced feedback service
feedback_service = EnhancedFeedbackService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "feedback"})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit comprehensive feedback"""
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
    """Get comprehensive global feedback summary"""
    summary = feedback_service.get_global_summary()
    return jsonify({"status": "success", "global_feedback": summary})

@app.route('/insights', methods=['GET'])
def get_learning_insights():
    """Get learning insights from feedback"""
    limit = request.args.get('limit', 10, type=int)
    insights = feedback_service.get_learning_insights(limit)
    return jsonify({
        "status": "success",
        "insights": insights,
        "total_count": len(insights)
    })

@app.route('/improvements', methods=['GET'])
def get_improvements():
    """Get comprehensive improvement suggestions"""
    improvements = feedback_service.get_system_improvements()
    return jsonify({
        "status": "success",
        "improvements": improvements
    })

@app.route('/analytics/<session_id>', methods=['GET'])
def get_session_analytics(session_id):
    """Get comprehensive analytics for a specific session"""
    try:
        with sqlite3.connect(feedback_service.db_path) as conn:
            cursor = conn.execute('''
                SELECT score, query_type, timestamp, confidence, strategy_used
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
            strategies = defaultdict(list)
            confidences = []
            
            for row in feedback_data:
                if row[1]:  # query_type
                    query_types[row[1]] += 1
                if row[3] is not None:  # confidence
                    confidences.append(row[3])
                if row[4]:  # strategy_used
                    strategies[row[4]].append(row[0])  # strategy -> scores
            
            # Calculate strategy performance
            strategy_performance = {}
            for strategy, strategy_scores in strategies.items():
                strategy_performance[strategy] = {
                    "avg_score": sum(strategy_scores) / len(strategy_scores),
                    "count": len(strategy_scores)
                }
            
            analytics = {
                "session_id": session_id,
                "total_feedback": len(scores),
                "average_score": sum(scores) / len(scores),
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "query_type_distribution": dict(query_types),
                "strategy_performance": strategy_performance,
                "score_distribution": {
                    "high": sum(1 for s in scores if s >= 0.8),
                    "medium": sum(1 for s in scores if 0.4 <= s < 0.8),
                    "low": sum(1 for s in scores if s < 0.4)
                },
                "latest_feedback": feedback_data[0][2] if feedback_data else None
            }
            
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summary/session/<session_id>', methods=['GET'])
def get_session_feedback_summary(session_id):
    """Get feedback summary for specific session"""
    try:
        with sqlite3.connect(feedback_service.db_path) as conn:
            cursor = conn.execute('''
                SELECT turn_id FROM feedback WHERE session_id = ?
            ''', (session_id,))
            turn_ids = [row[0] for row in cursor.fetchall()]
            
        summary = feedback_service.get_feedback_summary(turn_ids)
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "feedback_summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True)
