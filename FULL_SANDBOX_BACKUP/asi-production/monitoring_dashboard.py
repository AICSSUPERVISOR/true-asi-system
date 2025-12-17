#!/usr/bin/env python3.11
"""
ASI MONITORING DASHBOARD & MANAGEMENT INTERFACE
Real-time System Monitoring and Control

Features:
✅ Real-time system statistics
✅ Agent performance monitoring
✅ Task execution tracking
✅ API usage analytics
✅ S3 data persistence
✅ Web-based dashboard (Flask)
✅ RESTful API for management
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any
from flask import Flask, jsonify, render_template_string, request
import boto3

# ============================================================================
# CONFIGURATION
# ============================================================================

S3_BUCKET = "asi-knowledge-base-898982995956"
S3_REGION = "us-east-1"
DB_PATH = "/home/ubuntu/asi-production/asi_production.db"

# ============================================================================
# MONITORING SYSTEM
# ============================================================================

class ASIMonitor:
    """Monitor ASI system performance"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=S3_REGION)
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        if not os.path.exists(DB_PATH):
            return {"error": "Database not found"}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Agent statistics
        cursor.execute("SELECT COUNT(*) FROM agents")
        total_agents = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM agents WHERE status = 'busy'")
        active_agents = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(tasks_completed), SUM(tasks_failed) FROM agents")
        completed, failed = cursor.fetchone()
        
        # Task statistics
        cursor.execute("SELECT COUNT(*) FROM tasks")
        total_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        task_status = dict(cursor.fetchall())
        
        # Performance metrics
        cursor.execute("SELECT AVG(total_processing_time) FROM agents WHERE tasks_completed > 0")
        avg_time = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "idle": total_agents - active_agents,
                "total_completed": completed or 0,
                "total_failed": failed or 0
            },
            "tasks": {
                "total": total_tasks,
                "by_status": task_status
            },
            "performance": {
                "avg_processing_time": round(avg_time, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_top_agents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing agents"""
        if not os.path.exists(DB_PATH):
            return []
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, specialty, tasks_completed, tasks_failed, total_processing_time
            FROM agents
            ORDER BY tasks_completed DESC
            LIMIT ?
        ''', (limit,))
        
        columns = ['id', 'specialty', 'tasks_completed', 'tasks_failed', 'total_processing_time']
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tasks"""
        if not os.path.exists(DB_PATH):
            return []
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, type, specialty_required, status, created_at, completed_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        columns = ['id', 'type', 'specialty', 'status', 'created_at', 'completed_at']
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_specialty_stats(self) -> Dict[str, Any]:
        """Get statistics by specialty"""
        if not os.path.exists(DB_PATH):
            return {}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT specialty, COUNT(*) as agent_count, 
                   SUM(tasks_completed) as total_completed,
                   SUM(tasks_failed) as total_failed
            FROM agents
            GROUP BY specialty
            ORDER BY total_completed DESC
        ''')
        
        columns = ['specialty', 'agent_count', 'completed', 'failed']
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_s3_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics"""
        try:
            # List recent uploads
            response = self.s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix='COMPLETE_ASI/',
                MaxKeys=10
            )
            
            objects = []
            total_size = 0
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
                    total_size += obj['Size']
            
            return {
                "recent_uploads": len(objects),
                "total_size_mb": round(total_size / (1024**2), 2),
                "objects": objects
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# FLASK WEB DASHBOARD
# ============================================================================

app = Flask(__name__)
monitor = ASIMonitor()

# HTML Dashboard Template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>ASI Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
               background: #0a0e27; color: #fff; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { font-size: 2.5em; margin-bottom: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #8b92b8; margin-bottom: 30px; font-size: 1.1em; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                      gap: 20px; margin-bottom: 30px; }
        .stat-card { background: linear-gradient(135deg, #1e2139 0%, #252a47 100%); 
                     padding: 25px; border-radius: 15px; border: 1px solid #2d3350; }
        .stat-label { color: #8b92b8; font-size: 0.9em; margin-bottom: 8px; }
        .stat-value { font-size: 2.5em; font-weight: bold; color: #fff; }
        .stat-subvalue { color: #667eea; font-size: 0.9em; margin-top: 5px; }
        .section { background: #1e2139; padding: 25px; border-radius: 15px; margin-bottom: 20px; 
                   border: 1px solid #2d3350; }
        .section-title { font-size: 1.5em; margin-bottom: 20px; color: #667eea; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 12px; color: #8b92b8; font-weight: 600; 
             border-bottom: 2px solid #2d3350; }
        td { padding: 12px; border-bottom: 1px solid #2d3350; }
        .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
        .status-completed { background: #10b981; color: #fff; }
        .status-pending { background: #f59e0b; color: #fff; }
        .status-failed { background: #ef4444; color: #fff; }
        .refresh-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 12px 30px; border: none; border-radius: 8px; 
                       cursor: pointer; font-size: 1em; font-weight: 600; }
        .refresh-btn:hover { opacity: 0.9; }
        .timestamp { color: #8b92b8; font-size: 0.9em; margin-top: 20px; }
    </style>
    <script>
        function refreshDashboard() {
            location.reload();
        }
        setInterval(refreshDashboard, 30000); // Auto-refresh every 30 seconds
    </script>
</head>
<body>
    <div class="container">
        <h1>🚀 ASI Monitoring Dashboard</h1>
        <p class="subtitle">Real-time monitoring of 100,000 agents across 50+ industries</p>
        
        <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
        
        <div style="margin-top: 30px;">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Agents</div>
                    <div class="stat-value">{{ stats.agents.total | format_number }}</div>
                    <div class="stat-subvalue">{{ stats.agents.active }} active</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Tasks Completed</div>
                    <div class="stat-value">{{ stats.agents.total_completed }}</div>
                    <div class="stat-subvalue">{{ stats.agents.total_failed }} failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Tasks</div>
                    <div class="stat-value">{{ stats.tasks.total }}</div>
                    <div class="stat-subvalue">All time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Processing Time</div>
                    <div class="stat-value">{{ stats.performance.avg_processing_time }}s</div>
                    <div class="stat-subvalue">Per task</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🏆 Top Performing Agents</div>
                <table>
                    <thead>
                        <tr>
                            <th>Agent ID</th>
                            <th>Specialty</th>
                            <th>Completed</th>
                            <th>Failed</th>
                            <th>Avg Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for agent in top_agents[:10] %}
                        <tr>
                            <td>#{{ agent.id }}</td>
                            <td>{{ agent.specialty }}</td>
                            <td>{{ agent.tasks_completed }}</td>
                            <td>{{ agent.tasks_failed }}</td>
                            <td>{{ "%.2f"|format(agent.total_processing_time) }}s</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">📋 Recent Tasks</div>
                <table>
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Specialty</th>
                            <th>Status</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for task in recent_tasks[:15] %}
                        <tr>
                            <td>{{ task.id }}</td>
                            <td>{{ task.specialty }}</td>
                            <td><span class="status-badge status-{{ task.status }}">{{ task.status }}</span></td>
                            <td>{{ task.created_at[:19] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="timestamp">Last updated: {{ stats.timestamp }}</div>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Main dashboard"""
    stats = monitor.get_system_stats()
    top_agents = monitor.get_top_agents(10)
    recent_tasks = monitor.get_recent_tasks(15)
    
    return render_template_string(
        DASHBOARD_HTML,
        stats=stats,
        top_agents=top_agents,
        recent_tasks=recent_tasks
    )

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify(monitor.get_system_stats())

@app.route('/api/agents/top')
def api_top_agents():
    """API endpoint for top agents"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(monitor.get_top_agents(limit))

@app.route('/api/tasks/recent')
def api_recent_tasks():
    """API endpoint for recent tasks"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(monitor.get_recent_tasks(limit))

@app.route('/api/specialties')
def api_specialties():
    """API endpoint for specialty statistics"""
    return jsonify(monitor.get_specialty_stats())

@app.route('/api/s3')
def api_s3():
    """API endpoint for S3 statistics"""
    return jsonify(monitor.get_s3_stats())

@app.template_filter('format_number')
def format_number(value):
    """Format number with commas"""
    return f"{value:,}"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ASI MONITORING DASHBOARD")
    print("="*80)
    print("\n🚀 Starting monitoring dashboard...")
    print("\n📊 Dashboard will be available at:")
    print("   http://localhost:5000")
    print("\n📡 API Endpoints:")
    print("   GET /api/stats          - System statistics")
    print("   GET /api/agents/top     - Top performing agents")
    print("   GET /api/tasks/recent   - Recent tasks")
    print("   GET /api/specialties    - Specialty statistics")
    print("   GET /api/s3             - S3 storage statistics")
    print("\n✅ Dashboard ready! Press Ctrl+C to stop.")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
