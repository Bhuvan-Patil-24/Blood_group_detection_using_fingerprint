import sqlite3
import os
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                blood_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL,
                features TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, blood_group, confidence, image_path, features):
        """Save a new prediction to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            features_json = json.dumps(features) if features is not None else None
            
            cursor.execute('''
                INSERT INTO predictions (timestamp, blood_group, confidence, image_path, features)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, blood_group, float(confidence), image_path, features_json))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return False
    
    def get_predictions(self, limit=None):
        """Retrieve predictions from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM predictions ORDER BY timestamp DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                predictions.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'blood_group': row[2],
                    'confidence': row[3],
                    'image_path': row[4],
                    'features': json.loads(row[5]) if row[5] is not None else None
                })
            
            conn.close()
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            return []
    
    def get_recent_predictions(self, limit=10):
        """Get the most recent predictions"""
        return self.get_predictions(limit=limit)
    
    def get_prediction_by_id(self, prediction_id):
        """Retrieve a specific prediction by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
            row = cursor.fetchone()
            
            if row:
                prediction = {
                    'id': row[0],
                    'timestamp': row[1],
                    'blood_group': row[2],
                    'confidence': row[3],
                    'image_path': row[4],
                    'features': json.loads(row[5]) if row[5] is not None else None
                }
            else:
                prediction = None
            
            conn.close()
            return prediction
        except Exception as e:
            print(f"Error getting prediction by ID: {str(e)}")
            return None
