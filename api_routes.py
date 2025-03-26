from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from database.db_manager import DatabaseManager

api_bp = Blueprint('api', __name__)
db_manager = DatabaseManager()

@api_bp.route('/stats')
def get_stats():
    """Get prediction statistics"""
    try:
        predictions = db_manager.get_predictions()
        total_count = len(predictions)
        
        if total_count == 0:
            return jsonify({
                'total_predictions': 0,
                'average_confidence': 0,
                'blood_group_distribution': {}
            })
        
        # Calculate average confidence
        confidences = [float(pred.get('confidence', 0)) for pred in predictions]
        avg_confidence = sum(confidences) / total_count
        
        # Calculate blood group distribution
        blood_groups = {}
        for pred in predictions:
            group = pred.get('blood_group', 'Unknown')
            blood_groups[group] = blood_groups.get(group, 0) + 1
        
        return jsonify({
            'total_predictions': total_count,
            'average_confidence': round(avg_confidence, 4),
            'blood_group_distribution': blood_groups
        })
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify({
            'error': 'Failed to get statistics',
            'details': str(e)
        }), 500

@api_bp.route('/predictions')
def get_predictions():
    """Get recent predictions for table display"""
    limit = request.args.get('limit', 50, type=int)
    predictions = db_manager.get_predictions(limit=limit)
    return jsonify(predictions)

@api_bp.route('/model-info')
def get_model_info():
    """Get information about the current model"""
    # This would typically be stored in the database or a separate file
    # For this example, we'll return mock data
    return jsonify({
        'model_name': 'blood_group_model.h5',
        'last_trained': datetime.now().isoformat(),
        'validation_accuracy': 0.92
    })