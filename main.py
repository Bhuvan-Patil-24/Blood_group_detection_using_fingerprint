import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import numpy as np
from model.model import BloodGroupModel
from preprocessing.image_processor import ImageProcessor
from database.db_manager import DatabaseManager
from api_routes import api_bp
import tensorflow as tf
import secrets

app = Flask(__name__)
# Generate a secure random key for the application
app.secret_key = secrets.token_hex(32)  # Generates a 64-character hexadecimal string
model = BloodGroupModel()
image_processor = ImageProcessor()
db_manager = DatabaseManager()

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Load the trained model
MODEL_PATH = 'saved_models/blood_group_model.h5'
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Warning: Model not found. Please train the model first.")

# Try to load class names
CLASS_NAMES_PATH = 'saved_models/class_names.npy'
if os.path.exists(CLASS_NAMES_PATH):
    model.class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
    print(f"Loaded class names: {model.class_names}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Here you would typically:
        # 1. Validate the form data
        # 2. Send an email or store in database
        # 3. Show a success message
        
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Save uploaded image temporarily
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # Make prediction directly using the image
        blood_group, confidence = model.predict(temp_path)
        
        # Save result to database
        db_manager.save_prediction(
            blood_group=blood_group,
            confidence=confidence,
            image_path=temp_path,
            features=None  # No features used in this model
        )
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'blood_group': blood_group,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/results')
def results():
    # Get last 10 predictions from database
    predictions = db_manager.get_recent_predictions(10)
    return render_template('results.html', predictions=predictions)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
