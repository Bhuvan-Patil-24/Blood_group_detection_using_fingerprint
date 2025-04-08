import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_mail import Mail, Message
import numpy as np
from model.model import BloodGroupModel
from preprocessing.image_processor import ImageProcessor
from database.db_manager import DatabaseManager
from api_routes import api_bp
import tensorflow as tf
import secrets
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Generate a secure random key for the application
app.secret_key = secrets.token_hex(32)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///contact_messages.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Initialize extensions
mail = Mail(app)
model = BloodGroupModel()
image_processor = ImageProcessor()
db_manager = DatabaseManager()
db_manager.init_app(app)

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
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Server-side validation
        if not name or not email or not subject or not message:
            flash('All fields are required', 'error')
            return redirect(url_for('contact'))
            
        try:
            # Save message to database
            db_manager.save_contact_message(name, email, subject, message)
            
            # Send notification to admin
            admin_msg = Message(
                subject=f'New Contact Form Submission: {subject}',
                sender=app.config['MAIL_USERNAME'],
                recipients=[app.config['MAIL_USERNAME']],
                reply_to=email
            )
            admin_msg.body = f'''
            New message from {name} ({email}):
            
            Subject: {subject}
            
            Message:
            {message}
            '''
            
            # Send confirmation to user
            user_msg = Message(
                subject=f'Thank you for contacting us - {subject}',
                sender=app.config['MAIL_USERNAME'],
                recipients=[email],
                reply_to=app.config['MAIL_USERNAME']
            )
            user_msg.body = f'''
            Dear {name},
            
            Thank you for contacting us. We have received your message and will get back to you soon.
            
            Your message:
            {message}
            
            Best regards,
            Print2Type Team
            '''
            
            mail.send(admin_msg)
            mail.send(user_msg)
            
            flash('Your message has been sent successfully!', 'success')
            return redirect(url_for('contact'))
            
        except Exception as e:
            flash('An error occurred while processing your message. Please try again later.', 'error')
            app.logger.error(f'Error sending email: {str(e)}')
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
