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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Generate a secure random key for the application
app.secret_key = secrets.token_hex(32)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contact_messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'patilbhuvan27@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'kfur lxrm turz qdlr'     # Replace with your app password

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
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        
        # Server-side validation
        errors = []
        
        # Name validation
        if not name:
            errors.append('Name is required')
        elif not re.match(r'^[A-Za-z\s]{2,50}$', name):
            errors.append('Name should be between 2-50 characters and contain only letters and spaces')
        
        # Email validation
        if not email:
            errors.append('Email is required')
        elif not re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email.lower()):
            errors.append('Please enter a valid email address')
        
        # Subject validation
        if not subject:
            errors.append('Subject is required')
        elif len(subject) < 5 or len(subject) > 100:
            errors.append('Subject should be between 5-100 characters')
        
        # Message validation
        if not message:
            errors.append('Message is required')
        elif len(message) < 10 or len(message) > 1000:
            errors.append('Message should be between 10-1000 characters')
        
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('contact'))
        
        # If validation passes, process the message
        try:
            # Save message to database
            db_manager.save_contact_message(name, email, subject, message)
            logger.info(f"Message saved to database from {email}")
            
            # Send email notification
            try:
                # Send notification to admin
                admin_msg = Message(
                    subject=f"New Contact Form Submission: {subject}",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[app.config['MAIL_USERNAME']],  # Send to admin email
                    reply_to=email
                )
                admin_msg.body = f"""
                New contact form submission from {name} ({email}):
                
                Subject: {subject}
                
                Message:
                {message}
                
                Please respond to this message at your earliest convenience.
                """
                
                mail.send(admin_msg)
                logger.info(f"Admin notification email sent for message from {email}")
                
                # Send confirmation email to the user who submitted the form
                user_msg = Message(
                    subject="Thank you for contacting Print2Type",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[email],  # Send to the user's email
                    reply_to=app.config['MAIL_USERNAME']
                )
                user_msg.body = f"""
                Dear {name},
                
                Thank you for contacting Print2Type. We have received your message and will get back to you soon.
                
                Your message:
                {message}
                
                Best regards,
                Print2Type Team
                """
                
                mail.send(user_msg)
                logger.info(f"Confirmation email sent to {email}")
                
                flash('Thank you for your message! We will get back to you soon.', 'success')
                return redirect(url_for('contact'))
                
            except Exception as e:
                logger.error(f"Email sending error: {str(e)}")
                # Even if email sending fails, we still saved the message
                flash('Your message has been received. We will get back to you soon.', 'success')
                return redirect(url_for('contact'))
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            flash('An error occurred while saving your message. Please try again later.', 'danger')
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
