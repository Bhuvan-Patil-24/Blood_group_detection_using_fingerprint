import os
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_mail import Mail, Message
from pipeline import FingerprintBloodGroupPipeline
import secrets
import logging
from dotenv import load_dotenv
from database.models import db, Prediction, ContactMessage, User
from PIL import Image, ImageDraw
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Generate a secure random key for the application
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///contact_messages.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()
    # Create admin user if it doesn't exist
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        admin = User(
            username='admin',
            email=admin_email,
            is_admin=True
        )
        admin.set_password(admin_password)
        db.session.add(admin)
        db.session.commit()
        print(f"Admin user created with email: {admin_email}")

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Initialize extensions
mail = Mail(app)
pipeline  = FingerprintBloodGroupPipeline()

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Get the next page from query string (or default to index)
            next_page = request.args.get('next')
            if not next_page or next_page.startswith('//'):
                next_page = url_for('index')
                
            flash('Login successful!', 'success')
            return redirect(next_page)
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('profile.html', predictions=user_predictions)

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    users = User.query.all()
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    messages = ContactMessage.query.order_by(ContactMessage.timestamp.desc()).all()
    
    return render_template('admin.html', users=users, predictions=predictions, messages=messages)

# Main routes
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
            # Save message to database using SQLAlchemy
            new_message = ContactMessage(
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            db.session.add(new_message)
            db.session.commit()
            
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
            logger.error(f'Error in contact form: {str(e)}')
            return redirect(url_for('contact'))
            
    return render_template('contact.html')

@app.route('/upload')
def upload_form():
    if not current_user.is_authenticated:
        flash('You need to be logged in to upload and analyze fingerprint images.', 'info')
        return redirect(url_for('login', next=url_for('upload_form')))
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Generate a unique filename to avoid conflicts
        unique_filename = f"{secrets.token_hex(8)}_{file.filename}"
        
        # Save uploaded image in the static/uploads folder
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        temp_path = os.path.join(upload_folder, unique_filename)
        file.save(temp_path)
        
        logger.info(f"Saved uploaded image to: {temp_path}")
        
        # # Make prediction directly using the image
        # blood_group, confidence = model.predict(temp_path)

        result = pipeline.predict(temp_path, return_detailed=False)

        # Pipeline FAILED case
        if not result.get("status", False):
            logger.error(f"Pipeline prediction failed: {result.get('message')}")
            return jsonify({"error": result.get("message", "Unknown Error")}), 400

        # Extract values from pipeline result
        blood_group = result["blood_group"]
        confidence = float(result["confidence"])
        
        # Store path relative to static folder
        relative_path = 'uploads/' + unique_filename
        logger.info(f"Storing relative path in DB: {relative_path}")
        
        # Save result to database using SQLAlchemy
        new_prediction = Prediction(
            fingerprint_path=relative_path,
            blood_group=blood_group,
            confidence=float(confidence) * 100,  # Convert to percentage
            user_id=current_user.id  # Link prediction to current user
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        return jsonify({
            'blood_group': blood_group,
            'confidence': confidence
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Make sure to return JSON even for errors
        db.session.rollback()  # Roll back any failed database operations
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    # Get page number from query parameters, default to 1
    page = request.args.get('page', 1, type=int)
    # Number of items per page
    per_page = 5
    
    # Filter by user if not admin and logged in
    if current_user.is_authenticated:
        if current_user.is_admin:
            # Admin sees all predictions
            query = Prediction.query
        else:
            # Regular users see only their predictions
            query = Prediction.query.filter_by(user_id=current_user.id)
    else:
        # Not logged in - show nothing, redirect to login
        flash('Please log in to view prediction results.', 'info')
        return redirect(url_for('login', next=url_for('results')))
    
    # Get paginated predictions ordered by timestamp (newest first)
    pagination = query.order_by(Prediction.timestamp.desc()).paginate(
        page=page, 
        per_page=per_page,
        error_out=False
    )
    
    # Format the predictions for display
    formatted_predictions = []
    for pred in pagination.items:
        # Check if image file exists
        image_path = pred.fingerprint_path
        full_path = os.path.join('static', image_path)
        
        # If file doesn't exist, use a placeholder
        if not os.path.exists(full_path):
            image_path = 'img/placeholder-fingerprint.png'
            # Ensure the placeholder directory exists
            placeholder_dir = os.path.join('static', 'img')
            os.makedirs(placeholder_dir, exist_ok=True)
            
            # Create a placeholder image if it doesn't exist
            placeholder_path = os.path.join(placeholder_dir, 'placeholder-fingerprint.png')
            if not os.path.exists(placeholder_path):
                # Create a simple placeholder image
                img = Image.new('RGB', (150, 150), color=(240, 240, 240))
                d = ImageDraw.Draw(img)
                
                # Get text size for centering
                text = "No Image"
                text_width = 60  # Approximate text width
                text_height = 15  # Approximate text height
                text_x = (150 - text_width) // 2
                text_y = (150 - text_height) // 2
                
                # Draw text centered on the image
                d.text((text_x, text_y), text, fill=(100, 100, 100))
                
                # Save the image
                img.save(placeholder_path)
        
        formatted_predictions.append({
            'id': pred.id,
            'fingerprint_path': image_path,
            'blood_group': pred.blood_group,
            'confidence': f"{pred.confidence:.2f}%",
            'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return render_template('results.html', 
                         predictions=formatted_predictions,
                         pagination=pagination)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/clear_predictions', methods=['GET'])
@login_required
def clear_predictions():
    try:
        # Only allow admins to clear predictions
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required to clear predictions.', 'error')
            return redirect(url_for('results'))
            
        # Delete all prediction records
        Prediction.query.delete()
        
        # Clean up uploads folder
        upload_folder = os.path.join('static', 'uploads')
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        db.session.commit()
        flash('All predictions have been cleared successfully!', 'success')
        return redirect(url_for('results'))
    
    except Exception as e:
        flash('An error occurred while clearing predictions.', 'error')
        logger.error(f"Error clearing predictions: {str(e)}")
        return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)
