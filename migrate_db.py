import os
from flask import Flask
from database.models import db, User
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a new Flask app instance for migration
app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///contact_messages.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Recreate database tables
with app.app_context():
    print("Dropping all tables...")
    db.drop_all()
    
    print("Creating all tables with updated schema...")
    db.create_all()
    
    # Create admin user with credentials from environment variables
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
    
    print("Database migration completed successfully!") 