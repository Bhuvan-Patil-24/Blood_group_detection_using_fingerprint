# Blood Group Detection Using Fingerprint

A machine learning-based system that predicts blood groups from fingerprint images using deep learning and image processing techniques.

## Features

- **Fingerprint Analysis**: Upload and analyze fingerprint images
- **Blood Group Prediction**: Accurate prediction of blood groups with confidence scores
- **Admin Dashboard**: Monitor system performance and manage models
- **Database Integration**: Track and store prediction history
- **Interactive UI**: User-friendly interface for both users and administrators

## Prerequisites

- Python 3.10 or later
- Windows/Linux/MacOS
- Git (optional)
- 4GB RAM minimum (8GB recommended)
- Webcam (optional, for real-time capture)

## Installation

1. **Clone the Repository** (if using Git):
   ```bash
   git clone <repository-url>
   cd Blood_group_detection_using_fingerprint
   ```

2. **Set Up Python Virtual Environment**:

   For Windows:
   ```powershell
   # Open PowerShell as Administrator first and run:
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   .\venv\Scripts\activate
   ```

   For Linux/MacOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Blood_group_detection_using_fingerprint/
├── main.py                 # Main Flask application
├── api_routes.py          # API endpoints for admin dashboard
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html       # Main application interface
│   └── admin.html       # Admin dashboard interface
├── model/               # Model-related code
│   └── model.py        # BloodGroupModel implementation
├── preprocessing/       # Image processing
│   └── image_processor.py
├── feature_extraction/ # Feature extraction
│   └── feature_extractor.py
├── database/          # Database operations
│   └── db_manager.py
├── saved_models/      # Trained model storage
│   └── blood_group_model.h5
└── dataset_blood_group/  # Training dataset
    ├── A+/
    ├── A-/
    ├── B+/
    └── ...
```

## Configuration

1. **Database Setup**:
   - SQLite database will be automatically created on first run
   - Located at `blood_groups.db` in project root
   - No additional configuration needed

2. **Model Setup**:
   - Default model path: `saved_models/blood_group_model.h5`
   - Will be created after first training session
   - Can be replaced with pre-trained model

## Running the Application

1. **Start the Server**:
   ```bash
   python main.py
   ```

2. **Access the Application**:
   - Main Interface: `http://localhost:5000`
   - Admin Dashboard: `http://localhost:5000/admin`

## Usage Guide

### Main Interface
1. Upload fingerprint image using the upload button
2. Wait for processing and analysis
3. View predicted blood group and confidence score
4. Results are automatically saved to database

### Admin Dashboard
1. View system statistics and predictions
2. Monitor blood group distribution
3. Track prediction confidence levels
4. Train new models with custom parameters
5. Download trained models for backup

### Training New Models
1. Access the admin dashboard
2. Navigate to "Model Management"
3. Configure training parameters:
   - Dataset path
   - Number of epochs
   - Batch size
4. Click "Train Model" and monitor progress
5. New model will automatically be saved and used

## API Endpoints

- `/api/stats` - Get system statistics
- `/api/predictions` - Get recent predictions
- `/api/model-info` - Get current model information
- `/predict` - Make new prediction
- `/train` - Train new model

## Dependencies

```
# Web Framework
flask==2.3.3

# Machine Learning and Data Processing
tensorflow==2.16.1
numpy==1.24.3
pandas==2.0.3
scikit-image==0.21.0
scikit-learn==1.3.0

# Image Processing
opencv-python==4.8.0.76
Pillow==10.0.1

# Database
SQLAlchemy==2.0.23

# Utilities
matplotlib==3.7.3
tqdm==4.66.1
```

## Troubleshooting

1. **Virtual Environment Issues**:
   - Ensure Python 3.10 or later is installed
   - Run PowerShell as Administrator for Windows
   - Check execution policy settings

2. **Package Installation Errors**:
   - If TensorFlow fails: Try `pip install tensorflow-cpu`
   - If OpenCV fails: Try `pip install opencv-python-headless`
   - Update pip: `python -m pip install --upgrade pip`

3. **Runtime Errors**:
   - Check if model file exists
   - Verify dataset structure
   - Ensure database permissions
   - Check disk space for model training

4. **Performance Issues**:
   - Close unnecessary applications
   - Monitor system resources
   - Consider using GPU for training
   - Reduce batch size if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request