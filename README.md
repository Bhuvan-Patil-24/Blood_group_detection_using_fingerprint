# 🩸 Blood Group Detection Using Fingerprint

A deep learning and computer vision-based system for predicting human blood groups using fingerprint images.  
This project uses image preprocessing, feature extraction, and CNN/ResNet-based classification to accurately identify blood groups from fingerprint ridge patterns.

---

## 🚀 Key Features

- **Fingerprint Image Preprocessing**
  - Noise removal, normalization, ridge enhancement, orientation correction
- **Feature Extraction**
  - GLCM, LBP, Texture descriptors (optional hybrid model support)
- **Deep Learning Classification**
  - Custom CNN + ResNet50 classifier trained on ~6000 samples
- **High Evaluation Performance**
  - Achieved **80.01% accuracy**, **99.10% Top-3 Accuracy**
  - Includes ROC-AUC, confusion matrix, error analysis, and misclassification visualization
- **Interactive Web Interface (Flask Based)**
  - Image upload, real-time prediction, confidence score display
- **Admin Dashboard**
  - View statistics, performance, prediction history, model training
- **SQLite Database Integration**

---

## 📁 Project Structure
```bash
Blood_group_detection_using_fingerprint/
│
├── main.py # Flask application entry file
├── pipeline.py
├── api_routes.py
├── .gitignore
├── requirements.txt
├── README.md
│
├── model/
│ ├── cnn_model.py # CNN architecture + training utilities
│ ├── resNet_model.py # ResNet-based training module
│ ├── train.py # Training script
│ └── evaluate.py # Complete evaluation & visualization
│
├── saved_models/
| ├── fingerprint_validator.joblib
| ├── class_names.npy
| └── bloodgroup_cnn.keras
|
├── preprocessing/
│ └── image_processor.py # Image cleaning & enhancement
│
├── feature_extraction/
| ├── fingerprint_classifier.py
│ └── feature_extractor.py # Texture feature extraction
│
├── static/ # Assets & saved images/graphs
├── templates/
│ ├── about.html # Main UI interface
│ ├── base.html
│ ├── contact.html
│ ├── login.html
│ ├── profile.html
│ ├── register.html
│ ├── results.html
│ ├── upload.html
│ └── admin.html # Admin dashboard
│
├── database/
│ ├── migrate_db.py
│ ├── models.py
│ └── db_manager.py # SQLite storage operations
│
└── dataset_prepared/ # Dataset (train/val/test folders)
| ├── train/
| ├── validation/
| └── test/
```

---

## 🧠 Model Training Workflow

Fingerprint Image ➜ Preprocessing ➜ Feature Extraction ➜ CNN  ➜ Prediction


### 🖥 Model Training (Example)

```bash
python model/train.py
```

### 🖥 Model Evaluation (Example)

```bash
python model/evaluate.py
```

---

## 🧪 Dataset Details

- Total images: ~6000
- 8 blood group classes: A+, A-, B+, B-, AB+, AB-, O+, O-
- Image resolution: 128×128 grayscale
- Data split Strategy: 70% train — 15% validation — 15% test

---

### 📉 Confusion Matrix Result
```bash
      A+   A-  AB+  AB-   B+   B-   O+   O-
A+   173    0   26    1    0    0   18   65
A-     0  342    9   88   13   11    5   37
AB+    0    1  316    0   14    0    1   23
AB-    0    0    1  370    1    1    0    8
B+     0    2    7    9  309    0    0    0
B-     0   10    0   29   11  319    0    2
O+     2   12   11   24    0    0  235  143
O-     0    0    5   10    1    0    0  341
```
---

## 🔧 System Requirements

### Recommended Hardware

| Component | Requirement                                |
| --------- | ------------------------------------------ |
| RAM       | 8GB (16GB recommended)                     |
| GPU       | NVIDIA RTX 3050 / CUDA support recommended |
| Storage   | 5–10GB free                                |

### Software Requirements

- Python 3.10  (Mandatory)
- TensorFlow 2.16.1
- CUDA / cuDNN (optional for GPU)

---

## 📦 Setup & Installation

```bash
# Clone project
git clone <repo-url>
cd Blood_group_detection_using_fingerprint

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

---

## 🖼 Usage

### Start Application

```bash
python main.py
```

---

## 🔮 Future Scope

- Integration with biometric systems
- Mobile application
- Real-time capture from fingerprint sensor hardware
- Hybrid CNN-Transformer architecture
- Deployment on cloud for public access

---

## 🏁 Authors & Credits

Bhuvan Patil, Aniket Mishra, Prantik Deodhagale, Dhanshri Supratkar, Vishakha Padole  
Final Year Project - SBJITMR 2025

--- 

## 📜 License

This project is for educational and research purposes only.  
Unauthorized commercial use is prohibited.

---

## ⭐ Support

If this project helped you, leave a star on the repository 🤝  
For queries, feature suggestions, or collaborations: contact personally