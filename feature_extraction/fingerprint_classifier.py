"""
Fingerprint Classifier - Training and Inference Script
Binary classification: Fingerprint vs Non-Fingerprint
"""

import numpy as np
import joblib, cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# Import feature extractor
from feature_extraction.feature_extractor import FingerprintFeatureExtractor
from preprocessing.image_processor import ImageProcessor


class FingerprintClassifier:
    """
    Binary classifier to validate fingerprint vs non-fingerprint images
    """
    
    def __init__(self, model_type='auto'):
        """
        Initialize classifier
        
        Args:
            model_type (str): 'auto', 'random_forest', 'svm', or 'logistic'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.best_model_name = None
        self.training_metrics = {}
    
    def train(self, X_train, y_train, test_size=0.2, random_state=42):
        """
        Train classifier on feature vectors
        
        Args:
            X_train (np.ndarray): Feature vectors, shape (n_samples, n_features)
            y_train (np.ndarray): Labels (0: non-fingerprint, 1: fingerprint)
            test_size (float): Validation split ratio
            random_state (int): Random seed
            
        Returns:
            dict: Training results with metrics for all models
        """
        print("=" * 70)
        print("FINGERPRINT CLASSIFIER TRAINING")
        print("=" * 70)
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train
        )
        
        print(f"\nDataset Information:")
        print(f"  Training samples: {len(X_tr)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Fingerprints: {np.sum(y_train == 1)}")
        print(f"  Non-fingerprints: {np.sum(y_train == 0)}")
        
        # Standardize features
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1
            )
        }
        
        # Train and evaluate all models
        results = {}
        best_score = 0
        best_model = None
        best_model_name = None
        
        print("\n" + "-" * 70)
        print("MODEL COMPARISON")
        print("-" * 70)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_tr_scaled, y_tr)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            cm = confusion_matrix(y_val, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_tr_scaled, y_tr, cv=5, scoring='accuracy'
            )
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Print
            print(f"  Accuracy:    {accuracy:.4f}")
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
            print(f"  F1-Score:    {f1:.4f}")
            print(f"  CV Score:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Track best
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        # Select best model
        if self.model_type == 'auto':
            self.model = best_model
            self.best_model_name = best_model_name
        elif self.model_type == 'random_forest':
            self.model = models['Random Forest']
            self.best_model_name = 'Random Forest'
        elif self.model_type == 'svm':
            self.model = models['SVM']
            self.best_model_name = 'SVM'
        elif self.model_type == 'logistic':
            self.model = models['Logistic Regression']
            self.best_model_name = 'Logistic Regression'
        
        self.is_trained = True
        self.training_metrics = results
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {self.best_model_name}")
        print("=" * 70)
        best = results[self.best_model_name]
        print(f"Accuracy:    {best['accuracy']:.4f}")
        print(f"Precision:   {best['precision']:.4f}")
        print(f"Recall:      {best['recall']:.4f}")
        print(f"F1-Score:    {best['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(best['confusion_matrix'])
        print("\n[TN  FP]")
        print("[FN  TP]")
        
        return results
    
    def predict(self, features):
        """
        Predict class (0 or 1)
        
        Args:
            features (np.ndarray): Feature vector(s)
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def predict_proba(self, features):
        """
        Predict probability of fingerprint class
        
        Args:
            features (np.ndarray): Feature vector(s)
            
        Returns:
            np.ndarray: Probabilities [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_model_name = model_data.get('model_name', 'Unknown')
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        
        print(f"✓ Model loaded: {self.best_model_name}")


    def train_from_images(self, fingerprint_dir, non_fingerprint_dir, output_model='fingerprint_validator.joblib'):
        """
        Complete training pipeline: load images → extract features → train → save
        
        Args:
            fingerprint_dir (str): Directory with fingerprint images
            non_fingerprint_dir (str): Directory with non-fingerprint images
            output_model (str): Path to save trained model
            
        Returns:
            FingerprintClassifier: Trained classifier
        """
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE: IMAGES → FEATURES → MODEL")
        print("=" * 70)
        
        # Initialize
        preprocessor = ImageProcessor(target_size=(128, 128))
        extractor = FingerprintFeatureExtractor(target_size=(128, 128))
        
        features_list = []
        labels_list = []
        
        # Process fingerprints (label=1)
        print("\n[1/4] Loading fingerprint images...")
        fp_paths = list(Path(fingerprint_dir).glob('*.jpg')) + \
                list(Path(fingerprint_dir).glob('*.png')) + \
                list(Path(fingerprint_dir).glob('*.bmp'))
        
        print(f"Found {len(fp_paths)} fingerprint images")
        
        for i, path in enumerate(fp_paths, 1):
            try:
                img = preprocessor.preprocess_fingerprint(str(path))
                feat = extractor.extract_features(img)
                
                # Skip if features contain NaN or INF
                if np.isnan(feat).any() or np.isinf(feat).any():
                    print(f"  Skipped {path.name}: contains NaN/Inf values")
                    continue                
                
                features_list.append(feat)
                labels_list.append(1)
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(fp_paths)}")
            except Exception as e:
                print(f"  Skipped {path.name}: {e}")
        
        fp_count = len(labels_list)
        print(f"✓ Processed {fp_count} fingerprints")
        
        # Process non-fingerprints (label=0)
        print("\n[2/4] Loading non-fingerprint images...")
        nfp_paths = list(Path(non_fingerprint_dir).glob('*.jpg')) + \
                    list(Path(non_fingerprint_dir).glob('*.png')) + \
                    list(Path(non_fingerprint_dir).glob('*.bmp'))
        
        print(f"Found {len(nfp_paths)} non-fingerprint images")
        
        for i, path in enumerate(nfp_paths, 1):
            try:
                img = preprocessor.preprocess_fingerprint(str(path))
                feat = extractor.extract_features(img)
                
            # Skip if features contain NaN or INF
                if np.isnan(feat).any() or np.isinf(feat).any():
                    print(f"  Skipped {path.name}: contains NaN/Inf values")
                    continue

                features_list.append(feat)
                labels_list.append(0)
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(nfp_paths)}")
            except Exception as e:
                print(f"  Skipped {path.name}: {e}")
        
        nfp_count = len(labels_list) - fp_count
        print(f"✓ Processed {nfp_count} non-fingerprints")
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n[3/4] Dataset ready:")
        print(f"  Total samples: {len(X)}")
        print(f"  Fingerprints: {fp_count}")
        print(f"  Non-fingerprints: {nfp_count}")
        print(f"  Balance: {fp_count/len(X)*100:.1f}% / {nfp_count/len(X)*100:.1f}%")
        
        # Train classifier
        print("\n[4/4] Training classifier...")
        classifier = FingerprintClassifier(model_type='auto')
        classifier.train(X, y, test_size=0.2, random_state=42)
        
        # Save model
        classifier.save_model(output_model)
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Model saved to: {output_model}")
        
        return classifier
