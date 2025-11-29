"""
CNN Model Architecture for Blood Group Classification
Merged version combining your working model with enhanced features
Predicts 8 blood groups: A+, A-, B+, B-, O+, O-, AB+, AB-
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path


class BloodGroupCNN:
    """
    Convolutional Neural Network for blood group classification from fingerprints
    Compatible with both grayscale (1 channel) and RGB (3 channels) images
    """
    
    def __init__(self, input_shape=None, num_classes=8):
        """
        Initialize CNN model
        
        Args:
            input_shape (tuple): Input image dimensions (height, width, channels)
                               If None, will be auto-detected based on input data
            num_classes (int): Number of blood group classes (default: 8)
        """
        # Support both old and new initialization
        if input_shape is None:
            # Default to grayscale for pipeline compatibility
            self.input_shape = (128, 128, 1)
        else:
            self.input_shape = input_shape
            
        self.img_height = self.input_shape[0]
        self.img_width = self.input_shape[1]
        self.num_classes = num_classes
        self.model = None
        
        # Standard blood group labels
        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
        self.class_names = self.blood_groups  # Alias for compatibility
        self.is_trained = False
        
    def build_model(self, num_classes=None):
        """
        Build CNN architecture - optimized and simplified for stability
        Compatible with both training methods
        
        Args:
            num_classes (int): Number of classes (optional, uses self.num_classes if not provided)
        """
        if num_classes is not None:
            self.num_classes = num_classes
            
        print("Building model architecture...")
        
        model = Sequential(name='BloodGroup_CNN')
        
        # Block 1: Initial feature extraction
        model.add(Conv2D(32, (3, 3), activation='relu', 
                        padding='same', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2: Deeper feature extraction
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3: High-level features
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        
        # Block 4: Complex pattern recognition
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Output layer with softmax
        model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        print(f"✓ Model built with {self.model.count_params():,} parameters")
        
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate (float): Initial learning rate
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # Works with both sparse and categorical
            metrics=['accuracy']
        )
        
        print("✓ Model compiled successfully")
        print(f"  Total parameters: {self.model.count_params():,}")
    
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None,
              dataset_path=None, validation_data_path=None,
              epochs=50, batch_size=32, output_model='saved_models/bloodgroup_cnn.h5'):
        """
        Train the CNN model - supports both methods:
        1. Direct numpy arrays (X_train, y_train, X_val, y_val)
        2. Directory-based training (dataset_path)
        
        Args:
            X_train (np.ndarray): Training images (optional)
            y_train (np.ndarray): Training labels (optional)
            X_val (np.ndarray): Validation images (optional)
            y_val (np.ndarray): Validation labels (optional)
            dataset_path (str): Path to dataset directory (optional)
            validation_data_path (str): Path to validation directory (optional)
            epochs (int): Maximum training epochs
            batch_size (int): Batch size
            output_model (str): Path to save best model
            
        Returns:
            keras.callbacks.History: Training history
        """
        print("\n" + "=" * 70)
        print("CNN TRAINING - BLOOD GROUP CLASSIFICATION")
        print("=" * 70)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else '.', exist_ok=True)
        
        # Method 1: Direct numpy array training (for pipeline integration)
        if X_train is not None and y_train is not None:
            return self._train_from_arrays(X_train, y_train, X_val, y_val, 
                                          epochs, batch_size, output_model)
        
        # Method 2: Directory-based training (for your existing workflow)
        elif dataset_path is not None:
            return self._train_from_directory(dataset_path, validation_data_path,
                                             epochs, batch_size, output_model)
        
        else:
            raise ValueError("Provide either (X_train, y_train) or dataset_path")
    
    def _train_from_arrays(self, X_train, y_train, X_val, y_val,
                          epochs, batch_size, output_model):
        """Train from numpy arrays (new pipeline method)"""
        if self.model is None:
            self.compile_model()
        
        print(f"\nTraining Configuration:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val) if X_val is not None else 'N/A'}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
        print(f"  Blood groups: {', '.join(self.blood_groups)}")
        
        # Handle different label formats
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # One-hot encoded - convert to sparse
            y_train = np.argmax(y_train, axis=1)
            if y_val is not None:
                y_val = np.argmax(y_val, axis=1)
            # Change loss function
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                output_model,
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print("\nStarting training...\n")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Print final results
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        if X_val is not None:
            print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Model saved: {output_model}")
        
        return history
    
    def _train_from_directory(self, dataset_path, validation_data_path,
                             epochs, batch_size, output_model):
        """Train from directory structure (your original method)"""
        print("Setting up data generators...")
        
        try:
            # Convert dataset to tf.data.Dataset
            if validation_data_path:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_path,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    color_mode="grayscale"
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    validation_data_path,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    color_mode="grayscale"
                )
            else:
                # Use split from train data
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_path,
                    validation_split=0.2,
                    subset="validation",
                    seed=123,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    color_mode="grayscale"
                )
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_path,
                    validation_split=0.2,
                    subset="training",
                    seed=123,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size,
                    color_mode="grayscale"
                )
            
            # Get class names
            self.class_names = train_ds.class_names
            self.blood_groups = self.class_names
            print(f"Found classes: {self.class_names}")
            
            # Configure datasets for performance
            AUTOTUNE = tf.data.AUTOTUNE
            
            def preprocess(image, label):
                image = tf.cast(image, tf.float32) / 255.0
                return image, label
            
            train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
            val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
            
            # Build model if needed
            if self.model is None:
                # Auto-detect channels from dataset
                for images, labels in train_ds.take(1):
                    detected_shape = images.shape[1:]
                    if detected_shape != self.input_shape:
                        print(f"Auto-adjusting input shape from {self.input_shape} to {detected_shape}")
                        self.input_shape = detected_shape
                
                self.build_model(len(self.class_names))
                self.compile_model()
            
            print(f"\nTraining configuration:")
            print(f"- Batch size: {batch_size}")
            print(f"- Epochs: {epochs}")
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    output_model,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            print("\nStarting model training...")
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
            print(f"Model saved: {output_model}")
            
            return history
            
        except Exception as e:
            print(f"Error in training setup: {str(e)}")
            raise
    
    def predict(self, img):
        """
        Predict blood group from image
        Compatible with both file paths and numpy arrays
        
        Args:
            img: Either file path (str) or numpy array
            
        Returns:
            tuple: (predicted_class_index, class_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle file path input
        if isinstance(img, str):
            try:
                img = tf.keras.utils.load_img(
                    img, 
                    target_size=(self.img_height, self.img_width)
                )
                img = tf.keras.utils.img_to_array(img)
            except Exception as e:
                raise ValueError(f"Error loading image from path: {str(e)}")
        
        # Prepare image dimensions
        if img.ndim == 2:
            # Grayscale (H, W) -> (H, W, 1)
            img = np.expand_dims(img, axis=-1)
        
        if img.ndim == 3 and img.shape[-1] == 1:
            # Single channel, might need conversion for RGB models
            if self.input_shape[-1] == 3:
                img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 3:
            # RGB, might need conversion for grayscale models
            if self.input_shape[-1] == 1:
                img = np.mean(img, axis=-1, keepdims=True)
        
        if img.ndim == 3:
            # Add batch dimension (1, H, W, C)
            img = np.expand_dims(img, axis=0)
        
        # Normalize if needed
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        if img.max() > 1.0:
            img = img / 255.0
        
        # Make prediction
        probabilities = self.model.predict(img, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def predict_blood_group(self, image):
        """
        Predict blood group with label and confidence
        
        Args:
            image: Image (numpy array or file path)
            
        Returns:
            dict: {'blood_group': str, 'confidence': float, 'all_probabilities': dict}
        """
        predicted_class, probabilities = self.predict(image)
        
        # Use available class names
        class_label = (self.blood_groups[predicted_class] 
                      if predicted_class < len(self.blood_groups) 
                      else f"Class {predicted_class}")
        
        result = {
            'blood_group': class_label,
            'confidence': float(probabilities[predicted_class]),
            'all_probabilities': {
                (self.blood_groups[i] if i < len(self.blood_groups) else f"Class {i}"): 
                float(probabilities[i]) 
                for i in range(len(probabilities))
            }
        }
        
        return result
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        # Save class names in same directory
        class_names_path = os.path.join(os.path.dirname(path), 'class_names.npy')
        np.save(class_names_path, self.blood_groups)
        
        print(f"✓ Model saved: {path}")
        print(f"✓ Class names saved: {class_names_path}")
    
    def load_model(self, path):
        """Load a trained model"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = tf.keras.models.load_model(path)
        self.is_trained = True
        
        # Try to load class names
        class_names_path = os.path.join(os.path.dirname(path), 'class_names.npy')
        if os.path.exists(class_names_path):
            self.blood_groups = np.load(class_names_path, allow_pickle=True).tolist()
            self.class_names = self.blood_groups
            print(f"✓ CNN model loaded from: {path}")
            print(f"  Classes: {', '.join(self.blood_groups)}")
        else:
            print(f"✓ CNN model loaded from: {path}")
            print(f"  Warning: class_names.npy not found, using default labels")
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        
        self.model.summary()