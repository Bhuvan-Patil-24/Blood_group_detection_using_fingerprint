import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class BloodGroupModel:
    def __init__(self):
        self.model = None
        self.img_height = 128
        self.img_width = 128
        self.class_names = []
    
    def build_model(self, num_classes):
        """Build a simplified CNN model for blood group classification"""
        print("Building model architecture...")
        
        # Create a sequential model - simpler architecture to avoid TensorFlow bugs
        self.model = Sequential([
            # Input layer implicitly defined by the first layer's input_shape
            
            # First convolutional block - simplified
            Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(self.img_height, self.img_width, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Second convolutional block - simplified
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Third convolutional block - simplified
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten the output for dense layers
            Flatten(),
            
            # Dense layers - simplified
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, dataset_path, epochs=50, batch_size=32, validation_data_path=None):
        """Train the model using the provided dataset"""
        print("Setting up data generators...")
        try:
            # Convert dataset to tf.data.Dataset
            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                image_size=(self.img_height, self.img_width),
                batch_size=batch_size
            )
            
            # If validation path is provided, use it, otherwise use split from train data
            if validation_data_path:
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    validation_data_path,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size
                )
            else:
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_path,
                    validation_split=0.2,
                    subset="validation",
                    seed=123,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size
                )
                
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_path,
                    validation_split=0.2,
                    subset="training",
                    seed=123,
                    image_size=(self.img_height, self.img_width),
                    batch_size=batch_size
                )
            
            # Get class names
            self.class_names = train_ds.class_names
            print(f"Found classes: {self.class_names}")
            
            # Configure datasets for performance
            AUTOTUNE = tf.data.AUTOTUNE
            
            # Convert data types and normalize
            def preprocess(image, label):
                image = tf.cast(image, tf.float32) / 255.0
                return image, label
            
            train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
            val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
            
            # Enable prefetching
            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
            
            # Build model if not already built
            if self.model is None:
                print("Building model architecture...")
                self.build_model(len(self.class_names))
            
            # Create directories if they don't exist
            os.makedirs('saved_models', exist_ok=True)
            os.makedirs('checkpoints', exist_ok=True)
            
            print(f"Training configuration:")
            print(f"- Batch size: {batch_size}")
            print(f"- Epochs: {epochs}")
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    'saved_models/blood_group_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
            
            print("\nStarting model training...")
            # Train the model
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error in training setup: {str(e)}")
            raise
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            
            # Save class names
            np.save(os.path.join(os.path.dirname(path), 'class_names.npy'), self.class_names)
        else:
            raise ValueError("Model has not been built yet")
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path)
        
        # Load class names
        class_names_path = os.path.join(os.path.dirname(path), 'class_names.npy')
        if os.path.exists(class_names_path):
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
    
    def predict(self, img):
        """Predict blood group from image or file path"""
        if self.model is None:
            raise ValueError("Model has not been loaded or trained")
        
        # Check if input is a file path
        if isinstance(img, str):
            # Load and preprocess the image
            try:
                img = tf.keras.utils.load_img(
                    img, 
                    target_size=(self.img_height, self.img_width)
                )
                img = tf.keras.utils.img_to_array(img)
                img = np.expand_dims(img, axis=0)
            except Exception as e:
                raise ValueError(f"Error loading image from path: {str(e)}")
        else:
            # Ensure image has correct dimensions
            if len(img.shape) == 3:  # Single image
                img = np.expand_dims(img, axis=0)
        
        # Normalize image
        img = img / 255.0
        
        # Make prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        
        # Check if class names are available
        if not self.class_names or len(self.class_names) <= predicted_class:
            return f"Class {predicted_class}", confidence
        
        return self.class_names[predicted_class], confidence
