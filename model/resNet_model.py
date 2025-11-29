"""
ResNet50 Transfer Learning Model for Blood Group Classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class BloodGroupResNet50:
    """
    Transfer Learning model using ResNet50 for blood group classification
    Supports grayscale & RGB input automatically
    """

    def __init__(self, input_shape=None, num_classes=8):
        if input_shape is None:
            self.input_shape = (224, 224, 3)   # Required input size for ResNet50
        else:
            self.input_shape = input_shape

        self.img_height = self.input_shape[0]
        self.img_width = self.input_shape[1]
        self.num_classes = num_classes
        self.model = None

        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
        self.class_names = self.blood_groups
        self.is_trained = False

    # -------------------------------------------------------------------
    def build_model(self, num_classes=None):
        if num_classes is not None:
            self.num_classes = num_classes

        print("\nBuilding ResNet50 transfer learning model...")

        # Base model from ImageNet
        base_model = ResNet50(
            weights="D:/FYProject_25-26/Blood_group_detection_using_fingerprint/model/weights/resnet50.h5",
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze feature extraction layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=output)

        print(f"✓ ResNet50 Model built with {self.model.count_params():,} parameters")
        return self.model

    # -------------------------------------------------------------------
    def compile_model(self, learning_rate=1e-4):
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✓ Model compiled successfully\n")

    # -------------------------------------------------------------------
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None,
              dataset_path=None, validation_data_path=None,
              epochs=30, batch_size=16, output_model='saved_models/resnet50_bloodgroup.h5'):

        print("\n" + "=" * 70)
        print("TRAINING - TRANSFER LEARNING (RESNET50)")
        print("=" * 70)

        os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else '.', exist_ok=True)

        if X_train is not None:
            return self._train_from_arrays(X_train, y_train, X_val, y_val, epochs, batch_size, output_model)

        elif dataset_path is not None:
            return self._train_from_directory(dataset_path, validation_data_path, epochs, batch_size, output_model)

        else:
            raise ValueError("Provide either numpy arrays or dataset directory!")

    # -------------------------------------------------------------------
    def _train_from_arrays(self, X_train, y_train, X_val, y_val,
                           epochs, batch_size, output_model):

        if self.model is None:
            self.compile_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
            ModelCheckpoint(output_model, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1
        )

        self.is_trained = True
        return history

    # -------------------------------------------------------------------
    def _train_from_directory(self, dataset_path, validation_data_path,
                              epochs, batch_size, output_model):

        print("Setting up dataset loaders (ResNet requires RGB images)...")

        if validation_data_path:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path, image_size=(self.img_height, self.img_width),
                batch_size=batch_size, color_mode="rgb"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                validation_data_path, image_size=(self.img_height, self.img_width),
                batch_size=batch_size, color_mode="rgb"
            )
        else:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                validation_split=0.2, subset="training", seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=batch_size, color_mode="rgb"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                validation_split=0.2, subset="validation", seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=batch_size, color_mode="rgb"
            )
        
        self.class_names = train_ds.class_names
        print("Classes:", self.class_names)

        AUTOTUNE = tf.data.AUTOTUNE

        def preprocess(img, label):
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        train_ds = train_ds.map(preprocess).prefetch(AUTOTUNE)
        val_ds = val_ds.map(preprocess).prefetch(AUTOTUNE)


        if self.model is None:
            self.build_model(len(self.class_names))
            self.compile_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
            ModelCheckpoint(output_model, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        print("\nStarting training...\n")
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

        self.is_trained = True
        return history

    # -------------------------------------------------------------------
    def predict(self, img):
        if self.model is None:
            raise ValueError("Load or train model first")

        if isinstance(img, str):
            img = tf.keras.utils.load_img(img, target_size=(self.img_height, self.img_width))
            img = tf.keras.utils.img_to_array(img)

        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)

        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        probabilities = self.model.predict(img, verbose=0)[0]
        pred_index = np.argmax(probabilities)

        return pred_index, probabilities

    # -------------------------------------------------------------------
    def predict_blood_group(self, image):
        pred, probabilities = self.predict(image)

        return {
            "blood_group": self.class_names[pred],
            "confidence": float(probabilities[pred]),
            "all_probabilities": {self.class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}
        }

    # -------------------------------------------------------------------
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        np.save(os.path.join(os.path.dirname(path), "class_names.npy"), self.class_names)
        print("Model saved:", path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.class_names = np.load(os.path.join(os.path.dirname(path), "class_names.npy"), allow_pickle=True).tolist()
        self.is_trained = True
        print("Loaded model:", path)

    def summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()
