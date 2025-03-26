import os
import numpy as np
import tensorflow as tf
from model.model import BloodGroupModel
import time
from datetime import datetime, timedelta
import shutil
from sklearn.model_selection import train_test_split

# Fix for int overflow on Windows
tf.config.experimental.enable_tensor_float_32_execution(False)

# Set memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

def prepare_dataset(source_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Prepare the dataset with proper train/val/test split"""
    print("\nPreparing dataset with 70/20/10 split...")
    
    # Create directories for train/val/test splits
    train_dir = "dataset_split/train"
    val_dir = "dataset_split/validation"
    test_dir = "dataset_split/test"
    
    # Remove existing split directory if it exists
    if os.path.exists("dataset_split"):
        shutil.rmtree("dataset_split")
    
    # Create directories
    for blood_group in os.listdir(source_path):
        blood_group_path = os.path.join(source_path, blood_group)
        if os.path.isdir(blood_group_path):
            os.makedirs(os.path.join(train_dir, blood_group), exist_ok=True)
            os.makedirs(os.path.join(val_dir, blood_group), exist_ok=True)
            os.makedirs(os.path.join(test_dir, blood_group), exist_ok=True)
            
            # Get all images for this blood group
            images = [img for img in os.listdir(blood_group_path) 
                     if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Skip if no images found
            if not images:
                continue
                
            # Split images into train, val, test
            train_images, temp_images = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
            
            # Further split temp_images into val and test
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_images, test_images = train_test_split(temp_images, test_size=relative_test_ratio, random_state=42)
            
            # Copy images to respective directories
            for img in train_images:
                src = os.path.join(blood_group_path, img)
                dst = os.path.join(train_dir, blood_group, img)
                shutil.copy2(src, dst)
                
            for img in val_images:
                src = os.path.join(blood_group_path, img)
                dst = os.path.join(val_dir, blood_group, img)
                shutil.copy2(src, dst)
                
            for img in test_images:
                src = os.path.join(blood_group_path, img)
                dst = os.path.join(test_dir, blood_group, img)
                shutil.copy2(src, dst)
            
            print(f"  {blood_group}: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
    
    # Count total images in each split
    train_count = sum([len(files) for _, _, files in os.walk(train_dir)])
    val_count = sum([len(files) for _, _, files in os.walk(val_dir)])
    test_count = sum([len(files) for _, _, files in os.walk(test_dir)])
    total_count = train_count + val_count + test_count
    
    print(f"\nDataset split complete:")
    print(f"  Training:   {train_count} images ({train_count/total_count*100:.1f}%)")
    print(f"  Validation: {val_count} images ({val_count/total_count*100:.1f}%)")
    print(f"  Testing:    {test_count} images ({test_count/total_count*100:.1f}%)")
    print(f"  Total:      {total_count} images")
    
    return {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        'train_count': train_count,
        'val_count': val_count,
        'test_count': test_count,
        'total_count': total_count
    }

def train_model():
    """Train the blood group detection model"""
    try:
        # Force CPU training since there are issues with the TF backend
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
        
        # TensorFlow memory optimization
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Step 1: Initialize model
        print("\nStep 1: Initializing model...")
        model = BloodGroupModel()
        
        # Step 2: Prepare dataset with proper train/val/test split
        print("\nStep 2: Preparing dataset...")
        dataset_path = "dataset_blood_group"
        dataset_info = prepare_dataset(dataset_path)
        
        # Step 3: Calculate estimated training time
        # Very conservative estimate due to CPU training
        batch_size = 4  # Very small batch size to avoid memory issues
        epochs = 20  # Reduced epochs for initial training
        steps_per_epoch = dataset_info['train_count'] // batch_size + 1
        time_per_step = 0.3  # seconds - adjust based on your hardware (higher for CPU)
        estimated_time_seconds = steps_per_epoch * time_per_step * epochs
        hours_estimate = estimated_time_seconds / 3600
        
        print(f"\nTraining configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Estimated time: {hours_estimate:.1f} hours")
        print(f"  Training will complete around: {(datetime.now() + timedelta(hours=hours_estimate)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 4: Start training
        print("\nStep 4: Starting training...")
        history = model.train(dataset_info['train_dir'], 
                             epochs=epochs, 
                             batch_size=batch_size,
                             validation_data_path=dataset_info['val_dir'])
        
        # Step 5: Save the model
        print("\nStep 5: Saving model...")
        model.save_model('saved_models/blood_group_model.h5')
        
        # Step 6: Evaluate on test set
        print("\nStep 6: Evaluating model on test set...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_info['test_dir'],
            image_size=(model.img_height, model.img_width),
            batch_size=batch_size
        )
        
        # Preprocess test data
        test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
        
        # Evaluate
        test_loss, test_accuracy = model.model.evaluate(test_ds)
        print(f"\nTest results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        
        print("\nTraining completed successfully!")
        return history
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()