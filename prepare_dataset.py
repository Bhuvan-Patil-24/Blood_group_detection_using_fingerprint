import os
import shutil
from sklearn.model_selection import train_test_split
import zipfile
from datetime import datetime
import argparse

def prepare_dataset(source_path, output_path="dataset_prepared", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Prepare the dataset with proper train/val/test split and create a zip file for training elsewhere"""
    print("\nPreparing dataset with 70/20/10 split......")
    
    # Create directories for train/val/test splits
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "validation")
    test_dir = os.path.join(output_path, "test")
    
    # Remove existing output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Create directories
    os.makedirs(output_path, exist_ok=True)
    
    # Keep track of class distribution
    class_counts = {}
    
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
            
            class_counts[blood_group] = {
                'train': len(train_images),
                'val': len(val_images),
                'test': len(test_images),
                'total': len(images)
            }
            
            print(f"  {blood_group}: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
    
    # Count total images in each split
    train_count = sum([counts['train'] for counts in class_counts.values()])
    val_count = sum([counts['val'] for counts in class_counts.values()])
    test_count = sum([counts['test'] for counts in class_counts.values()])
    total_count = train_count + val_count + test_count
    
    print(f"\nDataset split complete:")
    print(f"  Training:   {train_count} images ({train_count/total_count*100:.1f}%)")
    print(f"  Validation: {val_count} images ({val_count/total_count*100:.1f}%)")
    print(f"  Testing:    {test_count} images ({test_count/total_count*100:.1f}%)")
    print(f"  Total:      {total_count} images")
    
    # Create a zip file of the prepared dataset
    print("\nCreating zip file of the prepared dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"blood_group_dataset_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_path))
                zipf.write(file_path, arcname)
    
    print(f"\nDataset preparation complete!")
    print(f"Zip file created: {zip_filename}")
    print("\nInstructions for training on Google Colab:")
    print("1. Upload the zip file to your Google Drive")
    print("2. Create a new Colab notebook")
    print("3. Mount your Google Drive and unzip the dataset")
    print("4. Use the following code to train your model:")
    print("""
    # Sample code for training in Colab
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    # Set up data generators
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset_prepared/train',
        image_size=(128, 128),
        batch_size=32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset_prepared/validation',
        image_size=(128, 128),
        batch_size=32
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset_prepared/test',
        image_size=(128, 128),
        batch_size=32
    )
    
    # Preprocess the data
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    train_ds = train_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Build the model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(train_ds.class_names), activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('blood_group_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    )
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save the model
    model.save('blood_group_model.h5')
    """)
    
    return {
        'zip_filename': zip_filename,
        'train_count': train_count,
        'val_count': val_count,
        'test_count': test_count,
        'total_count': total_count,
        'class_distribution': class_counts
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for training on Google Colab')
    parser.add_argument('--source', default='dataset_blood_group', help='Source directory containing blood group images')
    parser.add_argument('--output', default='dataset_prepared', help='Output directory for prepared dataset')
    parser.add_argument('--train', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validate ratio inputs - allow for tiny floating point imprecision
    if abs(args.train + args.val + args.test - 1.0) > 0.00001:
        print(f"Error: Train ({args.train}), validation ({args.val}), and test ({args.test}) ratios must sum to 1.0")
        parser.print_help()
        exit(1)
    
    prepare_dataset(args.source, args.output, args.train, args.val, args.test) 