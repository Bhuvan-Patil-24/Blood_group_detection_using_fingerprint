import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime

def load_test_data(test_dir):
    """Load test data using tf.keras.utils.image_dataset_from_directory"""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(128, 128),
        batch_size=32,
        shuffle=False
    )
    
    # Normalize images
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    test_ds = test_ds.map(preprocess)
    return test_ds

def main():
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model('saved_models/blood_group_model.h5')
    
    # Load class names
    class_names = np.load('saved_models/class_names.npy', allow_pickle=True).tolist()
    print(f"Class names: {class_names}")
    
    # Load test data
    print("Loading test data...")
    test_ds = load_test_data('dataset_split/test')
    
    # Get predictions
    print("Getting predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'class_names': class_names,
        'overall_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to evaluation_results/evaluation_results.json")

if __name__ == "__main__":
    main() 