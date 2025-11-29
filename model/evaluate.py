import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MODEL_PATH = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/saved_models/bloodgroup_cnn.keras"
TEST_DATA_PATH = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/test"
INPUT_SHAPE = (128, 128, 1)
BATCH_SIZE = 32

# Blood group classes
CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']


def load_test_data():
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="categorical"
    )
    return test_generator


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


def main():
    print("📌 Loading Model...")
    model = load_model(MODEL_PATH)

    print("📌 Loading Test Data...")
    test_generator = load_test_data()

    print("⚡ Evaluating Model...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Basic accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    print("\n📉 Confusion Matrix Saved as: confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(test_generator.classes, predictions, multi_class='ovr')
        print(f"🏆 ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed (need probabilistic targets per class).")


if __name__ == "__main__":
    main()
