"""
Enhanced Evaluation Script for Blood Group Classification CNN Model
Research-Level ML Model Evaluation Pipeline

Run with:
    python evaluate_model.py

Requirements:
    pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    cohen_kappa_score, log_loss, accuracy_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MODEL_PATH = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/saved_models/bloodgroup_cnn.keras"
TEST_DATA_PATH = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/test"
RESULTS_DIR = "evaluation_results"
INPUT_SHAPE = (128, 128, 1)
BATCH_SIZE = 32

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# --------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------
def load_test_data():
    """Load and preprocess test dataset"""
    print("📁 Loading test data...")
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="categorical"
    )
    
    # Extract class names from generator (sorted by index)
    class_names = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    
    print(f"✅ Loaded {test_generator.samples} test samples")
    print(f"📊 Classes: {class_names}")
    
    return test_generator, class_names


# --------------------------------------------------------
# METRICS CALCULATION
# --------------------------------------------------------
def calculate_top_k_accuracy(y_true, predictions, k=3):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = sum([1 for i, true_label in enumerate(y_true) if true_label in top_k_preds[i]])
    return correct / len(y_true)


def compute_all_metrics(y_true, y_pred, predictions, class_names):
    """Compute comprehensive evaluation metrics"""
    print("\n📊 Computing evaluation metrics...")
    
    metrics = {}
    
    # Basic accuracies
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['top3_accuracy'] = calculate_top_k_accuracy(y_true, predictions, k=3)
    
    # Classification report as dictionary
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract macro/micro/weighted averages
    metrics['macro_precision'] = report['macro avg']['precision']
    metrics['macro_recall'] = report['macro avg']['recall']
    metrics['macro_f1'] = report['macro avg']['f1-score']
    metrics['weighted_precision'] = report['weighted avg']['precision']
    metrics['weighted_recall'] = report['weighted avg']['recall']
    metrics['weighted_f1'] = report['weighted avg']['f1-score']
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Log loss
    metrics['log_loss'] = log_loss(y_true, predictions)
    
    # ROC-AUC (One-vs-Rest)
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, predictions, multi_class='ovr', average='weighted')
        metrics['roc_auc_ovo'] = roc_auc_score(y_true, predictions, multi_class='ovo', average='weighted')
    except Exception as e:
        print(f"⚠️  ROC-AUC computation warning: {e}")
        metrics['roc_auc_ovr'] = None
        metrics['roc_auc_ovo'] = None
    
    # Per-class metrics DataFrame
    per_class_metrics = []
    for i, class_name in enumerate(class_names):
        per_class_metrics.append({
            'Class': class_name,
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score'],
            'Support': int(report[class_name]['support'])
        })
    
    metrics_df = pd.DataFrame(per_class_metrics)
    
    return metrics, metrics_df


# --------------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """Plot confusion matrix (raw or normalized)"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        fmt = '.2f'
        filename = "normalized_confusion_matrix.png"
    else:
        title = "Confusion Matrix"
        fmt = 'd'
        filename = "confusion_matrix.png"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {filename}")


def plot_roc_curves(y_true, predictions, class_names):
    """Plot ROC curves for all classes"""
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    # Plot ROC curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved roc_curve.png")


def plot_precision_recall_curves(y_true, predictions, class_names):
    """Plot Precision-Recall curves for all classes"""
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    # Plot PR curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predictions[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], predictions[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved pr_curve.png")


def plot_prediction_distribution(y_true, y_pred, class_names):
    """Plot distribution of predictions per class"""
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    true_counts = pd.Series(y_true).value_counts().sort_index()
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, true_counts, width, label='True Distribution', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, pred_counts, width, label='Predicted Distribution', alpha=0.8, color='coral')
    
    ax.set_xlabel('Blood Group', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Distribution vs True Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved prediction_distribution.png")


def plot_misclassified_samples(test_generator, y_true, y_pred, predictions, class_names, max_samples=20):
    """Display grid of misclassified samples with predictions"""
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("🎉 No misclassified samples found!")
        return
    
    # Limit to max_samples
    num_samples = min(max_samples, len(misclassified_idx))
    selected_idx = misclassified_idx[:num_samples]
    
    # Calculate grid dimensions
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Get all images
    test_generator.reset()
    all_images = []
    for _ in range(len(test_generator)):
        batch = next(test_generator)
        all_images.extend(batch[0])
        if len(all_images) >= test_generator.samples:
            break
    all_images = np.array(all_images[:test_generator.samples])
    
    for idx, ax in enumerate(axes):
        if idx < num_samples:
            img_idx = selected_idx[idx]
            img = all_images[img_idx].squeeze()
            
            true_label = class_names[y_true[img_idx]]
            pred_label = class_names[y_pred[img_idx]]
            confidence = predictions[img_idx][y_pred[img_idx]] * 100
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                        fontsize=9, color='red')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Misclassified Samples (Total: {len(misclassified_idx)})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "misclassified_examples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved misclassified_examples.png ({num_samples} samples)")


# --------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------
def generate_evaluation_report(metrics, metrics_df, y_true, y_pred, class_names):
    """Generate comprehensive text report"""
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BLOOD GROUP CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write("="*70 + "\n\n")
        
        # Overall Metrics
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Top-1 Accuracy:          {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)\n")
        f.write(f"Top-3 Accuracy:          {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)\n")
        f.write(f"Cohen's Kappa Score:     {metrics['cohen_kappa']:.4f}\n")
        f.write(f"Log Loss:                {metrics['log_loss']:.4f}\n")
        if metrics['roc_auc_ovr'] is not None:
            f.write(f"ROC-AUC (OvR):           {metrics['roc_auc_ovr']:.4f}\n")
            f.write(f"ROC-AUC (OvO):           {metrics['roc_auc_ovo']:.4f}\n")
        f.write("\n")
        
        # Averaged Metrics
        f.write("AGGREGATED METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Macro Precision:         {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:            {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:          {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted Precision:      {metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall:         {metrics['weighted_recall']:.4f}\n")
        f.write(f"Weighted F1-Score:       {metrics['weighted_f1']:.4f}\n")
        f.write("\n")
        
        # Per-Class Metrics
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        f.write(cm_df.to_string())
        f.write("\n\n")
        
        # Misclassification Analysis
        misclassified_idx = np.where(y_true != y_pred)[0]
        f.write("MISCLASSIFICATION ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Misclassified:     {len(misclassified_idx)} / {len(y_true)}\n")
        f.write(f"Error Rate:              {len(misclassified_idx)/len(y_true)*100:.2f}%\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("End of Report\n")
        f.write("="*70 + "\n")
    
    print(f"✅ Saved evaluation_report.txt")
    
    # Also save metrics DataFrame as CSV
    csv_path = os.path.join(RESULTS_DIR, "per_class_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"✅ Saved per_class_metrics.csv")


# --------------------------------------------------------
# MAIN EVALUATION PIPELINE
# --------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("🩸 BLOOD GROUP CLASSIFICATION - MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print("🔧 Loading trained model...")
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    test_generator, class_names = load_test_data()
    
    # Generate predictions
    print("\n⚡ Generating predictions on test set...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    print(f"✅ Predictions complete: {len(predictions)} samples")
    
    # Compute metrics
    metrics, metrics_df = compute_all_metrics(y_true, y_pred, predictions, class_names)
    
    # Display key metrics
    print("\n" + "="*70)
    print("📈 KEY PERFORMANCE METRICS")
    print("="*70)
    print(f"Top-1 Accuracy:      {metrics['top1_accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy:      {metrics['top3_accuracy']*100:.2f}%")
    print(f"Weighted F1-Score:   {metrics['weighted_f1']:.4f}")
    print(f"Cohen's Kappa:       {metrics['cohen_kappa']:.4f}")
    if metrics['roc_auc_ovr']:
        print(f"ROC-AUC (OvR):       {metrics['roc_auc_ovr']:.4f}")
    print("="*70 + "\n")
    
    # Display per-class metrics
    print("📊 PER-CLASS METRICS")
    print("-"*70)
    print(metrics_df.to_string(index=False))
    print("\n")
    
    # Generate all visualizations
    print("🎨 Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
    plot_roc_curves(y_true, predictions, class_names)
    plot_precision_recall_curves(y_true, predictions, class_names)
    plot_prediction_distribution(y_true, y_pred, class_names)
    plot_misclassified_samples(test_generator, y_true, y_pred, predictions, class_names)
    
    # Generate report
    print("\n📝 Generating evaluation report...")
    generate_evaluation_report(metrics, metrics_df, y_true, y_pred, class_names)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"📁 All results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - normalized_confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - prediction_distribution.png")
    print("  - misclassified_examples.png")
    print("  - evaluation_report.txt")
    print("  - per_class_metrics.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()