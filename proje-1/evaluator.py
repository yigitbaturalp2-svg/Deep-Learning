import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

class Evaluator:
    @staticmethod
    def plot_learning_curves(history, title="Öğrenme Eğrisi"):
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Eğitim Kaybı', color='blue', linewidth=2)
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Doğrulama Kaybı', color='orange', linestyle='dashed', linewidth=2)
            
        plt.title(title)
        plt.xlabel('İterasyon (Epochs)')
        plt.ylabel('Kayıp (Loss)')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def evaluate_model(y_true, y_pred, labels=None, title="Model Başarımı"):
        print(f"\n--- {title} ---")
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        acc = accuracy_score(y_true_flat, y_pred_flat)
        precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f} (Weighted)")
        print(f"Recall   : {recall:.4f} (Weighted)")
        print(f"F1 Score : {f1:.4f} (Weighted)")
        
        print("\nSınıflandırma Raporu:")
        class_indices = np.arange(len(labels)) if labels is not None else None
        
        if labels is not None:
            print(classification_report(y_true_flat, y_pred_flat, labels=class_indices, target_names=labels, zero_division=0))
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=class_indices)
        else:
            print(classification_report(y_true_flat, y_pred_flat, zero_division=0))
            cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels if labels is not None else 'auto', 
                    yticklabels=labels if labels is not None else 'auto')
        plt.title(f"{title} - Karmaşıklık Matrisi (Confusion Matrix)")
        plt.xlabel("Tahmin")
        plt.ylabel("Gerçek")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
