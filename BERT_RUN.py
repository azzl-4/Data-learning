#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

#%% 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

#%% ROC曲线和AUC
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

#%% 学习曲线
def plot_learning_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training loss')
    plt.plot(epochs, val_losses, 'o-', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

#%% 使用保存的結果進行可視化
if __name__ == "__main__":
    # 读取保存的结果
    data = np.load('roberta_results.npz')
    
    train_labels = data['train_labels']
    train_preds = data['train_preds']
    val_labels = data['val_labels']
    val_preds = data['val_preds']
    train_losses = data['train_losses']
    val_losses = data['val_losses']

    # 可視化混淆矩陣
    plot_confusion_matrix(val_labels, val_preds, class_names=['Non-suicide', 'Suicide'])

    # 可視化ROC曲線
    plot_roc_curve(val_labels, val_preds)

    # 可視化學習曲線
    plot_learning_curve(train_losses, val_losses)

# %%
