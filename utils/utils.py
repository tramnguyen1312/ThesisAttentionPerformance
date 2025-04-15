import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt

def plot_training_log(log_path):

    log_df = pd.read_csv(log_path)

    plt.figure(figsize=(12, 5))

    # Vẽ biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(log_df["epoch"], log_df["train_accuracy"], label="Train Accuracy")
    plt.plot(log_df["epoch"], log_df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)

    # Vẽ biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(8, 6)):
    """
    Vẽ confusion matrix từ ground truth và predicted labels.

    :param y_true: Danh sách nhãn thực tế.
    :param y_pred: Danh sách nhãn dự đoán.
    :param class_names: Danh sách tên các lớp (class names).
    :param normalize: Có chuẩn hóa hàng hay không.
    :param figsize: Kích thước biểu đồ.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()