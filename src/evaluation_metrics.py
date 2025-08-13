import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_val, y_val, save_path=None):
    y_pred = model.predict(X_val)
    y_val_classes = np.argmax(y_val, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_val_classes, y_pred_classes))

    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    plt.show()

