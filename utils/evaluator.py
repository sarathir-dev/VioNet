from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os


def evaluate_model(model, X_test, y_test, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred,
          target_names=["Non-violence", "Violence"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                  "Non-violence", "Violence"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - VioNet")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve - VioNet")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()
