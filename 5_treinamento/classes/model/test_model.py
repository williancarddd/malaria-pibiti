import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from classes.paths import Paths

class TestModel:
    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.test_path = self.paths.get_project_paths()["test"]

    def predict_test(self, pred, test_labels, folder_name):
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(test_labels, axis=1)
        y_pred_proba = np.array([np.max(p) for p in pred.ravel()])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        auc_score = roc_auc_score(test_labels.ravel(), y_pred_proba.ravel())

        data = pd.DataFrame({
            "TN": [tn],
            "FP": [fp],
            "FN": [fn],
            "TP": [tp],
            "AUC": [auc_score]
        })


        csv_path = os.path.join(self.test_path, f'{folder_name}_cm_refined.csv')
        data.to_csv(csv_path, sep=',', mode='a' if os.path.exists(csv_path) else 'w', index=False, header=not os.path.exists(csv_path))
        
        fpr, tpr, _ = roc_curve(test_labels.ravel(), y_pred_proba.ravel())
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        roc_data.to_csv(os.path.join(self.test_path, f'{folder_name}_fpr_tpr_refined.csv'), sep=',', index=False)
