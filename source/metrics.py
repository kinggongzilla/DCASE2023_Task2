import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score, confusion_matrix


def metrics(anomaly_score_path, decison_result_path):
    anomaly_df = pd.read_csv(anomaly_score_path)
    anomaly_df = anomaly_df.reset_index()
    decision_df = pd.read_csv(decison_result_path)
    y = np.zeros((len(anomaly_df), 3))
    for ind in anomaly_df.index:
        name = anomaly_df.iloc[ind][1]
        anomaly_score = anomaly_df.iloc[ind][2]
        y_pred = decision_df.iloc[ind][1]
        y_true = 0.0
        if 'anomaly' in name:
            y_true = 1.0
        y[ind] = (y_pred, y_true, anomaly_score)
    y_preds = y[:, 0]
    y_trues = y[:, 1]
    anomalies = y[:, 2]
    auc = roc_auc_score(y_trues, anomalies)
    p_auc = roc_auc_score(y_trues, anomalies, max_fpr=0.1)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_trues, y_preds).ravel()
    prec = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
    recall = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
    f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
    return auc, p_auc, prec, recall, f1



if __name__ == '__main__':
    ANOMALY_SCORE_PATH = 'results/bearing/anomaly_score_bearing_section_0.csv'
    DECISION_RESULT_PATH = 'results/bearing/decision_result_bearing_section_0.csv'
    auc, pauc, prec, recall, f1 = metrics(ANOMALY_SCORE_PATH, DECISION_RESULT_PATH)
