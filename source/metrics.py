import pandas as pd
import numpy as np
import os
import sys
from statistics import harmonic_mean
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score


def metrics(anomaly_score_path, decison_result_path):
    """
    input:
    anomaly_score_path: path to a csv file containing the filnames in the first and the anomaly score in the second column
    decision_result_path: path to a csv file containing the filnames in the first and the anomaly decisions in the second column
    assumption currently: filename contains 'anomaly' for anomalous sample
    returns:
    accuracy, auc, p_auc with p=0.1, precision, recall, f1
    """
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
    accurracy = accuracy_score(y_trues, y_preds)
    auc = roc_auc_score(y_trues, anomalies)
    p_auc = roc_auc_score(y_trues, anomalies, max_fpr=0.1)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_trues, y_preds).ravel()
    prec = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
    recall = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
    f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
    return accurracy, auc, p_auc, prec, recall, f1

def metrics_data(resultspath):
    """
    input:
    path to results directory with the following structure:
    results
        machinetype name
            anomaly_score.csv
            decision_result.csv
    returns:
    datframe containing the metrics for each machine type
    """
    columns = ["machine", "accuracy", "auc", "p_auc", "precision", "recall", "f1"]

    df = pd.DataFrame({col: [] for col in columns})

    for dir_name in os.listdir(resultspath):
        anomaly_score_path = None
        decision_result_path = None
        if os.path.isdir(os.path.join(resultspath, dir_name)):
            for file in os.listdir(os.path.join(resultspath, dir_name)):
                if 'anomaly_score' in file:
                    anomaly_score_path = os.path.join(os.path.join(resultspath, dir_name), file)
                if 'decision_result' in file:
                    decision_result_path = os.path.join(os.path.join(resultspath, dir_name), file)
        if anomaly_score_path and decision_result_path is not None:
            accuracy, auc, p_auc, prec, recall, f1 = metrics(anomaly_score_path, decision_result_path)
        else:
            raise f'{os.path.join(resultspath, dir_name)} does not contain anomaly_score or decision_results file'
        df = df.append({"machine": dir_name, "accuracy": accuracy, "auc": auc, "p_auc": p_auc, "precision": prec, "recall": recall, "f1": f1},
                       ignore_index=True)
    return df



def overall_score(resultspath):
    """
    input:
    path to results directory with the following structure:
    results
        machinetype name
            anomaly_score.csv
            decision_result.csv
    returns:
    harmonic mean of auc and pauc scores over all machine types
    """
    pauc_auc_list = []
    for dir_name in os.listdir(resultspath):
        anomaly_score_path = None
        decision_result_path = None
        if os.path.isdir(os.path.join(resultspath, dir_name)):
            for file in os.listdir(os.path.join(resultspath, dir_name)):
                if 'anomaly_score' in file:
                    anomaly_score_path = os.path.join(os.path.join(resultspath, dir_name), file)
                if 'decision_result' in file:
                    decision_result_path = os.path.join(os.path.join(resultspath, dir_name), file)
        if anomaly_score_path and decision_result_path is not None:
            accuracy, auc, p_auc, prec, recall, f1 = metrics(anomaly_score_path, decision_result_path)
        else:
            raise f'{os.path.join(resultspath, dir_name)} does not contain anomaly_score or decision_results file'
        pauc_auc_list.append(auc)
        pauc_auc_list.append(p_auc)
    return harmonic_mean(pauc_auc_list)




if __name__ == '__main__':
    ANOMALY_SCORE_PATH = 'results/bearing/anomaly_score_bearing_section_0.csv'
    DECISION_RESULT_PATH = 'results/bearing/decision_result_bearing_section_0.csv'
    RESULT_PATH = 'results'
    accuracy, auc, p_auc, prec, recall, f1 = metrics(ANOMALY_SCORE_PATH, DECISION_RESULT_PATH)
    overall = overall_score(RESULT_PATH)
    df = metrics_data(RESULT_PATH)
