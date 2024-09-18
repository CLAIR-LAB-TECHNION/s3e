from collections import defaultdict
import glob
from itertools import combinations
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
)
from tqdm.auto import tqdm

from .misc import squash_predicate


def get_cooccurrence_matrix(states_data_dir, as_table=False):
    # load all state dictionaries
    discovered_states = {}
    for fname in tqdm(
        glob.glob(os.path.join(states_data_dir, "*.json")),
        desc="loading co-occurrence data",
    ):
        with open(fname, "r") as f:
            discovered_states[fname] = json.load(f)

    # save as a table. rows are datapoints, columns are predicates
    df = pd.DataFrame.from_dict(discovered_states).T

    # drop duplicates and
    df.drop_duplicates(inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    # calculate boolean coocurrence matrix
    cooc_table = df.T @ df

    if as_table:
        return cooc_table
    else:
        return cooc_table.to_numpy()


def threshold_with_cooc_mat(y_score, threshold, cooc_mat=None):
    y_pred = y_score > threshold
    if cooc_mat is None:
        return y_pred

    for dp in range(len(y_pred)):  # cancel out for each data point
        cancel_out = {}
        for j, k in combinations(range(y_pred.shape[-1]), 2):  # iterate all pairs
            if (
                y_pred[dp, j] and y_pred[dp, k] and not cooc_mat[j, k]
            ):  # both true but do not co-occur
                if y_score[dp, j] > y_score[dp, k]:  # j cancels out k
                    cancel_out.setdefault(j, []).append(k)
                else:  # k cancels out j
                    cancel_out.setdefault(k, []).append(j)

        # sort canceling predicates by score
        cancelers_by_score = sorted(cancel_out.keys(), key=lambda k: y_score[dp, k], reverse=True)
        for canceler in cancelers_by_score:
            if y_pred[dp, canceler]:  # this predicate may have already been cancelled
                for cancelled in cancel_out[
                    canceler
                ]:  # cancel out all items it is supposed to cancel
                    y_pred[dp, cancelled] = False

    return y_pred


def precision_recall_scores(y, y_pred):
    cmat = confusion_matrix(y, y_pred).ravel()

    # consider item that always has the same label value
    if cmat.size == 1:
        return 1.0, 1.0

    tn, fp, fn, tp = cmat

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def precision_recall_curve_with_cooc_mat(
    ground_truths, predicted_scores, num_thresholds, cooc_mat=None, ignored_keys=None
):
    # filter out ignored keys
    ignored_keys = ignored_keys or []
    ground_truths = {
        k1: {k2: v2 for k2, v2 in v1.items() if k2 not in ignored_keys}
        for k1, v1 in ground_truths.items()
        if k1
        in predicted_scores.keys()  # also filter GT based on predictions that we have
    }
    predicted_scores = {
        k1: {k2: v2 for k2, v2 in v1.items() if k2 not in ignored_keys}
        for k1, v1 in predicted_scores.items()
    }

    pred_to_idx = {
        pred: i for i, pred in enumerate(sorted(next(iter(ground_truths.values()))))
    }
    y = squash_predicate(ground_truths)
    y_score = squash_predicate(predicted_scores)

    thresholds = np.linspace(0, 1, num_thresholds + 1)
    precision = {label: [] for label in pred_to_idx.keys()}
    recall = {label: [] for label in pred_to_idx.keys()}
    for t in tqdm(thresholds, leave=False, desc="calculating per-threshold metric"):
        y_pred = threshold_with_cooc_mat(y_score, t, cooc_mat)
        for predicate, i in pred_to_idx.items():
            precision_t, recall_t = precision_recall_scores(y[:, i], y_pred[:, i])
            precision[predicate].append(precision_t)
            recall[predicate].append(recall_t)

        precision_t_macro = np.mean(list(map(lambda p: p[-1], precision.values())))
        precision.setdefault("macro average", []).append(precision_t_macro)
        recall_t_macro = np.mean(list(map(lambda r: r[-1], recall.values())))
        recall.setdefault("macro average", []).append(recall_t_macro)

        all_precision_t, all_recall_t = precision_recall_scores(
            y.flatten(),
            y_pred.flatten(),
        )
        precision.setdefault("micro average", []).append(all_precision_t)
        recall.setdefault("micro average", []).append(all_recall_t)

    return precision, recall, thresholds


def ap_score_with_cooc_mat(
    ground_truths, prediction_scores, num_thresholds, cooc_mat=None, ignored_keys=None
):
    precision, recall, thresholds = precision_recall_curve_with_cooc_mat(
        ground_truths, prediction_scores, num_thresholds, cooc_mat, ignored_keys
    )
    ap_score = {
        label: -np.sum(np.diff(recall[label]) * np.array(precision[label])[:-1])
        for label in precision
    }
    ap_score["macro average"] = np.sum(list(ap_score.values())) / len(ap_score)

    return ap_score, precision, recall, thresholds


def acc_curve_with_cooc_mat(
    ground_truths, predicted_scores, num_thresholds, cooc_mat=None
):
    pred_to_idx = {
        pred: i for i, pred in enumerate(sorted(next(iter(ground_truths.values()))))
    }
    y = squash_predicate(
        {k: v for k, v in ground_truths.items() if k in predicted_scores}
    )
    y_score = squash_predicate(predicted_scores)

    thresholds = np.linspace(0, 1, num_thresholds + 1)
    accuracy = {label: [] for label in pred_to_idx.keys()}
    for t in tqdm(thresholds, leave=False, desc="calculating per-threshold metric"):
        y_pred = threshold_with_cooc_mat(y_score, t, cooc_mat)
        for predicate, i in pred_to_idx.items():
            accuracy[predicate].append(accuracy_score(y[:, i], y_pred[:, i]))
        accuracy.setdefault("average", []).append(
            accuracy_score(y.flatten(), y_pred.flatten())
        )

    return accuracy, thresholds
