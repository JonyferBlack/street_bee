import numpy as np
import torch


def norm(input_):
    min_ = input_.min() * -1
    max_ = input_.max()
    output_ = input_ + min_
    output_ *= 0.4/max_
    return output_
    

def confusion(y_true, y_pred, threshold=0.5, eps=1e-9):
    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=0)
    precision = true_positive.div(y_pred.sum(dim=0).add(eps))
    recall = true_positive.div(y_true.sum(dim=0).add(eps))
    return recall, precision


def F1(prediction, truth):
    recall, precision = confusion(truth, prediction)
    return 2 * recall * precision / (precision + recall)


def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=0)
    precision = true_positive.div(y_pred.sum(dim=0).add(eps))
    recall = true_positive.div(y_true.sum(dim=0).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))