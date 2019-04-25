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

def iou(masks_true, masks_pred):
    """
    Get the IOU between each predicted mask and each true mask.

    Parameters
    ----------

    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    array-like
        A 2D array of shape (n_true_masks, n_predicted_masks), where
        the element at position (i, j) denotes the IoU between the `i`th true
        mask and the `j`th predicted mask.

    """
    if masks_true.shape[1:] != masks_pred.shape[1:]:
        raise ValueError('Predicted masks have wrong shape!')
    n_true_masks, height, width = masks_true.shape
    n_pred_masks = masks_pred.shape[0]
    m_true = masks_true.copy().reshape(n_true_masks, height * width).T
    m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
    numerator = np.dot(m_pred, m_true)
    denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)
    return numerator / (denominator - numerator)

def evaluate_image(masks_true, masks_pred, thresholds):
    """
    Get the average precision for the true and predicted masks of a single image,
    averaged over a set of thresholds

    Parameters
    ----------
    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    float
        The mean average precision of intersection over union between
        all pairs of true and predicted region masks.

    """
    int_o_un = iou(masks_true, masks_pred)
    benched = int_o_un > thresholds
    tp = benched.sum(-1).sum(-1)  # noqa
    fp = (benched.sum(2) == 0).sum(1)
    fn = (benched.sum(1) == 0).sum(1)
 