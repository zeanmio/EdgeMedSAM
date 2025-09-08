import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, gt):
    pred_bin = (pred > 127).astype(bool)
    gt_bin = (gt > 127).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return 2 * intersection / union if union > 0 else 1.0

def hausdorff(pred, gt):
    pred_pts = np.argwhere(pred > 127)
    gt_pts = np.argwhere(gt > 127)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('inf')
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)