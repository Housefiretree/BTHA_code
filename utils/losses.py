import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.losses import DiceLoss, FocalLoss

# --- Lovasz Loss Implementation ---
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
    logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for x in l:
        n += 1
        acc += x
    return acc / n

class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, labels):
        return lovasz_hinge(logits, labels, per_image=self.per_image)

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_float = target.float()
        target_grad_x = F.conv2d(target_float, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_float, sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        return F.mse_loss(pred_edge, target_edge)

class HybridSegmentationLoss(nn.Module):
    """
    Dice + Focal + Boundary + Lovasz
    """
    def __init__(self, lambda_dice=1.0, lambda_focal=1.0, lambda_boundary=0.5, lambda_lovasz=0.5):
        super(HybridSegmentationLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        self.lambda_lovasz = lambda_lovasz

        self.dice_loss = DiceLoss(sigmoid=False, batch=True) 
        self.focal_loss = FocalLoss(reduction='mean')
        self.boundary_loss = EdgeLoss()
        self.lovasz_loss = LovaszHingeLoss() # 输入需要是 Logits

    def forward(self, pred_prob, target):
        """
        pred_prob: [B, 1, H, W] (Sigmoid Applied)
        target: [B, 1, H, W]
        """
        # Dice
        loss_d = self.dice_loss(pred_prob, target)
        
        # Focal & Lovasz need Logits (approx)
        pred_prob_clamped = torch.clamp(pred_prob, 1e-6, 1.0 - 1e-6)
        pred_logits = torch.log(pred_prob_clamped / (1 - pred_prob_clamped))
        
        loss_f = self.focal_loss(pred_logits, target)
        
        # Lovasz Hinge (Binary) - inputs are (B, H, W) logits and (B, H, W) labels
        loss_l = self.lovasz_loss(pred_logits.squeeze(1), target.squeeze(1))
        
        # Boundary
        loss_b = self.boundary_loss(pred_prob, target)

        total_loss = (self.lambda_dice * loss_d + 
                      self.lambda_focal * loss_f + 
                      self.lambda_boundary * loss_b +
                      self.lambda_lovasz * loss_l)
                      
        return total_loss, loss_d, loss_f, loss_b, loss_l