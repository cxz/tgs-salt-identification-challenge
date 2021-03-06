import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np
import losses_lovasz
import losses_lovasz2


# def jaccard_distance(outputs, targets):
#     smooth = 100
#     jaccard_target = (targets == 1).float()
#     jaccard_output = F.sigmoid(outputs)
#
#     intersection = (jaccard_output * jaccard_target).sum()
#     sum_ = jaccard_output.sum() + jaccard_target.sum()
#     jacc = (intersection + smooth) / (sum_ - intersection + smooth)
#     return 1 - jacc

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

def jaccard(preds, trues, weight=None, is_average=True, eps=1e-6):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1)

def dice_loss(preds, trues, weight=None, is_average=True, eps=1e-6):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossBinaryMixedDiceBCE:
    def __init__(self, dice_weight, bce_weight, dice_smooth=0):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def __call__(self, outputs, targets):
        smooth = 0
        dice_activation = 'sigmoid'
        dice_loss_ = self.dice_weight * (1 - dice_loss(outputs, targets.float()))
        bce_loss_ = self.bce_weight * self.bce_loss(outputs, targets)
        return dice_loss_ + bce_loss_



class LossLovasz:
    def __call__(self, outputs, targets):
        return losses_lovasz.lovaszloss(outputs, targets)

class LossHinge:
    def __call__(self, outputs, targets):
        return losses_lovasz2.lovasz_hinge(outputs, targets)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)
    
class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)
    
class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, weight=None, size_average=True):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.jacc = JaccardLoss()
        self.dice = DiceLoss()
        self.mapping = {'bce': self.bce,
                        'jaccard': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, input, target):
        loss = 0
        for k, v in self.weights.items():
            if not v:
                continue
            val = self.mapping[k](input, target)
            self.values[k] = val
            if k != 'bce':
                loss += self.weights[k] * (1 - val)
            else:
                loss += self.weights[k] * val
        return loss
