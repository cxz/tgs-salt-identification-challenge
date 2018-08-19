import numpy as np
import utils
from torch import nn
import torch
import metric


def validation_binary(model: nn.Module, criterion, valid_loader):
    with torch.no_grad():
        model.eval()
        losses = []
        iou = []

        for inputs, targets in valid_loader:                        
            inputs = utils.cuda(inputs)
            with torch.no_grad():
                targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            targets_npy = targets.cpu().numpy()
            outputs_npy = outputs.cpu().numpy()

            # iou per image, not batch
            for t, o in zip(targets_npy, outputs_npy):
                iou.append(metric.iou_metric(t, o > .5))
                
            # in batch it's distorted
            #iou += [metric.iou_metric(targets_npy, (outputs_npy > .5))]

        valid_loss = np.mean(losses)  # type: float
        valid_iou = np.mean(iou).astype(np.float64)        

        print('valid loss: {:.4f}, iou: {:.4f}'.format(valid_loss, valid_iou))
        
        metrics = {'val_loss': valid_loss, 'val_iou': valid_iou}
        return metrics


