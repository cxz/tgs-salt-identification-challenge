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
                targets = [utils.cuda(t) for t in targets]
            t0 = targets[0]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            iou += [metric.iou_metric(t0.cpu().numpy(), (outputs.cpu().numpy() > .5))]

        valid_loss = np.mean(losses)  # type: float
        valid_iou = np.mean(iou).astype(np.float64)        

        print('valid loss: {:.4f}, iou: {:.4f}'.format(valid_loss, valid_iou))
        
        metrics = {'val_loss': valid_loss, 'val_iou': valid_iou}
        return metrics


