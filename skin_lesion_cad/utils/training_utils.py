import torch
import torch.nn as nn
import kornia.losses as losses
from skin_lesion_cad.training.losses import MWNLoss

def get_loss(name):
    if name == 'cross_entropy':
        return  nn.CrossEntropyLoss()
    elif name == 'focal':
        return losses.FocalLoss(0.6)
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    elif name == 'mwnl':
        return MWNLoss()
    
    else:
        raise ValueError(f'Loss {name} not supported')
