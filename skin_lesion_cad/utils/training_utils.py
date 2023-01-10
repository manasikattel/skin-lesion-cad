import torch
import torch.nn as nn
import kornia.losses as losses
from skin_lesion_cad.training.losses import MWNLoss
from skin_lesion_cad.config.mwn_config import _C as cfg

mwnl_params = {'gamma':0.6, 'beta':0.1, 'type':'fix', 'sigmoid':'normal',
               'num_class_list':None, 'cfg':cfg, 'device':None}

def get_loss(name, device=None, num_classes=None):
    if name == 'cross_entropy':
        return  nn.CrossEntropyLoss()
    elif name == 'focal':
        return losses.FocalLoss(0.6)
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    elif name == 'mwnl':
        mwnl_params['device'] = f'cuda:{device}'
        mwnl_params['num_class_list'] = list(range(num_classes))
        loss = MWNLoss(para_dict=mwnl_params)
        loss.reset_epoch(0)
        return loss
    
    else:
        raise ValueError(f'Loss {name} not supported')
