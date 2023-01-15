from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from skin_lesion_cad.utils.training_utils import get_loss

import torchmetrics

def get_regnet_model(model_cls: str, weights: str):
    """Retrieves a regnet model from torchvision.models

    Args:
        model_cls (str): Model name from torchvision.models.
        weights (str): Weights to use for the model.

    Raises:
        ValueError: If the model_cls is not supported.

    Returns:
        torch model
    """
    if model_cls == 'regnet_y_800mf':
        return models.regnet_y_800mf(weights=weights)
    elif model_cls == 'regnet_y_1_6gf':
        return models.regnet_y_1_6gf(weights=weights)
    elif model_cls == 'regnet_y_3_2gf':
        return models.regnet_y_3_2gf(weights=weights)
    elif model_cls == 'regnet_y_8gf':
        return models.regnet_y_8gf(weights=weights)
    elif model_cls == 'regnet_y_16gf':
        return models.regnet_y_16gf(weights=weights)
    else:
        raise ValueError(f"model_cls {model_cls} not supported")

class RegNetY(LightningModule):
    def __init__(self, num_classes,
                 model_class,
                 weights="IMAGENET1K_V2",
                 learning_rate=1e-4,
                 loss:str = "cross_entropy",
                 chkp_pretrained: str = None,
                 device=None):
        super().__init__()
        
        # save the hyperparameters
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = get_loss(loss, device=device, num_classes=num_classes)
        self.num_classes = num_classes
        self.chkp_pretrained = chkp_pretrained
        
        # load the model
        if self.chkp_pretrained is not None and self.chkp_pretrained != 'None':
            
            if self.chkp_pretrained =='/home/mira1/vlex_mira/vcad/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet/lightning_logs/version_1/checkpoints/epoch=52-valid_kappa=0.8180.ckpt':
                self.chkp_pretrained = '/home/user0/cad_vlanasi/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet/lightning_logs/version_1/checkpoints/epoch=52-valid_kappa=0.8180.ckpt'
                chkp_pretrained = '/home/user0/cad_vlanasi/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet/lightning_logs/version_1/checkpoints/epoch=52-valid_kappa=0.8180.ckpt'
            
            if self.chkp_pretrained =='/home/mira1/vlex_mira/vcad/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet_tune2class/lightning_logs/version_1/checkpoints/epoch=28-valid_acc=0.9136.ckpt':
                self.chkp_pretrained = '/home/user0/cad_vlanasi/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet_tune2class/lightning_logs/version_1/checkpoints/epoch=28-valid_acc=0.9136.ckpt'
                chkp_pretrained = '/home/user0/cad_vlanasi/skin-lesion-cad/outputs/regnety32gf_pretrain_imagenet_tune2class/lightning_logs/version_1/checkpoints/epoch=28-valid_acc=0.9136.ckpt'
            # load from checkpoint
            self.model = RegNetY.load_from_checkpoint(chkp_pretrained).model
            
            # replace las fc if trained with a different number of classes
            if self.model.fc.out_features != num_classes:
                num_filters = self.model.fc.in_features
                self.model.fc = nn.Linear(num_filters, num_classes)
            
            print(f"Loaded model {chkp_pretrained} and set for {num_classes} classes")
        else:
            self.model = get_regnet_model(model_class, weights=weights)

            # replace the last FC layer
            num_filters = self.model.fc.in_features
            self.model.fc = nn.Linear(num_filters, num_classes)
        
        # set up metrics to train
        if self.num_classes == 2:
            self.train_acc = torchmetrics.Accuracy(task='multiclass',
                                                num_classes=num_classes, top_k=1)
            self.valid_acc = torchmetrics.Accuracy(task='multiclass',
                                                num_classes=num_classes, top_k=1)

        elif self.num_classes >= 3:
            self.train_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes,task="multiclass")
            self.valid_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes,task="multiclass")
        
        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x).flatten(1)
        return y

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']  
        batch_size = len(y)
        y_hat = self.model(x)
        
        loss = self.criterion(y_hat, y)
        # need to manually average loss per batch
        if self.loss == 'focal':
            loss = loss.mean()

        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        if self.num_classes == 2:
            self.train_acc(y_hat, y)
            self.log('train_acc', self.train_acc,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)
        elif self.num_classes >= 3:
            self.train_kappa(y_hat, y)
            self.log('train_kappa', self.train_kappa,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        batch_size = len(y)
        y_hat = self.model(x)
        
        loss = self.criterion(y_hat, y)
        # need to manually average loss per batch
        if self.loss == 'focal':
            loss = loss.mean()
        self.log("val_loss", loss, batch_size=batch_size)
        
        if self.num_classes == 2:
            self.valid_acc(y_hat, y)
            self.log('valid_acc', self.valid_acc,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        elif self.num_classes >= 3:
            self.valid_kappa(y_hat, y)
            self.log('valid_kappa', self.valid_kappa,
                    on_step=True,  on_epoch=True,
                    prog_bar=True, logger=True,
                    batch_size=batch_size)

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        names = batch['name']
        batch_size = len(names)
        y_hat =  [i.argmax().item() for i in self.model(x)]
        return names,y_hat
    
    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        # for out in training_step_outputs:
        if self.loss == 'mwn':
            self.criterion.reset_epoch(self.current_epoch)
            
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.2, patience=8)
        
        #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor":"val_loss"}}


if __name__ == "__main__":
    model = RegNetY()
    print(model)