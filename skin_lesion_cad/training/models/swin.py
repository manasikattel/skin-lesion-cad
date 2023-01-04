from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

import torchmetrics

def get_swinv2_model(model_cls: str, weights: str):
    """Retrieves a regnet model from torchvision.models

    Args:
        model_cls (str): Model name from torchvision.models.
        weights (str): Weights to use for the model.

    Raises:
        ValueError: If the model_cls is not supported.

    Returns:
        torch model
    """
    if model_cls == 'swin_v2_t':
        return models.swin_v2_t(weights=weights)
    elif model_cls == 'swin_v2_s':
        return models.swin_v2_s(weights=weights)
    elif model_cls == 'swin_v2_b':
        return models.swin_v2_b(weights=weights)
    else:
        raise ValueError(f"model_cls {model_cls} not supported")

class SwinModel(LightningModule):
    def __init__(self, num_classes,
                 model_class,
                 learning_rate=1e-4,
                 weights="IMAGENET1K_V1"):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = get_swinv2_model(model_class, weights=weights)

        # replace the last FC layer
        num_filters = self.model.head.in_features
        self.model.head = nn.Linear(num_filters, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass',
                                               num_classes=num_classes, top_k=1)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass',
                                               num_classes=num_classes, top_k=1)
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
        
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        batch_size = len(y)
        y_hat = self.model(x)
        
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)
        
        self.valid_acc(y_hat, y)
        self.log('valid_acc', self.valid_acc,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.2, patience=5)
        
        #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor":"val_loss"}}


if __name__ == "__main__":
    model = SwinModel()
    print(model)