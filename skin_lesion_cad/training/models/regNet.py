from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import LightningModule

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
    def __init__(self, num_classes, model_cls, weights="IMAGENET1K_V2"):
        super().__init__()

        self.model = get_regnet_model(model_cls, weights=weights)

        # replace the last FC layer
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, num_classes)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = RegNetY()
    print(model)