
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, LightningModule

from skin_lesion_cad.data.transforms import DeNormalize
from skin_lesion_cad.training.models.regNet import RegNetY
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from skin_lesion_cad.validation.validate import validate2csv

from tqdm import tqdm
root = Path("skin_lesion_cad").resolve()


@hydra.main(version_base="1.2", config_path=root/Path("config"), config_name="train_config.yaml")
def main(cfg: DictConfig):

    hydra_logpath = Path(HydraConfig.get().run.dir).resolve()
    print("LOGGING TO: ", hydra_logpath)
    
    
    if cfg.model.num_classes == 3:
        cllback_filename = "{epoch:02d}-{valid_kappa:.4f}"
        cllback_monitor = "valid_kappa"

    elif cfg.model.num_classes == 2:
        cllback_filename = "{epoch:02d}-{valid_acc:.4f}"
        cllback_monitor = "valid_acc"
    # saves top-K checkpoints based on "val_loss" metric
    
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                        monitor=cllback_monitor,
                                        mode="max",
                                        filename=cllback_filename)

    
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=hydra_logpath.resolve(),
                               version=1, name="lightning_logs")
    # prepare data
    melanoma_data_module = hydra.utils.instantiate(config=cfg)
    melanoma_data_module = melanoma_data_module.data
    melanoma_data_module.prepare_data()
    
    # get model and trainer
    model = hydra.utils.instantiate(config=cfg.model)
    trainer = Trainer(**cfg.pl_trainer, logger=logger,
                      auto_lr_find=True,
                      callbacks=[checkpoint_callback])

    # find optimal learning rate
    print('Default LR: ', model.learning_rate)
    trainer.tune(model, datamodule=melanoma_data_module)
    print('Tuned LR: ', model.learning_rate)
    
    # train model
    print("Training model...")
    trainer.fit(model=model,
                datamodule=melanoma_data_module)

if __name__ == "__main__":
    main()