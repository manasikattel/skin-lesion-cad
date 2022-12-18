
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from skin_lesion_cad.data.transforms import DeNormalize
from skin_lesion_cad.training.models.regNet import RegNetY
root = Path("skin_lesion_cad").resolve()


@hydra.main(version_base="1.2", config_path=root/Path("config"), config_name="train_config.yaml")
def main(cfg: DictConfig):

    # prepare data
    melanoma_data_module = hydra.utils.instantiate(config=cfg)
    melanoma_data_module = melanoma_data_module.data
    melanoma_data_module.prepare_data()

    # get model and trainer
    trainer = Trainer()
    model = RegNetY(2)

    trainer.fit(model=model, datamodule=melanoma_data_module)


if __name__ == "__main__":
    main()    
