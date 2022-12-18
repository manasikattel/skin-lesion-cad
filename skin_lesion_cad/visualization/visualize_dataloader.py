
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule

from skin_lesion_cad.data.transforms import DeNormalize

root = Path("skin_lesion_cad").resolve()


@hydra.main(version_base="1.2", config_path=root/Path("config"), config_name="train_config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(OmegaConf.to_yaml(cfg))
    # seed_everything(cfg.seed)

    # print(f"Instantiating datamodule <{cfg._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config=cfg)
    data_module = datamodule.data
    data_module.prepare_data()
    for data in data_module.train_dataloader():
        # image = DeNormalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]
        #                     )(data["image"])
        image = data["image"]
        # image = (255 * (image - 1))
        image = (255 * image)
        image = image.numpy().astype(
            np.uint8).squeeze().transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(f"{data['name']} {data['label']}")
        plt.show()

if __name__ == "__main__":
    main()
