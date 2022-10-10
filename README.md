# skin-lesion-cad

## Set up the environment

Rung following code to set up a conda environment with all the packages needed to run the project.

```
conda update -n base -c defaults conda &&
conda create -n cad_skin anaconda &&
conda activate cad_skin && 
pip install -r requirements.txt
```

## Download the data
Run `skin-lesion-cad/data/make_dataset.py` to download data sets and extract them into corresponding folders
```
python skin-lesion-cad/data/make_dataset.py
```
Or alternatively, download the data sets manually and extract them into corresponding folders.

At the end the following structure of the `/data` folder should be created:
```
data/
├── processed
└── raw
    ├── chall1
    │   ├── train
    │   │   ├── nevus
    │   │   └── others
    │   └── val
    │       ├── nevus
    │       └── others
    └── chall2
        ├── train
        │   ├── bcc
        │   ├── mel
        │   └── scc
        └── val
            ├── bcc
            ├── mel
            └── scc

18 directories
```

## Segment the images
Performs basic preprocessing in images:
* hair removal
* inpainting
* segmentation

Saves inpainted image and mask in corresponding folders in the `/data/processed`.

Run script 

```$ python -m skin_lesion_cad.utils.segm_script chall1/train/nevus```

from repo root `../skin-lesion-cad$`. Parameter passed (`chall1/train/nevus`) defines which images from which folder to process.

Could also pass `--resize` option to resize images by a factor. For example to downscale image by 2 run

``` python -m skin_lesion_cad.utils.segm_script chall1/train/nevus --resize 0.5```