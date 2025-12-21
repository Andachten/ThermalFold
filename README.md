# ThermalFold
Predicting Temperature-Dependent Protein Structure from Sequence by ThermalFold

## Installation
```git clone https://github.com/NJU-PZLab/ThermalFold.git```

```cd ThermalFold```

```conda env create -f environment.yml -n thermalFold```

```conda activate thermalFold```

```pip install .```
## Download model weight by:
```wget https://zenodo.org/records/17951162/files/model_weight.pt```

Please copy the downloaded model_weight.pt file to ./weight path

## Usage
Before inference: ```conda activate thermalFold```

On first use, downloading the ESM2 weights may take some time.
``` 
from ThermalFold.predict_utils import thermalFold_predictor
from ThermalFold.esm_utils import ESM_embedding,ESMH5Cache

cfg_fname = 'weight/model_conf.yaml'
weight_path  = 'weight/model_weight.pt'


##Thioredoxin
temperature = 430
sequence = 'HGSTTFNIQDGPDFQDRVVNSETPVVVDFHAQWCGPCKILGPRLEKMVAKQHGKVVMAKVDIDDHTDLAIEYEVSAVPTVLAMKNGDVVDKFVGIKDEDQLEAFLKKLIG'

## Create ESM Embeddings cache
## The prediction will run on cuda device 0
with ESM_embedding('esm2_650M',device='cuda:0',cache_path='./esm_cache') as esm:
    predictor = thermalFold_predictor(cfg_fname=cfg_fname,weight_path=weight_path,esm=esm,device='cuda:0')
    inputs = [[sequence,temperature]]
    ## return PDB format strings
    res = predictor.predict(inputs)[0]
```
More example, see example/TF_prediction.ipynb (which has been tested on NVIDIA RTX 4090 CUDA 12.8)

ThermalFold was trained on proteins with sequence lengths under 200 amino acids; therefore, it is recommended to test it on proteins that are also shorter than 200 amino acids.