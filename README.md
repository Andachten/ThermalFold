# ThermalFold
Predicting Temperature-Dependent Protein Structure from Sequence by ThermalFold

## Create conda environment and Install dependencies
```git clone https://github.com/Andachten/ThermalFold.git```

```conda env create -f environment.yml -n thermalFold```

```conda actiavte thermalFold```

```pip install .```
## Download model weight by:
```wget https://zenodo.org/records/17951162/files/model_weight.pt -p weight/```

## Usage
Before inference: ```conda actiavte thermalFold```

On first use, downloading the ESM2 weights may take some time.
``` 
from ThermalFold.predict_utils import thermalFold_predictor
from ThermalFold.esm_utils import ESM_embedding,ESMH5Cache

cfg_fname = 'weight/model_conf.yaml'
weight_path  = 'weight/model_weight.pt'


##Thioredoxin
temperature = 410
sequence = 'HGSTTFCIQDGPDFQDRVVNSETPVVVDFHAQWCGPCKILGPRLEKMVAKQHGKVVMACVDIDDHTDLAIEYEVSAVPTVLAMKNGDVVDKFVGIKDEDQLEAFLKKLIG'

## Create ESM Embeddins cache
## The prediction will run on cuda device 0
with ESM_embedding('esm2_650M',device='cuda:0',cache_path='./esm_cache') as esm:
    predictor = thermalFold_predictor(cfg_fname=cfg_fname,weight_path=weight_path,esm=esm,device='cuda:0')
    inputs = [[seq,temp]]
    ## return PDB format strings
    res = predictor.predict(inputs)[0]
```
More example, see example/TF_prediction.ipynb