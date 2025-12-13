# ThermalFold
Predicting Temperature-Dependent Protein Structure from Sequence by ThermalFold

## Install dependencies
``` pip install -r requirements.txt```

## Usage
``` 
from predict import thermalFold
from omegaconf import OmegaConf
from lightning.pytorch import seed_everything
import torch

model_name = f'model_weight/last.ckpt'
cfg_name = 'train_conf.yaml'
cfg = OmegaConf.load(cfg_name)
model = thermalFold(model_name=model_name,cfg=cfg,device='cuda:0')

seed_everything(42)
seq  = 'GSTEKQLEAIDQLHLEYAKRAAPFNNWMESAMEDLQDMFIVHTIEEIEGLISAHDQFKSTLPDADREREAILAIHKEAQRIAESNHIKLSGSNPYTTVTPQIINSKWEKVQQLVPKRDHALLEEQSKQQ'
temp = 400
inputs = [[seq,temp]]
results = model.predict(inputs,num_samples=1)
with open('result.pdb','w') as f:
  f.write(results[0])
```
