import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule

def custom_collate_fn(batch):
    return list(batch)  # 将 batch 中的样本组合成一个列表
class RMSFDataset(Dataset):
    def __init__(self, fname):
        with open(fname,'rb') as f:
            lst = pickle.load(f)
        self.lst = sum(lst,[])

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        data = self.lst[idx]
        seq,temp_lst,rmsf_lst = data
        return seq,temp_lst,rmsf_lst

class RMSFDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=8,shuffle=True):
        super(RMSFDataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [0.9, 0.1])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=custom_collate_fn, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn,num_workers=2)


class RMSFpredictor_MLP(nn.Module):
    def __init__(self, mlp_hidden_size=512, num_mlp_layers=3, dropout_rate=0.1,device='cuda:0'):
        super(RMSFpredictor_MLP, self).__init__()

        layers = []
        input_size = 1280+1
        for _ in range(num_mlp_layers - 1):
            layers.append(nn.Linear(input_size, mlp_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = mlp_hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*layers)
        self.device = device
        
        self.to(self.device)
    
    def predict_from_ESMembeddings(self,embeddings,temperature,tolist=True):
        temperature = torch.tensor(temperature).to(torch.float32).to(self.device)
        embeddings = embeddings.to(self.device)
        batch_size, seq_length, hidden_size = embeddings.size()
        temperature = temperature.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_length, 1)
        sequence_output_with_temp = torch.cat((embeddings, temperature), dim=-1)
        res = self.mlp(sequence_output_with_temp).squeeze(-1)
        if not tolist:
            return res
        res = res.detach().cpu().numpy()
        res_lst = []
        for i,rmsf in enumerate(res):
            res_lst.append(rmsf)
        return res_lst
    def load_ckpt(self,fname):
        checkpoint = torch.load(fname, map_location='cpu')
        new_state_dict = {k.replace("module.", "").replace('predictor.',''): v for k, v in checkpoint['state_dict'].items()}
        #print(new_state_dict.keys())
        self.load_state_dict(new_state_dict, strict=False)



