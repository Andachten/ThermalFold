import hydra
from ThermalFold.data_utils import covert_SeqCoord2pdb
from ThermalFold.model.thermalfold_alphafold3 import Alphafold3
from ThermalFold.alphafold3_pytorch import Alphafold3Input
from ThermalFold.alphafold3_pytorch.models.components.alphafold3 import (
    ComputeModelSelectionScore,
    LossBreakdown,
    Sample,
)
from ThermalFold.alphafold3_pytorch.utils.model_utils import dict_to_device
from ThermalFold.alphafold3_pytorch.utils.utils import exists, not_exists
import torch
from omegaconf import OmegaConf
from itertools import islice
import pathlib

def create_batches(data, batch_size):
    it = iter(data)  # 将列表转换为迭代器
    while True:
        batch = list(islice(it, batch_size))  # 从迭代器中提取batch_size个元素
        if not batch:  # 如果 batch 为空，结束
            break
        yield batch
class thermalFold_predictor:
    def __init__(
        self,
        cfg_fname:str|pathlib.Path, 
        weight_path:str|pathlib.Path,
        esm,
        device:str='cuda:0'
        ):

        self.weight_path = weight_path
        self.device = device
        cfg = OmegaConf.load(cfg_fname)
        self.model = hydra.utils.instantiate(cfg.model)
        self.setup()
        self.esm = esm
    def setup(self):
        checkpoint = torch.load(self.weight_path, map_location='cpu',weights_only=False)
        new_state_dict = {k.replace('network.',''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.compute_model_selection_score = ComputeModelSelectionScore(
            is_fine_tuning=False
        )
        self.model.to(self.device)
    
    @torch.no_grad()
    def prepare(self):
        self.af3i_lst = []
        for i,(sequence,temperature) in enumerate(self.seq_input):
            ESM_emb_data = self.esm.get(sequence)
            seq_len = len(sequence)
            af3i = Alphafold3Input(
                proteins=[sequence],
                temperature=temperature,
                chains=torch.Tensor([0, -1]).to(int),
                is_molecule_types=torch.Tensor([True, False, False, False, False]).to(bool).repeat(seq_len, 1),
                plm_embedding=ESM_emb_data,
                seq=sequence,
                fname='XX',
            )
            self.af3i_lst.append(af3i)
    
    @torch.inference_mode()
    def get_pdb(
        self,
        sampled_atom_pos,  # type: ignore
        atom_mask,  # type: ignore
        seq_lst,
        b_factors,  # type: ignore
    ):
        batch_size = len(atom_mask)
        res_lst = []
        for b in range(batch_size):
            example_atom_mask = atom_mask[b]
            sampled_atom_positions = sampled_atom_pos[b][example_atom_mask].float().cpu().numpy()
            example_b_factors = (
                b_factors[b][example_atom_mask].float().cpu().numpy()
                if exists(b_factors)
                else None
            )
            seq = seq_lst[b]
            strings_result = covert_SeqCoord2pdb(seq,
                                                 sampled_atom_positions,
                                                 temperature_factor=example_b_factors)
            res_lst.append(strings_result)
        return res_lst
    
    @torch.no_grad()
    def predict_(self,batch,num_samples):
        batch_dict = self.model.batch_dict_prepare(batch)
        batch_dict = dict_to_device(batch_dict,device=self.device)
        samples = []
        for _ in range(num_samples):
            batch_sampled_atom_pos, logits = self.model(
                **batch_dict,
                return_loss=False,
                return_confidence_head_logits=True,
                return_distogram_head_logits=True,
                rollout_show_tqdm_pbar=False,
            )
            plddt = self.compute_model_selection_score.compute_confidence_score.compute_plddt(
                logits.plddt
            )
            samples.append((batch_sampled_atom_pos, logits.pde, plddt, logits.distance))
        plddt_ndx = torch.vstack([x[2].mean(axis=1).detach().cpu() for x in samples])
        plddt_top_ndx = plddt_ndx.max(axis=0)

        top_batch_sampled_atom_pos = torch.vstack([x[0] for x in samples])[plddt_top_ndx.indices]
        top_sample_plddt = torch.vstack([x[2] for x in samples])[plddt_top_ndx.indices]

        seq_lst = batch_dict['seq_lst']
        pdb_strings = self.get_pdb(
                sampled_atom_pos=top_batch_sampled_atom_pos,
                atom_mask=~batch_dict["missing_atom_mask"],
                b_factors=top_sample_plddt,
                seq_lst=seq_lst,
            )
        return pdb_strings

    
    def predict(self,seq_input,batch_size=1,num_samples=5):
        self.seq_input = seq_input
        self.prepare()
        batches = create_batches(self.af3i_lst, batch_size)
        result_lst = []
        for i,batch in enumerate(batches):
            result = self.predict_(batch, num_samples)
            result_lst.append(result)
        return sum(result_lst,[])