import os
import h5py
import torch
from torch import nn
import numpy as np
from functools import partial
import hashlib
from typing import Union, List, Dict
from pathlib import Path

esm_model_dict = {
    "esm2_8M": {
        "esm_s_dim": 320,
        "esm_z_dim": 120,
        "esm_num_layers": 7,
    },
    "esm2_35M": {
        "esm_s_dim": 480,
        "esm_z_dim": 240,
        "esm_num_layers": 13,
    },
    "esm2_150M": {
        "esm_s_dim": 640,
        "esm_z_dim": 600,
        "esm_num_layers": 31,
    },
    "esm2_650M": {
        "esm_s_dim": 1280,
        "esm_z_dim": 660,
        "esm_num_layers": 34,
    },
    "esm2_3B": {
        "esm_s_dim": 2560,
        "esm_z_dim": 1440,
        "esm_num_layers": 37,
    },
    "esm2_15B": {
        "esm_s_dim": 5120,
        "esm_z_dim": 1920,
        "esm_num_layers": 49,
    },
}

load_fn = torch.hub.load
esm_registry = {
    "esm2_8M": partial(load_fn, "facebookresearch/esm:main", "esm2_t6_8M_UR50D"),
    "esm2_35M": partial(load_fn, "facebookresearch/esm:main", "esm2_t12_35M_UR50D"),
    "esm2_150M": partial(load_fn, "facebookresearch/esm:main", "esm2_t30_150M_UR50D"),
    "esm2_650M": partial(load_fn, "facebookresearch/esm:main", "esm2_t33_650M_UR50D"),
    "esm2_3B": partial(load_fn, "facebookresearch/esm:main", "esm2_t36_3B_UR50D"),
    "esm2_15B": partial(load_fn, "facebookresearch/esm:main", "esm2_t48_15B_UR50D"),
}

class ESMH5Cache:
    def __init__(self, cache_fname: str, mode='a'):
        """
        初始化 HDF5 缓存管理器
        cache_path: HDF5 文件路径
        """
        self.cache_fname = Path(cache_fname)
        self.cache_fname.parent.mkdir(exist_ok=True,parents=True)
        self.mode = mode
        

        if self.cache_fname.exists():
            try:
                with h5py.File(self.cache_fname, 'r'):
                    pass
            except OSError:
                print(f"[WARNING] 缓存文件损坏或不是HDF5: {self.cache_fname}, 将重新创建")
                self.cache_fname.unlink()  # 删除坏文件
        
        self.h5file = h5py.File(self.cache_fname, self.mode)

    @staticmethod
    def seq_hash(seq: str) -> str:
        """将氨基酸序列转为 SHA256 哈希字符串"""
        return hashlib.sha256(seq.encode('utf-8')).hexdigest()

    def exists(self, seq: str) -> bool:
        """检查序列是否存在缓存"""
        return self.seq_hash(seq) in self.h5file

    def get(self, seq: str) -> Union[torch.Tensor, None]:
        """获取缓存数据，如果不存在返回 None"""
        key = self.seq_hash(seq)
        if key in self.h5file:
            arr = self.h5file[key][:]
            return torch.from_numpy(arr)
        return None

    def put(self, seq: str, tensor: torch.Tensor):
        """写入缓存（单条），如果已存在则跳过"""
        key = self.seq_hash(seq)
        if key not in self.h5file:
            self.h5file.create_dataset(key, data=tensor.detach().cpu().numpy())

    def put_batch(self, seq_list: List[str], tensor_list: List[torch.Tensor]):
        """批量写入缓存，自动跳过已经存在的记录"""
        for seq, tensor in zip(seq_list, tensor_list):
            self.put(seq, tensor)

    def __len__(self) -> int:
        """返回当前缓存数据条数"""
        return len(self.h5file.keys())

    def keys(self) -> List[str]:
        """返回所有缓存的哈希键"""
        return list(self.h5file.keys())

    def close(self):
        """关闭HDF5文件"""
        self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



class ESM_embedding:
    def __init__(self, esm_name: str, device: str, cache_path: str, mode='a'):
        if esm_name not in esm_model_dict:
            raise ValueError(f"ESM模型 {esm_name} 不在已知列表中: {list(esm_model_dict.keys())}")

        self.esm_name = esm_name
        self.device = torch.device(device)
        self.mode = mode

        # 初始化缓存
        cache_file = Path(cache_path) / f"{esm_name}.h5py"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = ESMH5Cache(str(cache_file),mode=self.mode)

        # 加载ESM模型
        self.model, self.alphabet = esm_registry[esm_name]()
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def get(self, seq: str) -> torch.Tensor:
        """
        获取序列在 ESM 模型最后一层的embedding（去掉<cls>等特殊token），并缓存。
        如果缓存中存在则直接返回。
        """
            # --------------- 序列检查 ---------------
        # 1. 检查类型
        if not isinstance(seq, str):
            print(f"[ERROR] 序列类型错误: 期待 str, 收到 {type(seq).__name__}")
            return None

        # 2. 检查字符合法性（只允许标准氨基酸大写字母）
        allowed_residues = set("ACDEFGHIKLMNPQRSTVWY")  # 20 种标准氨基酸
        invalid_chars = [ch for ch in seq if ch not in allowed_residues]

        if invalid_chars:
            print(f"[ERROR] 序列包含非法字符: {set(invalid_chars)}")
            return None

        # 先查缓存
        cached = self.cache.get(seq)
        if cached is not None:
            return cached

        # ESM要求输入是 batch of (name, seq)
        batch_data = [("seq", seq)]
        _, _, tokens = self.batch_converter(batch_data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            # 输出字典包含 last_hidden_state
            results = self.model(tokens, repr_layers=[self.model.num_layers])
            token_repr = results["representations"][self.model.num_layers]  # (B, L, D)

        # 去掉特殊token (<cls>=index0, <eos>=最后)
        token_embedding = token_repr[0, 1:-1, :].cpu()

        # 写入缓存
        ## 可写模式
        if self.mode == 'a':
            self.cache.put(seq, token_embedding)

        return token_embedding

    def close(self):
        """关闭缓存文件"""
        self.cache.close()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时关闭缓存文件
        self.cache.close()