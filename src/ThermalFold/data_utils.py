import random
import gemmi
import torch
import h5py
import pickle
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from ThermalFold.alphafold3_pytorch import Alphafold3Input
from ThermalFold.alphafold3_pytorch.common.amino_acid_constants import restype_name_to_compact_atom_names
from typing import List,Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
def parse_key_name(key_name='NAME=P00283:TEMP=281:SEQLEN=128'):
    dic = {}
    for v in key_name.split(":"):
        k,v = v.split("=")
        dic[k] = v
    return dic

def cif_to_pdb(cif_file, pdb_file):
    cif_file,pdb_file = str(cif_file),str(pdb_file)
    # 读取 CIF 文件
    structure = gemmi.read_structure(cif_file)
    
    # 写入 PDB 文件
    structure.write_pdb(pdb_file)
def pdb_to_cif(pdb_file,cif_file):
    pdb_file,cif_file = str(pdb_file),str(cif_file)
    # 读取 PDB 文件
    structure = gemmi.read_structure(pdb_file)
    # 保存为 mmCIF 格式
    structure.make_mmcif_block().write_file(cif_file)
def cif_to_sequences(cif_path: str) -> dict:
    """
    从 mmCIF 文件提取蛋白质序列（单字母），返回 {chainID: sequence} 字典
    
    参数:
        cif_path (str): mmCIF 文件路径
        
    返回:
        dict: {chainID: 单字母序列}
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    ppb = PPBuilder()
    
    sequences = {}
    for model in structure:  # 遍历模型
        for chain in model:  # 遍历链
            seq = ""
            for pp in ppb.build_peptides(chain):  # 提取多肽片段
                seq += str(pp.get_sequence()).replace("X", "")  # 去掉未知残基 X
            if seq:  # 只保留非空序列
                sequences[chain.id] = seq
    
    return sequences

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O'
}
one_to_three = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "X": "UNK"
}
def get_seq(u,chain):
    return ''.join([three_to_one[res.resname] for res in u.select_atoms(f'protein and chainID {chain}').residues])

def remove_blankInlst(lst):
    return [x for x in lst if x != '']
def universe_to_string(universe):
    """
    Converts an MDAnalysis Universe to a PDB format string without writing to a file.

    Parameters:
        universe (MDAnalysis.Universe): The Universe to convert.

    Returns:
        str: The PDB format string representation of the Universe.
    """
    from MDAnalysis.coordinates.PDB import PDBWriter
    import io
    buffer = io.StringIO()
    with PDBWriter(buffer) as writer:
        writer.write(universe)
        pdb_string = buffer.getvalue()
    buffer.close()
    return pdb_string

def extract_coordinates(u):
    """
    Extracts atomic coordinates based on a predefined atom order for each residue type.
    Includes special handling for terminal residues with 'OT1' replacing 'O'.
    
    Parameters:
        pdb_file (str): Path to the PDB file.
    
    Returns:
        tuple: A dictionary of coordinates by residue (`compact_coordinates`)
               and a numpy array of all atom coordinates (`coord_arr`).
    """
    
    # Select protein atoms
    protein = u.select_atoms("protein")
    
    compact_coordinates = {}
    coord_lst = []  # To store all coordinates in sequence
    coord_res_lst = []
    
    for residue in protein.residues:
        resname = residue.resname
        atom_order = restype_name_to_compact_atom_names.get(resname, [])
        coords = []
        coord_res_lst.append([])
        for atom_name in atom_order:
            if not atom_name:  # Skip empty entries
                coords.append(None)
                continue
            
            # Attempt to select the atom
            atom = residue.atoms.select_atoms(f"name {atom_name}")
            
            # Handle terminal residue 'O' substitution with 'OT1'
            if not atom and atom_name == "O":
                #print(f"Substituting OT1 for O in residue {residue.resname} (resid {residue.resid})")
                atom = residue.atoms.select_atoms("name OT1")
            
            if not atom and residue.resname == 'ILE':
                #print(f"Substituting CD for CD1 in residue {residue.resname} (resid {residue.resid})")
                atom = residue.atoms.select_atoms("name CD")
            
            if atom:
                coords.append(atom.positions[0])  # Append atomic position
                coord_lst.append(atom.positions[0])  # Add to global list
                coord_res_lst[-1].append(atom.positions[0])
            else:
                raise ValueError(
                    f"Missing atom '{atom_name}' in residue {residue.resname} (resid {residue.resid})."
                )
        
        compact_coordinates[residue.resid] = coords
    
    # Stack all coordinates into a single numpy array
    coord_arr = np.vstack(coord_lst)
    
    return {'arr':coord_arr,'dic':compact_coordinates,'res_lst':coord_res_lst}

def covert_SeqCoord2pdb(seq, coordinates, backbone_mode=False, temperature_factor=None):
    seq_3code = [one_to_three[i] for i in seq]
    atomname_lst = [remove_blankInlst(restype_name_to_compact_atom_names[resname]) 
                    for resname in seq_3code]
    
    if backbone_mode or coordinates.shape[0] == len(seq) * 4:
        atomname_lst = [lst[:4] for lst in atomname_lst]

    # Initialize universe
    universe = mda.Universe.empty(
        n_atoms=len(coordinates),  # Total number of atoms
        n_residues=len(seq),  # Total number of residues
        atom_resindex=np.repeat(np.arange(len(seq)), [len(atoms) for atoms in atomname_lst]),  # Atom-to-residue mapping
        residue_segindex=np.zeros(len(seq)),  # All residues belong to the same segment
        trajectory=True
    )
    
    # Add topology attributes
    universe.add_TopologyAttr("resnames", [aa for aa in seq_3code])
    universe.add_TopologyAttr("names", [atomname for atomname in sum(atomname_lst, [])])
    universe.add_TopologyAttr("resids", np.arange(1, len(seq) + 1))
    universe.add_TopologyAttr("segids", ["PROT"])
    universe.add_TopologyAttr("chainIDs", ["A"] * len(coordinates))  # 设置所有 chainID 为 A

    # Assign positions
    universe.atoms.positions = coordinates

    # Assign temperature factors
    if temperature_factor is None:
        universe.add_TopologyAttr("tempfactors", np.ones(len(coordinates)))
    else:
        if len(temperature_factor) != len(coordinates):
            raise ValueError("Length of temperature_factor array must match the number of atoms.")
        universe.add_TopologyAttr("tempfactors", temperature_factor)

    # Convert universe to string
    strings_result = universe_to_string(universe)
    return strings_result


def get_ThermalFold_input(pdb_fname,temperature,esm):
    pdb_fname = Path(pdb_fname)
    name = pdb_fname.stem
    u = mda.Universe(pdb_fname)
    sequence = get_seq(u,'A')
    seq_len = len(sequence)
    esm_em = esm.get(sequence)
    coord_res_lst = extract_coordinates(u)['res_lst']
    coord_res_tensor_lst = [torch.Tensor(np.array(res_lst)) for res_lst in coord_res_lst]
    coord_arr = extract_coordinates(u)['arr']
    af3i = Alphafold3Input(
            proteins=[sequence],
            missing_atom_indices=[None] * len(coord_res_tensor_lst),
            atom_pos=coord_res_tensor_lst,
            temperature=temperature,
            chains=torch.Tensor([0, -1]).to(int),
            resolved_labels=torch.ones(coord_arr.shape[0]).to(int),
            is_molecule_types=torch.Tensor([True, False, False, False, False]).to(bool).repeat(seq_len, 1),
            plm_embedding=esm_em,
            seq=sequence,
            fname=name,
        )
    dic = dict(
            Alphafold3Input=af3i,
            sequence_length=seq_len,
            source=str(pdb_fname),
            ID=pdb_fname.stem,
            temperature=temperature,
            esm_name = esm.esm_name
        )
        
    return dic
def Seq_to_Cluster(cluster_seq_dic:Dict[str,List[str]])->Dict[str,str]:
    seq_to_cluster = {}
    for clu, seqs in cluster_seq_dic.items():
        for seq in seqs:
            seq_to_cluster[seq] = clu
    return seq_to_cluster

def split_cluster(cluster_seq_dic: Dict[str, List[str]], 
                  train_ratio=0.8, 
                  val_ratio=0.1, 
                  seed=42,
                  train_include: Optional[List[str]] = None,
                  val_include: Optional[List[str]] = None,
                  test_include: Optional[List[str]] = None
                 ) -> Tuple[List[str], List[str], List[str]]:

    random.seed(seed)
    train_include = set(train_include or [])
    val_include = set(val_include or [])
    test_include = set(test_include or [])

    # 建立序列 -> 簇映射
    seq_cluster_dic = Seq_to_Cluster(cluster_seq_dic)

    # 先确定强制分配的簇集合
    forced_train_clusters = set()
    forced_val_clusters = set()
    forced_test_clusters = set()

    for seq in train_include:
        forced_train_clusters.add(seq_cluster_dic[seq])
    for seq in val_include:
        forced_val_clusters.add(seq_cluster_dic[seq])
    for seq in test_include:
        forced_test_clusters.add(seq_cluster_dic[seq])

    # 冲突检查：一个簇不能在多个集合
    all_forces = {}
    for clu in forced_train_clusters:
        all_forces.setdefault(clu, []).append("train")
    for clu in forced_val_clusters:
        all_forces.setdefault(clu, []).append("val")
    for clu in forced_test_clusters:
        all_forces.setdefault(clu, []).append("test")

    for clu, sets in all_forces.items():
        if len(sets) > 1:
            raise ValueError(f"冲突: 簇 {clu} 被强制分配到多个集合 {sets}")

    
    clusters = [(name, len(seqs), seqs) for name, seqs in cluster_seq_dic.items()]
    random.shuffle(clusters)  # 为非强制分配部分打乱顺序

    total_seqs = sum(size for _, size, _ in clusters)
    train_target = total_seqs * train_ratio
    val_target = total_seqs * val_ratio

    train_set, val_set, test_set = [], [], []
    train_count = val_count = test_count = 0

    
    for name, size, seqs in clusters:
        if name in forced_train_clusters:
            train_set.extend(seqs)
            train_count += size
        elif name in forced_val_clusters:
            val_set.extend(seqs)
            val_count += size
        elif name in forced_test_clusters:
            test_set.extend(seqs)
            test_count += size

    
    for name, size, seqs in clusters:
        if (name in forced_train_clusters or 
            name in forced_val_clusters or 
            name in forced_test_clusters):
            continue

        train_gap = abs((train_count + size) - train_target)
        val_gap = abs((val_count + size) - val_target)
        test_gap = abs((test_count + size) - (total_seqs - train_target - val_target))

        set_gaps = {
            'train': train_gap if train_count < train_target else float('inf'),
            'val': val_gap if val_count < val_target else float('inf'),
            'test': test_gap if test_count < (total_seqs - train_target - val_target) else float('inf'),
        }
        target_set = min(set_gaps, key=set_gaps.get)

        if target_set == 'train':
            train_set.extend(seqs)
            train_count += size
        elif target_set == 'val':
            val_set.extend(seqs)
            val_count += size
        else:
            test_set.extend(seqs)
            test_count += size

    total = len(train_set) + len(val_set) + len(test_set)
    print(f"Total: {total}")
    print(f"Train: {len(train_set)} ({len(train_set)/total:.2%})")
    print(f"Val:   {len(val_set)} ({len(val_set)/total:.2%})")
    print(f"Test:  {len(test_set)} ({len(test_set)/total:.2%})")

    return train_set, val_set, test_set


class thermalFoldDataset(Dataset):
    def __init__(self, h5py_fname):
        self.h5py_fname = h5py_fname
        self.keys = None
        self._load_keys()
    
    def _load_keys(self):
        with h5py.File(self.h5py_fname, 'r') as h5f:
            self.keys = list(h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5py_fname, 'r') as h5f:
            data_void = h5f[key][()]
            data_bytes = bytes(data_void)
            data = pickle.loads(data_bytes)
        return data['Alphafold3Input']