import sys

sys.path.append('/')

import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import esm
from transformers import AutoTokenizer, RobertaModel
from utils import *  # 确保包含 label_sequence, label_smiles, smile_to_graph, sequence_to_graph 等

# Convert the sets to DataFrame and save as CSV (without column names)
def save_to_csv(entries, filename):
    # The entries contain [Drug_SMILES, Protein_Seq, Interaction_Value]
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, header=False)

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()

    mol_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))

    edges = []
    bond_type_np = np.zeros((mol_size, mol_size))
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_type_np[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondTypeAsDouble()
        bond_type_np[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondTypeAsDouble()
        # bond_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    # print('@@@@@@@@@@@@@@@@@')
    # print(np.array(edges).shape,'edges')
    # print(np.array(g).shape,'g')

    mol_adj = np.zeros((mol_size, mol_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    # print(np.array(mol_adj).shape,'mol_adj')
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    bond_edge_index = []
    bond_type = []
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        bond_edge_index.append([i, j])
        bond_type.append(bond_type_np[i, j])
    # print(bond_edge_index)
    # print('smile_to_graph')
    # print('mol_features',np.array(mol_features).shape)
    # print('bond_edge_index',np.array(bond_edge_index).shape)
    # print('bond_type',np.array(bond_type).shape)
    return mol_size, mol_features, bond_edge_index, bond_type
    # return mol_size, mol_features, bond_edge_index

# target sequence to target graph
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    # print('***',(os.path.abspath(os.path.join(distance_dir, target_key + '.npy'))))
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    # distance_map = np.load(contact_map_file)
    distance_map = np.load(contact_map_file, allow_pickle=True)
    rows, cols = distance_map.shape
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        # distance_map[i, i] = 1
        # if i + 1 < target_size:
        #     distance_map[i, i + 1] = 1
        # 只要不是最后一行，就连接下一个
        if i < rows - 1:
            distance_map[i, i + 1] = 1
            distance_map[i + 1, i] = 1
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    # print(len(index_row))
    # print(len(index_col))
    # print(len(index_row_))
    # print(len(index_col_))
    # print(distance_map.shape)
    # print((len(index_row) * 1.0) / (distance_map.shape[0] * distance_map.shape[1]))
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature(target_sequence)
    # residue_distance = np.array(target_edge_distance)  # consistent with edge
    # print('target_feature', target_feature.shape)
    # print(target_edge_index)
    # print('target_edge_index', np.array(target_edge_index).shape)
    # print('residue_distance', residue_distance.shape)
    # return target_size, target_feature, residue_edge_index, residue_distance
    return target_size, target_feature, target_edge_index, target_edge_distance


# data write to csv file
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)




# --- 核心划分函数 ---

def cv2(data: pd.DataFrame):
    """药物冷启动：测试集药物在训练集中从未出现"""
    data_size = len(data)
    train_raw, test_raw = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]
    protein_list = train_raw["protein"].unique()
    test_filtered = test_raw[test_raw['protein'].isin(protein_list)].copy()
    test_drugs = test_filtered['drug'].unique()
    train_filtered = train_raw[~train_raw['drug'].isin(test_drugs)].copy()
    split_idx = int(len(test_filtered) * 1 / 3)
    return train_filtered, test_filtered[:split_idx], test_filtered[split_idx:]


def cv3(data: pd.DataFrame):
    """蛋白质冷启动：测试集蛋白质在训练集中从未出现"""
    data_size = len(data)
    train_raw, test_raw = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]
    drug_list = train_raw["drug"].unique()
    test_filtered = test_raw[test_raw['drug'].isin(drug_list)].copy()
    test_prots = test_filtered['protein'].unique()
    train_filtered = train_raw[~train_raw['protein'].isin(test_prots)].copy()
    split_idx = int(len(test_filtered) * 1 / 3)
    return train_filtered, test_filtered[:split_idx], test_filtered[split_idx:]


def cv4(data: pd.DataFrame):
    """双冷启动：药物和蛋白质均不相关"""
    data_size = len(data)
    train_raw, test_raw = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]
    test_drugs = test_raw['drug'].unique()
    test_prots = test_raw['protein'].unique()
    train_filtered = train_raw[
        (~train_raw['drug'].isin(test_drugs)) & (~train_raw['protein'].isin(test_prots))
        ].copy()
    split_idx = int(len(test_raw) * 1 / 3)
    return train_filtered, test_raw[:split_idx], test_raw[split_idx:]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ChemBERTa
chem_path = "/home/severus/code/three/DDGR-DTI-main/DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(chem_path)
model_chem = RobertaModel.from_pretrained(chem_path).to(device).eval()
# ESM-1b
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
esm_model = esm_model.to(device).eval()
batch_converter = alphabet.get_batch_converter()


def get_embeddings(smile):
    encodings = tokenizer(smile, return_tensors='pt', padding="max_length", max_length=290, truncation=True).to(device)
    with torch.no_grad():
        output = model_chem(**encodings)
        return output.last_hidden_state[0, 0, :].cpu().numpy()


def Get_Protein_Feature(sequence):
    sequence = sequence[:1000]
    _, _, batch_tokens = batch_converter([("prot", sequence)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33])
    return results["representations"][33][0, 1:len(sequence) + 1].mean(0).cpu().numpy()



def append_degree_bias_labels(train_df, dev_df, test_df, percentile=50):
    """
    仅根据训练集计算节点热度阈值，并将偏置标签打在所有数据集上。
    """

    pos_train = train_df[train_df['interaction'] == 1.0] if 1.0 in train_df['interaction'].values else train_df

    drug_degrees = pos_train['drug'].value_counts().to_dict()
    prot_degrees = pos_train['protein'].value_counts().to_dict()

    d_deg_values = list(drug_degrees.values())
    t_deg_values = list(prot_degrees.values())
    d_threshold = np.percentile(d_deg_values, percentile) if d_deg_values else 0
    t_threshold = np.percentile(t_deg_values, percentile) if t_deg_values else 0

    print(f"[*] Causal Bias Threshold - Drug: >{d_threshold:.1f} | Protein: >{t_threshold:.1f}")

    def apply_bias(df):
        bias_labels = []
        for _, row in df.iterrows():
            d_deg = drug_degrees.get(row['drug'], 0)
            t_deg = prot_degrees.get(row['protein'], 0)

            if d_deg > d_threshold or t_deg > t_threshold:
                bias_labels.append(1.0)
            else:
                bias_labels.append(0.0)
        df['bias_label'] = bias_labels
        return df

    return apply_bias(train_df.copy()), apply_bias(dev_df.copy()), apply_bias(test_df.copy())


def create_CPI_dataset2(dataset='BindingDB', cv_type="cv1",
                       base_dir='/home/severus/code/three/DDGR-DTI-main/data'):
    dataset = "DrugBank"  # 强制覆盖
    data_file = os.path.join(base_dir, f'{dataset}.txt')
    proteins_file = os.path.join(base_dir, f'proteins_{dataset}.pkl')

    if not os.path.exists(proteins_file):
        proteins = {}
        seq_list = []
        count = 1
        for line in open(data_file, 'r'):
            arrs = line.strip().split(' ' if ' ' in line else '\t')
            if len(arrs) < 3: break
            if arrs[1] not in seq_list:
                key = f'prot{count}'
                proteins[key] = arrs[1]
                seq_list.append(arrs[1])
                count += 1
        with open(proteins_file, 'wb') as f:
            pickle.dump(proteins, f)
    else:
        with open(proteins_file, 'rb') as f:
            proteins = pickle.load(f)

    proteins_rev = {v: k for k, v in proteins.items()}

    all_entries = []
    for line in open(data_file, 'r'):
        arrs = line.strip().split(' ' if ' ' in line else '\t')
        if len(arrs) < 3: break
        all_entries.append([arrs[0], arrs[1], float(arrs[2])])

    df_all = pd.DataFrame(all_entries, columns=["drug", "protein", "interaction"])
    df_all = df_all.sample(frac=1, random_state=3407).reset_index(drop=True)

    # 3. 执行 CV 划分
    print(f"Split mode: {cv_type}")
    if cv_type == "cv1":
        block = len(df_all) // 10
        train_set = df_all[:block * 7].copy()
        rest = df_all[block * 7:].copy()
        dev_set, test_set = rest[:len(rest) // 3].copy(), rest[len(rest) // 3:].copy()
    elif cv_type == "cv2":
        train_set, dev_set, test_set = cv2(df_all)
    elif cv_type == "cv3":
        train_set, dev_set, test_set = cv3(df_all)
    elif cv_type == "cv4":
        train_set, dev_set, test_set = cv4(df_all)
    else:
        raise ValueError(f"未知的划分模式: {cv_type}")

    # train_set, dev_set, test_set = append_degree_bias_labels(train_set, dev_set, test_set, percentile=75)

    drug_smiles = df_all['drug'].unique()
    pro_keys = list(proteins.keys())


    fcfp_feature = {s: get_embeddings(s) for s in tqdm(drug_smiles, desc="Drug FCFP Embedding")}
    esm_feature = {k: Get_Protein_Feature(proteins[k]) for k in tqdm(pro_keys, desc="Protein ESM Embedding")}


    def build_dataset(df):
        xd_list = df["drug"].tolist()
        prot_list = df["protein"].tolist()
        y_list = df["interaction"].astype(float).tolist()

        new_xd, new_target, new_y = [], [], []
        for drug, prot, y in zip(xd_list, prot_list, y_list):
            seq = str(prot).strip()
            new_xd.append(drug)
            new_target.append(proteins_rev[seq])
            new_y.append(y)

        dataset = DTADataset(xd=new_xd, target_key=new_target, y=new_y,
                             esm=esm_feature, fcfp=fcfp_feature)

        dataset.raw_smiles = new_xd
        dataset.raw_proteins = new_target

        return dataset

    return build_dataset(train_set), build_dataset(dev_set), build_dataset(test_set)

