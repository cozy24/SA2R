import os
import argparse
import torch
import scipy.sparse as sp
import dgl
from dgl.data.utils import load_graphs
import random
import numpy as np
import scipy.io as sio
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from sklearn.metrics import auc, precision_recall_curve
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

def to_rank_score(s):
    return torch.argsort(torch.argsort(s)).float() / (s.numel() - 1)
    
def _to_numpy(x):
    """支持 torch.Tensor / np.ndarray / list，统一转为 1D numpy"""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().ravel()
    return np.asarray(x, dtype=float).ravel()

def report(name, scores, output_dict, labels=None):
    """
    name: 指标名
    scores: 节点分数（tensor or numpy）
    output_dict: 用于保存结果的 dict
    labels: 可选，标签（默认用外部的 ano_label）
    """
    arr = _to_numpy(scores)
    y   = _to_numpy(labels)

    # 过滤掉非 0/1 的标签（如 -1 表示无标签）
    mask = np.isin(y, [0, 1])
    arr, y = arr[mask], y[mask]

    # 如果过滤后没剩下数据
    if arr.shape[0] == 0:
        print(f"{name:15s} 跳过：没有有效的 0/1 标签节点。")
        output_dict[name] = {"auroc": None, "auprc": None}
        return np.nan, np.nan

    # 形状与有效性检查
    if len(np.unique(y)) < 2:
        print(f"{name:15s} 跳过：labels 只有一个类别，无法计算 AUROC/AUPRC。")
        output_dict[name] = {"auroc": None, "auprc": None}
        return np.nan, np.nan

    # 处理 NaN/Inf
    if not np.all(np.isfinite(arr)):
        finite = np.isfinite(arr)
        if finite.any():
            arr = np.nan_to_num(arr, nan=float(np.nanmean(arr[finite])),
                                posinf=float(np.max(arr[finite])),
                                neginf=float(np.min(arr[finite])))
        else:
            raise ValueError(f"[{name}] scores 全是 NaN/Inf。")

    auroc  = roc_auc_score(y, arr)
    auprc = average_precision_score(y, arr)
    # print(f"{name:15s} AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    output_dict[name] = {
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4)
    }
    return auroc, auprc

def node_metrics(feat_dst, reconstructed_feat):
    node_mse = torch.mean((feat_dst - reconstructed_feat) ** 2, dim=1)  # [N]
    feat_norm = F.normalize(feat_dst, p=2, dim=1)
    recon_norm = F.normalize(reconstructed_feat, p=2, dim=1)
    node_sim = 1 - torch.sum(feat_norm * recon_norm, dim=1)  # [N]
    return node_mse, node_sim


def compute_summary(results):
    aurocs        = [r["auroc"] for r in results]
    auprcs       = [r["auprc"] for r in results]
    train_times = [r["train_time"] for r in results]

    return {
        "average_auroc": sum(aurocs) / len(aurocs),
        "average_auprc": sum(auprcs) / len(auprcs),
        "average_train_time_sec": sum(train_times) / len(train_times),
    }
    
class MultiHotEmbed(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        # 如果是 numpy array，转 torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        # 如果是 tensor，确保类型 float
        elif isinstance(x, torch.Tensor):
            x = x.float()
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")
        
        # 输出连续向量
        return self.proj(x).detach().cpu().numpy()


cos_dist = lambda *args, **kwargs: -1 * torch.nn.functional.cosine_similarity(*args, **kwargs)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def prc_auroc_score(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    auroc_precision_recall = auc(recall, precision)
    return auroc_precision_recall


def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label, dtype=np.int64))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def preprocess_features(features, dataset_name):
    if dataset_name in ['Amazon', 'AmazonFull', 'amazon', 'tfinance', 'elliptic', 'dgraphfin']:
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return torch.tensor(features.astype(np.float32))
    return features


def print_feature_stats(features: torch.Tensor, labels: torch.Tensor):
    total_nodes = labels.shape[0]
    
    is_normal = (labels == 0)
    is_ano = (labels == 1)
    is_background = ~is_normal & ~is_ano  
    
    # 计数
    num_normal = is_normal.sum().item()
    num_ano = is_ano.sum().item()
    num_background = is_background.sum().item()
    num_valid = num_normal + num_ano  
    
    # 打印节点统计
    print("==== 节点统计 ====")
    print(f"总节点数: {total_nodes}")
    print(f"背景节点: {num_background} ({num_background/total_nodes:.2%})")
    print(f"正常节点: {num_normal}")
    print(f"  - 占整体比例: {num_normal/total_nodes:.2%}")
    print(f"  - 占有效样本(0/1)比例: {num_normal/num_valid:.2%}" if num_valid > 0 else "  - 无有效样本")
    print(f"异常节点: {num_ano}")
    print(f"  - 占整体比例: {num_ano/total_nodes:.2%}")
    print(f"  - 占有效样本(0/1)比例: {num_ano/num_valid:.2%}" if num_valid > 0 else "  - 无有效样本")
    print(f"有效样本(0/1类)总数: {num_valid} ({num_valid/total_nodes:.2%})")
    print()

def load_dataset(dataset_name, normalize=False, to_bidirected=False):
    if dataset_name in ['AmazonFull', 'YelpChiFull', 'elliptic', 'dgraphfin', 'tsocial', 'tfinance']:
        graph_list, _ = load_graphs(f"datasets/{dataset_name}")
        graph = graph_list[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    elif dataset_name == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    elif dataset_name == 'amazon':
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    else:
        adj, features_np, labels_np, _, _ = load_mat(dataset_name)
        features = torch.FloatTensor(features_np.toarray())
        labels = torch.LongTensor(labels_np)
        graph = dgl.from_scipy(adj)

    print(graph)

    print_feature_stats(features, labels)
    features = preprocess_features(features, dataset_name)

    return graph, features, labels

def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
