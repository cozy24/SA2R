import os
import argparse
from tqdm import tqdm
import torch
import time
import os
import torchmetrics
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl.data.utils import load_graphs
import random
import json
import numpy as np
import scipy.io as sio
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import DGLDataset, FraudYelpDataset, FraudAmazonDataset, CoraGraphDataset, RedditDataset
from sklearn.metrics import auc, precision_recall_curve


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
def prc_auc_score(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall

def halo(x_u, x_v, eps=1e-6):
    dist = (x_u - x_v).pow(2).sum(1).sqrt()/((x_u.pow(2).sum(1) + x_v.pow(2).sum(1)+eps).sqrt())
    return dist
def print_feature_stats(features: torch.Tensor, labels: torch.Tensor):
    """
    打印正常节点、异常节点以及全体节点的特征统计量 (mean, std, min, max)
    并输出占比（同时计算相对于整体和仅正常/异常节点的比例）
    """
    total_nodes = labels.shape[0]
    
    # 筛选正常节点(0)、异常节点(1)和背景节点(其他标签)
    is_normal = (labels == 0)
    is_ano = (labels == 1)
    is_background = ~is_normal & ~is_ano  # 标签不为0且不为1的节点
    
    # 计数
    num_normal = is_normal.sum().item()
    num_ano = is_ano.sum().item()
    num_background = is_background.sum().item()
    num_valid = num_normal + num_ano  # 仅正常和异常节点（有效样本）
    
    # 全体节点特征统计
    all_mean = features.mean(dim=0).mean().item()
    all_std = features.std(dim=0).mean().item()
    all_min = features.min().item()
    all_max = features.max().item()
    
    # 正常节点特征统计 (label=0)
    normal_feat = features[is_normal]
    normal_mean = normal_feat.mean(dim=0).mean().item() if num_normal > 0 else 0.0
    normal_std = normal_feat.std(dim=0).mean().item() if num_normal > 0 else 0.0
    normal_min = normal_feat.min().item() if num_normal > 0 else 0.0
    normal_max = normal_feat.max().item() if num_normal > 0 else 0.0
    
    # 异常节点特征统计 (label=1)
    ano_feat = features[is_ano]
    ano_mean = ano_feat.mean(dim=0).mean().item() if num_ano > 0 else 0.0
    ano_std = ano_feat.std(dim=0).mean().item() if num_ano > 0 else 0.0
    ano_min = ano_feat.min().item() if num_ano > 0 else 0.0
    ano_max = ano_feat.max().item() if num_ano > 0 else 0.0
    
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
    
    # 打印特征统计
    print("==== 特征统计量对比 ====")
    print(f"全体节点 -> mean: {all_mean:.6f}, std: {all_std:.6f}, min: {all_min:.6f}, max: {all_max:.6f}")
    print(f"正常节点 -> mean: {normal_mean:.6f}, std: {normal_std:.6f}, min: {normal_min:.6f}, max: {normal_max:.6f}")
    print(f"异常节点 -> mean: {ano_mean:.6f}, std: {ano_std:.6f}, min: {ano_min:.6f}, max: {ano_max:.6f}")
    print("=======================")

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

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

def load_dataset(dataset_name):
    raw_name = dataset_name
    if dataset_name in ['ACM', 'Amazon','Facebook','Reddit','YelpChi', 'cora', 'weibo', 't_finance', 'citeseer', 'questions', 'Flickr', 'tolokers', 'BlogCatalog']:
        adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset_name)
        features = features.todense()
        # nx_graph = nx.from_scipy_sparse_matrix(adj)
        nx_graph = nx.from_scipy_sparse_array(adj)
        graph = dgl.from_networkx(nx_graph)
    elif dataset_name in ['AmazonFull', 'YelpChiFull', 'elliptic', 'tfinance', 'dgraphfin']:
        graph = load_graphs(f"datasets/{dataset_name}")[0][0]
        ano_label = graph.ndata["label"].cpu().detach().numpy()
        features = graph.ndata["feature"]
    elif dataset_name in ['amazon']:
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        ano_label = graph.ndata["label"].cpu().detach().numpy()
        features = graph.ndata["feature"]
    elif dataset_name in ['yelp']:
        dataset = FraudYelpDataset()
        graph = dataset[0]
        ano_label = graph.ndata["label"].cpu().detach().numpy()
        features = graph.ndata["feature"]
    # elif dataset_name in ['dgraph']:
    #     # 1. 加载NPZ文件
    #     data = np.load('datasets/dgraphfin.npz', allow_pickle=True)
    #     print(f"成功加载数据集，包含键: {data.files}")
        
    #     # 2. 提取核心数据
    #     x = data["x"]                  # 节点特征 (num_nodes, 17)
    #     y = data["y"]                  # 节点标签 (num_nodes,)
    #     edge_index = data["edge_index"]# 边索引 (num_edges, 2)
    #     edge_type = data["edge_type"]  # 边类型 (num_edges,)
    #     edge_timestamp = data["edge_timestamp"]  # 边时间戳 (num_edges,)
    #     train_mask = data["train_mask"]          # 训练掩码 (num_nodes,)
    #     valid_mask = data["valid_mask"]          # 验证掩码 (num_nodes,)
    #     test_mask = data["test_mask"]            # 测试掩码 (num_nodes,)
        
    #     # 3. 数据基本验证
    #     num_nodes = x.shape[0]
    #     num_edges = edge_index.shape[0]
    #     print(f"节点数: {num_nodes}, 边数: {num_edges}")
    #     print(f"节点特征维度: {x.shape[1]}")
    #     print(f"边类型数量: {len(np.unique(edge_type))} (预期11种)")
    #     print(f"标签分布: {np.bincount(y)} (0:正常, 1:欺诈, 2/3:背景)")
    #     print(f"训练/验证/测试节点数: {train_mask.sum()}/{valid_mask.sum()}/{test_mask.sum()}")
        
    #     # 4. 转换为DGL图
    #     # DGL要求边索引格式为(源节点列表, 目标节点列表)
    #     u, v = edge_index[:, 0], edge_index[:, 1]
    #     g = dgl.graph((u, v), num_nodes=num_nodes)  # 显式指定节点数确保完整性
        
    #     # 5. 添加节点特征
    #     g.ndata["feature"] = torch.tensor(x, dtype=torch.float32)  # 节点特征
    #     g.ndata["label"] = torch.tensor(y, dtype=torch.long)    # 节点标签
        
    #     print("\nDGL图构建完成:")
    #     dgl.save_graphs("datasets/dgraphfin", [g])

    #     print("DGL图已成功保存为 dgraphfin")
    #     print(g)

    else:
        raise Exception(f"Unimplemented dataset: {dataset_name}")

    if dataset_name in ['tfinance', 'amazon', 'Amazon', 'AmazonFull', 'dgraphfin', 'elliptic']:
        features = torch.tensor(preprocess_features(features))
    else:
        features = torch.tensor(features, dtype=torch.float32)
    # if dataset_name in ["elliptic"]:
    #     graph = dgl.remove_self_loop(graph)
    #     graph = dgl.add_self_loop(graph)

    features = features.to(torch.float32)
    print(graph)
    print(print_feature_stats(features, ano_label))
    return graph.long(), features, ano_label

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