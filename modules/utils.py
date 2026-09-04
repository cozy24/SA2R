import os
import argparse

from matplotlib.lines import Line2D
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
import dgl.function as fn
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

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

def feature_stats(x: torch.Tensor):
    stats = {
        "global_mean": x.mean().item(),
        "global_var": x.var(unbiased=False).item(),
        "global_min": x.min().item(),
        "global_max": x.max().item(),
        "nan": torch.isnan(x).any().item(),
        "inf": torch.isinf(x).any().item(),
        "zero_var_dims": (x.var(dim=0, unbiased=False) == 0).sum().item(),
    }
    return stats

def analyze_context_inconsistency_case_study_fast(
    graph,
    features,
    labels,
    dataset_name="dataset",
    anomaly_scores=None,
    normal_label=0,
    anomaly_label=1,
    top_ratio=0.05,
    plot=True,
    bins=40,
    save_dir="plot_case_study",
    save_name=None,
    dpi=300,
    eps=1e-12,
    make_bidirected=True,
    return_numpy=True
):
    """
    Scalable case study for first-order and second-order Context Inconsistency Score, CIS.

    This implementation is designed for large graphs.
    It avoids explicitly constructing A^2.

    First-order CIS:
        CIS_1(v) = 1 - mean_{u in N_1(v)} ((cos(x_v, x_u) + 1) / 2)

    Scalable second-order CIS:
        CIS_2(v) = 1 - mean similarity between node v and its path-weighted
        second-order context, computed by two rounds of message passing.

    The second-order context removes the self-return contribution v -> u -> v.
    It does not explicitly enumerate and deduplicate strict 2-hop neighbors,
    which is usually too expensive for million-edge graphs.
    """

    if features.dim() != 2:
        raise ValueError("features should have shape [num_nodes, feat_dim].")

    device = features.device
    num_nodes = graph.num_nodes()

    if features.shape[0] != num_nodes:
        raise ValueError(
            f"features.shape[0] = {features.shape[0]}, but graph.num_nodes() = {num_nodes}"
        )

    if labels.shape[0] != num_nodes:
        raise ValueError(
            f"labels.shape[0] = {labels.shape[0]}, but graph.num_nodes() = {num_nodes}"
        )

    labels = labels.to(device)

    if anomaly_scores is not None:
        if isinstance(anomaly_scores, np.ndarray):
            anomaly_scores = torch.from_numpy(anomaly_scores).float()
        anomaly_scores = anomaly_scores.to(device)

    # Treat graph as undirected.
    # If your graph has already been converted to bidirected in load_dataset,
    # set make_bidirected=False to save time and memory.
    if make_bidirected:
        graph = dgl.to_bidirected(graph, copy_ndata=False)

    graph = graph.to(device)

    # Normalize features once. Then dot product equals cosine similarity.
    x = F.normalize(features, p=2, dim=1, eps=eps)

    with torch.no_grad():
        with graph.local_scope():
            graph.ndata["x"] = x

            # ------------------------------------------------------------
            # 1-hop CIS: exact edge-wise average cosine similarity
            # ------------------------------------------------------------
            graph.apply_edges(fn.u_dot_v("x", "x", "edge_cos"))
            edge_cos = graph.edata["edge_cos"].squeeze(-1)
            edge_sim = (edge_cos + 1.0) / 2.0
            graph.edata["edge_sim"] = edge_sim

            graph.update_all(
                fn.copy_e("edge_sim", "m"),
                fn.mean("m", "sim_1hop")
            )

            sim_1hop = graph.ndata["sim_1hop"]
            degree = graph.in_degrees().to(device).float()

            isolated_mask = degree == 0
            sim_1hop[isolated_mask] = 1.0
            cis_1hop = 1.0 - sim_1hop

            # ------------------------------------------------------------
            # 2-hop CIS: scalable path-weighted second-order context
            # ------------------------------------------------------------
            # sum1[v] = sum_{u in N(v)} x_u
            graph.update_all(
                fn.copy_u("x", "m"),
                fn.sum("m", "sum1")
            )
            sum1 = graph.ndata["sum1"]

            # deg[v] = |N(v)|
            deg = degree.clamp(min=1.0)
            graph.ndata["sum1"] = sum1
            graph.ndata["deg"] = degree

            # sum2_raw[v] = sum_{u in N(v)} sum_{w in N(u)} x_w
            graph.update_all(
                fn.copy_u("sum1", "m"),
                fn.sum("m", "sum2_raw")
            )
            sum2_raw = graph.ndata["sum2_raw"]

            # count2_raw[v] = sum_{u in N(v)} deg(u)
            graph.update_all(
                fn.copy_u("deg", "m"),
                fn.sum("m", "count2_raw")
            )
            count2_raw = graph.ndata["count2_raw"]

            # Remove self-return paths v -> u -> v.
            # In a bidirected graph, each direct neighbor contributes one self-return.
            self_return_count = degree
            sum2 = sum2_raw - self_return_count.unsqueeze(1) * x
            count2 = count2_raw - self_return_count

            no_2hop_context = count2 <= 0

            count2_safe = count2.clamp(min=1.0)
            mean_2hop_context = sum2 / count2_safe.unsqueeze(1)

            # Since x is normalized, dot with averaged normalized features
            # is the average cosine similarity to the second-order context.
            sim_2hop = torch.sum(x * mean_2hop_context, dim=1)
            sim_2hop = (sim_2hop + 1.0) / 2.0

            # Nodes without valid 2-hop context are assigned similarity 1.
            sim_2hop[no_2hop_context] = 1.0
            cis_2hop = 1.0 - sim_2hop

    normal_mask = labels == normal_label
    anomaly_mask = labels == anomaly_label

    normal_indices = torch.where(normal_mask)[0]
    anomaly_indices = torch.where(anomaly_mask)[0]

    if len(normal_indices) == 0:
        raise ValueError("No normal nodes found. Please check normal_label.")
    if len(anomaly_indices) == 0:
        raise ValueError("No anomalous nodes found. Please check anomaly_label.")

    normal_cis_1hop_t = cis_1hop[normal_mask]
    anomaly_cis_1hop_t = cis_1hop[anomaly_mask]
    normal_cis_2hop_t = cis_2hop[normal_mask]
    anomaly_cis_2hop_t = cis_2hop[anomaly_mask]

    normal_mean_cis_1hop = float(normal_cis_1hop_t.mean().detach().cpu().item())
    anomaly_mean_cis_1hop = float(anomaly_cis_1hop_t.mean().detach().cpu().item())
    normal_mean_cis_2hop = float(normal_cis_2hop_t.mean().detach().cpu().item())
    anomaly_mean_cis_2hop = float(anomaly_cis_2hop_t.mean().detach().cpu().item())

    # Select representative case nodes.
    cis_sum = cis_1hop + cis_2hop

    if anomaly_scores is not None:
        normal_scores = anomaly_scores[normal_indices]
        selected_normal_node = normal_indices[torch.argmin(normal_scores)].item()

        anomaly_scores_sub = anomaly_scores[anomaly_indices]
        k = max(1, int(len(anomaly_indices) * top_ratio))
        top_pos = torch.topk(anomaly_scores_sub, k=k, largest=True).indices
        top_anomaly_indices = anomaly_indices[top_pos]

        top_cis_sum = cis_sum[top_anomaly_indices]
        selected_anomaly_node = top_anomaly_indices[torch.argmax(top_cis_sum)].item()
    else:
        selected_normal_node = normal_indices[torch.argmin(cis_sum[normal_indices])].item()
        selected_anomaly_node = anomaly_indices[torch.argmax(cis_sum[anomaly_indices])].item()

    def scalar(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    def int_scalar(x):
        if torch.is_tensor(x):
            return int(x.detach().cpu().item())
        return int(x)

    def node_summary(node_id):
        node_id = int(node_id)
        score = None
        if anomaly_scores is not None:
            score = scalar(anomaly_scores[node_id])

        return {
            "node_id": node_id,
            "label": int_scalar(labels[node_id]),
            "degree": int_scalar(graph.in_degrees(node_id)),
            "similarity_1hop": scalar(sim_1hop[node_id]),
            "similarity_2hop": scalar(sim_2hop[node_id]),
            "CIS_1hop": scalar(cis_1hop[node_id]),
            "CIS_2hop": scalar(cis_2hop[node_id]),
            "CIS_sum": scalar(cis_sum[node_id]),
            "anomaly_score": score
        }

    case_table = {
        "normal_case": node_summary(selected_normal_node),
        "anomaly_case": node_summary(selected_anomaly_node)
    }

    safe_dataset_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name))

    if save_name is None:
        save_name = f"{safe_dataset_name}_context_inconsistency_case_study_1hop_2hop.pdf"

    saved_path = None

    if plot:
        # ====================================================
        # Colors
        # ====================================================
        normal_color = "#F4A261"
        anomaly_color = "#8FB1E3"

        normal_line_color = "#C65D1E"
        anomaly_line_color = "#2F5C99"

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        saved_path = os.path.join(
            save_dir,
            save_name
        )

        # ====================================================
        # Convert tensors to NumPy arrays
        # ====================================================
        normal_cis_1hop_np = (
            normal_cis_1hop_t
            .detach()
            .cpu()
            .numpy()
        )

        anomaly_cis_1hop_np = (
            anomaly_cis_1hop_t
            .detach()
            .cpu()
            .numpy()
        )

        normal_cis_2hop_np = (
            normal_cis_2hop_t
            .detach()
            .cpu()
            .numpy()
        )

        anomaly_cis_2hop_np = (
            anomaly_cis_2hop_t
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # Figure
        # ====================================================
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8, 3.5)
        )

        # ====================================================
        # First-order CIS
        # ====================================================
        axes[0].hist(
            normal_cis_1hop_np,
            bins=bins,
            alpha=0.60,
            density=True,
            color=normal_color,
            edgecolor="none",
            label="Normal"
        )

        axes[0].hist(
            anomaly_cis_1hop_np,
            bins=bins,
            alpha=0.60,
            density=True,
            color=anomaly_color,
            edgecolor="none",
            label="Anomaly"
        )

        # Mean lines remain visible without legend entries
        axes[0].axvline(
            normal_mean_cis_1hop,
            linestyle="--",
            linewidth=2.0,
            color=normal_line_color,
            zorder=4
        )

        axes[0].axvline(
            anomaly_mean_cis_1hop,
            linestyle="--",
            linewidth=2.0,
            color=anomaly_line_color,
            zorder=4
        )

        axes[0].set_xlabel(
            "First-order CIS",
            fontsize=14,
            labelpad=4
        )

        axes[0].set_ylabel(
            "Density",
            fontsize=14,
            labelpad=4
        )

        # ====================================================
        # Second-order CIS
        # ====================================================
        axes[1].hist(
            normal_cis_2hop_np,
            bins=bins,
            alpha=0.60,
            density=True,
            color=normal_color,
            edgecolor="none",
            label="Normal"
        )

        axes[1].hist(
            anomaly_cis_2hop_np,
            bins=bins,
            alpha=0.60,
            density=True,
            color=anomaly_color,
            edgecolor="none",
            label="Anomaly"
        )

        # Mean lines remain visible without legend entries
        axes[1].axvline(
            normal_mean_cis_2hop,
            linestyle="--",
            linewidth=2.0,
            color=normal_line_color,
            zorder=4
        )

        axes[1].axvline(
            anomaly_mean_cis_2hop,
            linestyle="--",
            linewidth=2.0,
            color=anomaly_line_color,
            zorder=4
        )

        axes[1].set_xlabel(
            "Second-order CIS",
            fontsize=14,
            labelpad=4
        )

        axes[1].set_ylabel(
            "Density",
            fontsize=14,
            labelpad=4
        )

        # ====================================================
        # Axis style
        # ====================================================
        for ax in axes:
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=12,
                direction="in",
                top=True,
                right=True,
                width=0.8,
                length=3.5
            )

            ax.grid(
                True,
                which="major",
                linestyle=":",
                linewidth=0.6,
                color="0.85",
                alpha=0.80
            )

            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            # ============================================================
            # Shared legend with mean lines
            # ============================================================
            legend_handles = [
                Patch(
                    facecolor=normal_color,
                    edgecolor="none",
                    alpha=0.60,
                    label="Normal"
                ),

                Patch(
                    facecolor=anomaly_color,
                    edgecolor="none",
                    alpha=0.60,
                    label="Anomaly"
                ),

                Line2D(
                    [0],
                    [0],
                    color=normal_line_color,
                    linestyle="--",
                    linewidth=2.0,
                    label="Normal mean"
                ),

                Line2D(
                    [0],
                    [0],
                    color=anomaly_line_color,
                    linestyle="--",
                    linewidth=2.0,
                    label="Anomaly mean"
                ),
            ]

            legend = fig.legend(
                handles=legend_handles,

                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),

                ncol=4,

                frameon=True,
                fancybox=True,
                shadow=False,

                facecolor="white",
                edgecolor="0.78",
                framealpha=0.90,

                fontsize=12,

                handlelength=1.50,
                handletextpad=0.40,
                columnspacing=1.10,
                labelspacing=0.25,

                borderpad=0.35,
                borderaxespad=0.0
            )

            legend_frame = legend.get_frame()

            legend_frame.set_linewidth(0.45)

            legend_frame.set_boxstyle(
                "round,pad=0.18,rounding_size=0.45"
            )

        # ====================================================
        # Layout
        # ====================================================
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.16,
            top=0.80,
            wspace=0.25
        )

        # ====================================================
        # Save
        # ====================================================
        fig.savefig(
            saved_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.06
        )

        plt.close(fig)

    result = {
        "normal_mean_cis_1hop": normal_mean_cis_1hop,
        "anomaly_mean_cis_1hop": anomaly_mean_cis_1hop,
        "normal_mean_cis_2hop": normal_mean_cis_2hop,
        "anomaly_mean_cis_2hop": anomaly_mean_cis_2hop,
        "selected_normal_node": selected_normal_node,
        "selected_anomaly_node": selected_anomaly_node,
        "case_table": case_table,
        "saved_path": saved_path,
        "num_nodes": graph.num_nodes(),
        "num_edges": graph.num_edges()
    }

    if return_numpy:
        result.update({
            "node_cis_1hop": cis_1hop.detach().cpu().numpy(),
            "node_cis_2hop": cis_2hop.detach().cpu().numpy(),
            "node_similarity_1hop": sim_1hop.detach().cpu().numpy(),
            "node_similarity_2hop": sim_2hop.detach().cpu().numpy(),
            "normal_cis_1hop": normal_cis_1hop_t.detach().cpu().numpy(),
            "anomaly_cis_1hop": anomaly_cis_1hop_t.detach().cpu().numpy(),
            "normal_cis_2hop": normal_cis_2hop_t.detach().cpu().numpy(),
            "anomaly_cis_2hop": anomaly_cis_2hop_t.detach().cpu().numpy(),
        })

    return result

def load_dataset(dataset_name, normalize=False, to_bidirected=False):
    if dataset_name in ['weibo', 'AmazonFull', 'YelpChiFull', 'elliptic', 'dgraphfin', 'tsocial', 'tfinance']:
        graph_list, _ = load_graphs(f"datasets/{dataset_name}")
        graph = graph_list[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    elif dataset_name == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    elif dataset_name in ['amazon']:
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        features = torch.FloatTensor(graph.ndata['feature'])
        labels = graph.ndata['label']
    else:
        adj, features_np, labels_np, _, _ = load_mat(dataset_name)
        features = torch.FloatTensor(features_np.toarray())
        labels = torch.LongTensor(labels_np)
        graph = dgl.from_scipy(adj)
        graph.ndata['feature'] = features
        graph.ndata['label'] = labels

        # 保存
        # dgl.save_graphs("amazon", [graph])

    print(graph)
    case_result = analyze_context_inconsistency_case_study_fast(
        graph=graph,
        features=features,
        labels=labels,
        dataset_name=dataset_name,
        anomaly_scores=None,
        normal_label=0,
        anomaly_label=1,
        plot=True,
        save_dir="plot_case_study",
        make_bidirected=False,
        return_numpy=True
    )

    print("图片保存路径:", case_result["saved_path"])
    print(case_result["case_table"])
    
    print(feature_stats(features))
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
