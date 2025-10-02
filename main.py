import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from tqdm import tqdm
import torch
import json
import os
import dgl
from collections import defaultdict
import datetime
import numpy as np

from modules import utils
from modules.utils import prc_auc_score
from modules import model
from pathlib import Path
import torch.nn.functional as F
# 设置CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

from sklearn.metrics import auc
import time

# 参数解析
parser = argparse.ArgumentParser(description='UGAD')
parser.add_argument('--print_results', type=utils.str2bool, default=True)
parser.add_argument('--expr_name', type=str, default="None")
parser.add_argument('--dataset', type=str, default='YelpChi')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size_sampling', type=int, default=819200)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--std', type=str, default="false")
parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, e.g. --gpu 0")


args = parser.parse_args()

# 加载数据
graph, features, ano_label = utils.load_dataset(args.dataset)
if args.print_results:
    # 基础信息：节点数和边数
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    
    # 区分同质图和异质图，计算孤立节点
    if graph.is_homogeneous:
        # 同质图：直接计算所有节点的入度（无多边类型）
        in_degrees = graph.in_degrees()
        isolated_nodes = torch.sum(in_degrees == 0).item()
        isolated_info = f"\n  # isolated nodes = {isolated_nodes}"
    else:
        # 异质图：获取所有边类型并分别计算
        etypes = graph.canonical_etypes
        isolated_info = "\n  Isolated nodes per edge type:"
        # 统计每种边类型下的孤立节点
        for etype in etypes:
            # 计算当前边类型的入度（仅针对目标节点类型）
            in_deg = graph.in_degrees(etype=etype)
            isolated = torch.sum(in_deg == 0).item()
            # 边类型格式化显示（如 (user, follow, user) → "user->follow->user"）
            etype_str = f"{etype[0]}->{etype[1]}->{etype[2]}"
            isolated_info += f"\n    - {etype_str}: {isolated}"
        # 额外统计"所有边类型中均无入边"的节点（全局孤立节点）
        has_in_edge_list = []
        for etype in etypes:
            in_deg = graph.in_degrees(etype=etype)
            has_in_edge_list.append(in_deg > 0)  # 标记有入边的节点
        # 合并所有边类型的结果：全为False的节点是全局孤立节点
        if has_in_edge_list:
            has_any_in_edge = torch.stack(has_in_edge_list, dim=0).any(dim=0)
            global_isolated = torch.sum(~has_any_in_edge).item()
            isolated_info += f"\n  # globally isolated nodes (no in-edges in any type) = {global_isolated}"
    
    # 特征形状
    feature_shape = tuple(features.shape)
    
    # 输出所有信息
    print(
        f"{args.dataset}\n"
        f"  num_nodes = {num_nodes}, num_edges = {num_edges}"
        f"{isolated_info}\n"
        f"  feature shape: {feature_shape}"
    )

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

graph = graph.to(device)
features = features.to(device)

hidden_dim = args.hidden

# 处理图结构
graph = graph.remove_self_loop()
graph = graph.add_self_loop()
graph.ndata["label"] = torch.tensor(ano_label, dtype=torch.long, device=device)
graph.ndata["features"] = features

# 训练
epochs = args.epoch
# 早停相关参数
patience = args.patience  # 允许损失不下降的最大 epoch 数

all_best_results_list = []
all_results_list = []  # 每次 run 的完整 results_dict
utils.set_random_seeds(args.seed)

results = []  # 存储每次运行的详细结果

# 输出文件夹
os.makedirs("efficiency", exist_ok=True)
save_file = f"efficiency/{args.dataset}_bs{args.batch_size_sampling}.json"
num_runs = args.run
for run in range(num_runs):
    run_log = {"epoch_times": [], "peak_memory": []}

    print(f"\n==== Run {run+1}/{num_runs} ====")
    best_loss = float('inf')
    counter = 0  # 记录损失不下降的连续 epoch 数
    best_auprc = 0.0
    best_result = None
    results_loss_rec = []
    results_loss = []
    pbar = tqdm(total=epochs, desc=args.dataset)
    train_total_sec = 0
    # 每次运行用不同种子（如 args.seed + run_idx，确保种子不重复）
    # current_seed = args.seed + run_idx
    # utils.set_random_seeds(current_seed)
    # 批采样
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(graph.num_nodes()).to(device), sampler,
        batch_size=args.batch_size_sampling,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    # 初始化模型
    model_GAD = model.HUGE(features.shape[1], args.hidden, args.order).cuda()
    optimizer = torch.optim.Adam(model_GAD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_GAD.train()
    for epoch in range(epochs):
        start = time.time()

        # 清空 CUDA 内存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_loss = 0
        model_GAD.train()
        # print('*********************Train***************************')
        for input_nodes, output_nodes, blocks in dataloader:
            optimizer.zero_grad()
            block = blocks[-1]
            feat_src = block.srcdata['features'].to(device)
            feat_dst = block.dstdata['features'].to(device)
            dst_n = block.num_dst_nodes()
            dst_idx = torch.arange(block.num_dst_nodes())

            # 前向
            out = model_GAD.forward(blocks, feat_src)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(output_nodes)

        # 平均到节点
        
        epoch_loss /= graph.num_nodes()
        results_loss.append(epoch_loss)

        # 打印当前 epoch 的损失
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')
        end = time.time()
        elapsed = end - start
        # 记录时间
        run_log["epoch_times"].append(elapsed)
        # 记录显存
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_mem = 0
        run_log["peak_memory"].append(peak_mem)

        print(f"[Run {run+1}] Epoch {epoch+1} | time: {elapsed:.3f}s | peak mem: {peak_mem:.2f} MB")

        # 早停逻辑
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
                break
        
        pbar.update(1)
        
        num_t = 5  # 多个 t 值
        t_values = torch.linspace(50, 500, num_t, dtype=torch.long, device=graph.device)
        # 评估：使用视角差异作为异常分数
        time_eval_start = datetime.datetime.now()
        # 初始化三个列表来保存不同评分
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model_GAD.eval()
                # print('*********************Test***************************')

                N = graph.num_nodes()
                device = graph.device

                # —— 初始化所有异常评分缓存 —— 
                all_neighbor_sim = torch.zeros(N, device=device)
                all_neighbor_mse = torch.zeros(N, device=device)
                all_neighbor_sim_h = torch.zeros(N, device=device)
                all_neighbor_mse_h = torch.zeros(N, device=device)
                all_reconstruction_sim = torch.zeros(N, device=device)
                all_reconstruction_mse = torch.zeros(N, device=device)
                all_diffusion_sim = torch.zeros(N, device=device)
                all_diffusion_mse = torch.zeros(N, device=device)
                all_contrast_sim = torch.zeros(N, device=device)
                all_contrast_mse = torch.zeros(N, device=device)
                all_fused_score = torch.zeros(N, device=device)
                all_diffusion_multiscale = torch.zeros(N, device=device)
                # 存储每个 t 下的分数，用于后续求均值
                diffusion_mse_t = torch.zeros(N, num_t, device=device)
                diffusion_sim_t = torch.zeros(N, num_t, device=device)


                for input_nodes, output_nodes, blocks in dataloader:
                    assert (input_nodes >= 0).all() and (input_nodes < N).all(), "input_nodes index out of range!"
                    assert (output_nodes >= 0).all() and (output_nodes < N).all(), "output_nodes index out of range!"
                    block    = blocks[-1]
                    feat_src = block.srcdata['features'].to(device)
                    feat_dst = block.dstdata['features'].to(device)
                    dst_n    = block.num_dst_nodes()
                    dst_idx  = torch.arange(dst_n, device=device)
                    num_dst = block.num_dst_nodes()

                    # —— 前向传播 —— 
                    # —— 针对每个 t 计算异常分数 ——  
                    for i, t_val in enumerate(t_values):
                        t_fixed = torch.full((feat_src.shape[0],), t_val, device=feat_src.device)
                        out = model_GAD(blocks, feat_src, t_fixed)
                        reconstructed_feat = out['reconstructed_feat']
                        h_str = out['h_str']
                        h_att = out['h_att']
                        h_low = out['h_low']
                        noise = out['noise']
                        pred_noise = out['pred_noise']

                        mse, sim = node_metrics(noise, pred_noise)
                        diffusion_mse_t[output_nodes, i] = mse
                        diffusion_sim_t[output_nodes, i] = sim

                    all_reconstruction_mse[output_nodes], all_reconstruction_sim[output_nodes] = node_metrics(feat_dst, reconstructed_feat)
                    all_contrast_mse[output_nodes], all_contrast_sim[output_nodes] = node_metrics(h_str, h_att)

                # —— 对所有 t 的分数取均值作为最终异常分数 ——  
                all_diffusion_mse = diffusion_mse_t.mean(dim=1)
                all_diffusion_sim = diffusion_sim_t.mean(dim=1)

                # 如果不指定权重，就是均等加和
                all_fused_score = all_reconstruction_mse + all_contrast_mse + all_diffusion_mse

                # —— 报告函数 ——  
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
                    y   = _to_numpy(ano_label if labels is None else labels)

                    # 形状与有效性检查
                    if arr.shape[0] != y.shape[0]:
                        raise ValueError(f"[{name}] scores 与 labels 长度不一致: {arr.shape[0]} vs {y.shape[0]}")
                    if len(np.unique(y)) < 2:
                        print(f"{name:15s} 跳过：labels 只有一个类别，无法计算 AUC/AUPRC。")
                        output_dict[name] = {"auc_roc": None, "auprc": None}
                        return np.nan, np.nan

                    # 处理 NaN/Inf（可选）
                    if not np.all(np.isfinite(arr)):
                        finite = np.isfinite(arr)
                        if finite.any():
                            arr = np.nan_to_num(arr, nan=float(np.nanmean(arr[finite])),
                                                posinf=float(np.max(arr[finite])),
                                                neginf=float(np.min(arr[finite])))
                        else:
                            raise ValueError(f"[{name}] scores 全是 NaN/Inf。")

                    # 计算指标
                    mask = (y == 0) | (y == 1)   # 只保留标签为 0 或 1 的位置
                    auc  = roc_auc_score(y[mask], arr[mask])
                    aupr = average_precision_score(y[mask], arr[mask])

                    print(f"{name:15s} AUC-ROC: {auc:.4f} | AUPRC: {aupr:.4f}")

                    output_dict[name] = {
                        "auc_roc": round(float(auc), 4),
                        "auprc": round(float(aupr), 4)
                    }
                    return auc, aupr

                # —— 保存文件夹 ——  
                os.makedirs("results", exist_ok=True)
                results_dict = {}
                report("Rec", all_reconstruction_mse, results_dict)
                report("Ctr", all_contrast_mse, results_dict)
                report("Noise", all_diffusion_mse, results_dict)
                auc, auprc = report("Fused", all_fused_score, results_dict)

                if auprc > best_auprc:
                    best_auprc = auprc
                    best_result = {
                        "epoch": epoch,
                        "auc": auc,
                        "auprc": auprc
                    }

                print(f"\nBest AUPRC: {best_auprc:.4f} at Epoch {best_result['epoch']} "
                    f"(AUC: {best_result['auc']:.4f})")

        time_eval_end = datetime.datetime.now()
    
    pbar.close()
    all_best_results_list.append(best_result)
    all_results_list.append(results_dict)
    results.append(run_log)
# ====== 统计 ======
# all_times = [t for r in results for t in r["epoch_times"]]
# all_mems = [m for r in results for m in r["peak_memory"]]

# stats = {
#     "time_mean": float(np.mean(all_times)),
#     "time_var": float(np.var(all_times)),
#     "mem_mean": float(np.mean(all_mems)),
#     "mem_var": float(np.var(all_mems)),
# }

# output = {"stats": stats, "runs": results}

# 保存结果
# with open(save_file, "w") as f:
#     json.dump(output, f, indent=2)

# print(f"\nResults saved to {save_file}")
# print("Statistics:", stats)

best_auc_vals = [d["auc"] for d in all_best_results_list]
best_aupr_vals = [d["auprc"] for d in all_best_results_list]

best_summary = {
    "auc_mean": float(np.mean(best_auc_vals)),
    "auc_std": float(np.std(best_auc_vals)),
    "auprc_mean": float(np.mean(best_aupr_vals)),
    "auprc_std": float(np.std(best_aupr_vals)),
}

# --- 从 all_results_list 提取 ---
fused_auc_vals = [d["Fused"]["auc_roc"] for d in all_results_list]
fused_aupr_vals = [d["Fused"]["auprc"] for d in all_results_list]

fused_summary = {
    "auc_mean": float(np.mean(fused_auc_vals)),
    "auc_std": float(np.std(fused_auc_vals)),
    "auprc_mean": float(np.mean(fused_aupr_vals)),
    "auprc_std": float(np.std(fused_aupr_vals)),
}

# --- 打印 ---
print("📊 Best Summary:", best_summary)
print("📊 Fused Summary:", fused_summary)
if args.run ==10:
    # --- 保存 ---
    folder = f"results/{args.dataset}"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(
        folder,
        f"order{args.order}_hidden{args.hidden}_epoch{args.epoch}_patience{args.patience}.json"
    )

    save_dict = {
        "fused_summary": fused_summary,
        "best_summary": best_summary,
        "all_results": all_results_list,
        "all_best_results": all_best_results_list,
    }

    with open(filename, "w") as f:
        json.dump(save_dict, f, indent=4)

    print(f"✅ Saved results to {filename}")
# 等待 CUDA 完成所有操作
if torch.cuda.is_available():
    torch.cuda.synchronize()

# 打印信息
print(">>> Training finished", flush=True)

# 强制退出
import os
os._exit(0)
