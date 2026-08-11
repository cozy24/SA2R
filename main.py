import warnings
warnings.filterwarnings('ignore')

import argparse
from tqdm import tqdm
import torch
import os
import dgl
import json
import io
import contextlib
import numpy as np
from modules import utils
from modules.utils import *
from modules import model
import time

parser = argparse.ArgumentParser(description='SA2R')
parser.add_argument('--print_results', type=utils.str2bool, default=True)
parser.add_argument('--expr_name', type=str, default="None")
parser.add_argument('--dataset', type=str, default='Amazon')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--std', type=int, default=1)
parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, e.g. --gpu 0")
parser.add_argument("--order", type=int, default=2)
parser.add_argument("--run", type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--w1', type=float, default=1.0)
parser.add_argument('--w2', type=float, default=1.0)
parser.add_argument('--w3', type=float, default=1.0)

args = parser.parse_args()
utils.set_random_seeds(args.seed)

graph, features, ano_label = utils.load_dataset(args.dataset, normalize=args.std, to_bidirected=True)

graph.ndata['features'] = features 
graph.ndata['label'] = ano_label

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

hidden_dim = args.hidden
if len(ano_label) > 1e6:
    batch_size = 8192 * 16
else:
    batch_size = len(ano_label)

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
dataloader = dgl.dataloading.DataLoader(
    graph, torch.arange(graph.num_nodes()), sampler,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0)
num_runs = args.run
all_test_results = []

for run in range(num_runs):
    
    model_GAD = model.Model(graph.ndata['features'].shape[1], args.hidden, args.order, label=graph.ndata["label"], w1=args.w1, w2=args.w2, w3=args.w3).to(device)
    optimizer = torch.optim.Adam(model_GAD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_times = []

    torch.cuda.reset_peak_memory_stats(device)
 
    epochs = args.epoch
    results_loss_rec = []
    results_loss = []
    pbar = tqdm(total=epochs, desc=args.dataset, disable=True)
    train_total_sec = 0

    patience = args.patience 
    best_loss = float('inf')
    counter = 0 
    run_test_results = []
    for epoch in range(epochs):
        t_start = time.time()
        epoch_loss = 0
        model_GAD.train()
        for input_nodes, output_nodes, blocks in dataloader:
            optimizer.zero_grad()
            blocks = [b.to(device) for b in blocks]  
            block = blocks[-1]
            feat_src = block.srcdata['features'].to(device)
            feat_dst = block.dstdata['features'].to(device)
            dst_n = block.num_dst_nodes()
            dst_idx = torch.arange(block.num_dst_nodes())

            out = model_GAD.forward(blocks, feat_src)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(output_nodes)
        t_end = time.time()
        epoch_times.append(t_end - t_start)
        epoch_loss /= graph.num_nodes()
        results_loss.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        peak_train_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        pbar.update(1)
        
        if (epoch + 1) % 10 == 0:
            
            torch.cuda.reset_peak_memory_stats(device)
            
            
            with torch.no_grad():
                start = time.time()
                num_t = 5  
                t_values = torch.linspace(200, 300, num_t, dtype=torch.long, device=device)
                model_GAD.eval()

                N = graph.num_nodes()

                all_reconstruction_mse = torch.zeros(N, device=device)
                all_noise_mse = torch.zeros(N, device=device)
                all_contrast_mse = torch.zeros(N, device=device)
                all_fused_score = torch.zeros(N, device=device)
                noise_mse_t = torch.zeros(N, num_t, device=device)
                h_str_full = torch.zeros(N, args.hidden, device=device)
                h_att_full = torch.zeros(N, args.hidden, device=device)
                feat_full = torch.zeros(N, feat_dst.shape[1], device=device)
                rec_feat_full = torch.zeros(N, feat_dst.shape[1], device=device)
                noise_full = torch.zeros(N, feat_dst.shape[1], num_t, device=device)
                pred_noise_full = torch.zeros(N, feat_dst.shape[1], num_t, device=device)

                for input_nodes, output_nodes, blocks in dataloader:
                    blocks = [b.to(device) for b in blocks]  
                    block = blocks[-1]
                    feat_src = block.srcdata['features'].to(device)
                    feat_dst = block.dstdata['features'].to(device)
                    dst_n = block.num_dst_nodes()
                    dst_idx = torch.arange(block.num_dst_nodes())
                    num_dst = block.num_dst_nodes()

                    for i, t_val in enumerate(t_values):
                        t_fixed = torch.full((feat_src.shape[0],), t_val, device=feat_src.device)
                        out = model_GAD(blocks, feat_src, t_fixed)
                        reconstructed_feat = out['reconstructed_feat']
                        h_str = out['h_str']
                        h_att = out['h_att']
                        noise = out['noise']
                        pred_noise = out['pred_noise']
                        mse, _= node_metrics(noise[dst_idx], pred_noise[dst_idx])
                        noise_mse_t[output_nodes, i] = mse
                        noise_full[output_nodes, :, i] = noise[dst_idx]
                        pred_noise_full[output_nodes, :, i] = pred_noise[dst_idx]


                    all_reconstruction_mse[output_nodes], _ = node_metrics(feat_dst, reconstructed_feat)
                    all_contrast_mse[output_nodes], _ = node_metrics(h_str, h_att)
                    h_str_full[output_nodes] = h_str
                    h_att_full[output_nodes] = h_att
                    feat_full[output_nodes] = feat_dst
                    rec_feat_full[output_nodes] = reconstructed_feat

                all_noise_mse = noise_mse_t.mean(dim=1)
                rec_rank = to_rank_score(all_reconstruction_mse)
                con_rank = to_rank_score(all_contrast_mse)
                dif_rank = to_rank_score(all_noise_mse)

                all_fused_score = (
                    args.alpha * con_rank +
                    ( 1- args.alpha) / 2 * rec_rank +
                    ( 1- args.alpha) / 2 * dif_rank
                )

                os.makedirs("results", exist_ok=True)
                results_dict = {}
                # report() 内部可能打印单次测试结果，此处屏蔽该输出。
                with contextlib.redirect_stdout(io.StringIO()):
                    auroc, auprc = report("Fused", all_fused_score, results_dict, ano_label)


                result = {
                    "run": run+1,
                    "epoch": epoch + 1,
                    "auroc": auroc,
                    "auprc": auprc
                }
                run_test_results.append(result.copy())

                # 每次测试完成后立即输出当前结果。
                print(
                    f"Run {run + 1:02d}/{num_runs:02d} | "
                    f"Epoch {epoch + 1:03d} | "
                    f"AUROC: {float(auroc):.4f} | "
                    f"AUPRC: {float(auprc):.4f}",
                    flush=True
                )
            end = time.time()
            inference_time = end - start
            peak_infer_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB   
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            results = {
                "avg_epoch_time_sec": avg_epoch_time,
                "inference_time_sec": inference_time,
                "peak_train_mem_MB": peak_train_mem,
                "peak_inference_mem_MB": peak_infer_mem
            }
    pbar.close()

    all_test_results.extend(run_test_results)

# 按测试 epoch 汇总所有 run，只输出 AUROC 和 AUPRC 的均值与标准差。
results_by_epoch = {}
for test_result in all_test_results:
    results_by_epoch.setdefault(test_result["epoch"], []).append(test_result)
print("***************Average results:")
epoch_statistics = []
for epoch in sorted(results_by_epoch):
    epoch_results = results_by_epoch[epoch]
    auroc_values = np.asarray(
        [item["auroc"] for item in epoch_results], dtype=float
    )
    auprc_values = np.asarray(
        [item["auprc"] for item in epoch_results], dtype=float
    )

    statistics = {
        "epoch": epoch,
        "auroc_mean": float(auroc_values.mean()),
        "auroc_std": float(auroc_values.std()),
        "auprc_mean": float(auprc_values.mean()),
        "auprc_std": float(auprc_values.std())
    }
    epoch_statistics.append(statistics)
    
    print(
        f"Epoch {epoch:03d} | "
        f"AUROC: {statistics['auroc_mean']:.4f} \\pm "
        f"{statistics['auroc_std']:.4f} | "
        f"AUPRC: {statistics['auprc_mean']:.4f} \\pm "
        f"{statistics['auprc_std']:.4f}",
        flush=True
    )

# 将每个 run 的逐 epoch 测试结果整理为分组结构。
run_results = {}
for test_result in all_test_results:
    run_key = f"run_{test_result['run']}"
    run_results.setdefault(run_key, []).append({
        "epoch": int(test_result["epoch"]),
        "auroc": float(test_result["auroc"]),
        "auprc": float(test_result["auprc"])
    })

dataset_dir = os.path.join("results", args.dataset)
os.makedirs(dataset_dir, exist_ok=True)

summary_file = os.path.join(
    dataset_dir,
    f"run{args.run}_hidden{args.hidden}_alpha{args.alpha}_"
    f"order_{args.order}_results.json"
)

summary = {
    "run_results": run_results,
    "epoch_statistics": epoch_statistics
}

with open(summary_file, "w") as f:
    json.dump(summary, f, indent=4)
