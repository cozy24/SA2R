import warnings
warnings.filterwarnings('ignore')

import argparse
from tqdm import tqdm
import torch
import os
import dgl
import json
from modules import utils
from modules.utils import *
from modules import model
import time

parser = argparse.ArgumentParser(description='AND_ONE')
parser.add_argument('--print_results', type=utils.str2bool, default=True)
parser.add_argument('--expr_name', type=str, default="None")
parser.add_argument('--dataset', type=str, default='Amazon')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--epoch', type=int, default=300)
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
print(f"Using device: {device}")

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
all_runs_best_results = []
all_runs_results = []

for run in range(num_runs):
    
    model_GAD = model.AND_ONE(graph.ndata['features'].shape[1], args.hidden, args.order, label=graph.ndata["label"], w1=args.w1, w2=args.w2, w3=args.w3).to(device)
    optimizer = torch.optim.Adam(model_GAD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_times = []

    torch.cuda.reset_peak_memory_stats(device)
 
    epochs = args.epoch
    results_loss_rec = []
    results_loss = []
    pbar = tqdm(total=epochs, desc=args.dataset)
    train_total_sec = 0

    patience = args.patience 
    best_loss = float('inf')
    counter = 0 
    best_auroc = 0.0
    best_result = None
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
                print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
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
                print('-------------------Test--------------------------')

                N = graph.num_nodes()

                all_reconstruction_mse = torch.zeros(N, device=device)
                all_noise_mse = torch.zeros(N, device=device)
                all_contrast_mse = torch.zeros(N, device=device)
                all_fused_score = torch.zeros(N, device=device)
                # 存储每个 t 下的分数，用于后续求均值
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
                    args.alpha * rec_rank +
                    ( 1- args.alpha) / 2 * con_rank +
                    ( 1- args.alpha) / 2 * dif_rank
                )

                os.makedirs("results", exist_ok=True)
                results_dict = {}
                auroc, auprc = report("Fused", all_fused_score, results_dict, ano_label)


                if auprc > best_auroc:
                    best_auroc = auprc
                    best_result = {
                        "run": run+1,
                        "train_time": train_total_sec,
                        "epoch": epoch,
                        "auroc": auroc,
                        "auprc": auprc
                    }
                    print(f"\n best AUPRC: {best_auroc:.4f} at Epoch {best_result['epoch']} "f"(AUROC: {best_result['auroc']:.4f})")  

                result = {
                    "run": run+1,
                    "train_time": train_total_sec,
                    "epoch": epoch,
                    "auroc": auroc,
                    "auprc": auprc
                }
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

    all_runs_best_results.append(best_result) 
    all_runs_results.append(result) 

summary_results     = compute_summary(all_runs_results)
summary_best        = compute_summary(all_runs_best_results)

print(f"\n🏆 Average Results over {len(all_runs_best_results)} runs:")
print(f"   - AUROC   : {summary_best['average_auroc']:.4f}")
print(f"   - AUPRC : {summary_best['average_auprc']:.4f}")

if args.run == 10:
    dataset_dir = os.path.join("results", args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    summary_file = os.path.join(
        dataset_dir,
        f"run{args.run}_hidden{args.hidden}_order{args.order}_epoch{args.epoch}_results.json"
    )

    summary = {
        "results_average": summary_results,
        "best_results_average": summary_best,
        "all_runs": all_runs_results,
        "all_best_runs": all_runs_best_results
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Saved all runs (results + best results) summary to {summary_file}")
