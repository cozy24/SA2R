import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.functional as F
import torch
import datetime
import dgl
def node_metrics(feat_dst, reconstructed_feat):
    """
    feat_dst: [N, d] 原始节点特征
    reconstructed_feat: [N, d] 重建节点特征
    返回:
        node_mse: 每个节点的 MSE  (越大表示重建误差越大)
        node_sim: 每个节点的余弦相似度 (越小表示越不相似)
    """
    # 每个节点的 MSE
    node_mse = torch.mean((feat_dst - reconstructed_feat) ** 2, dim=1)  # [N]

    # 每个节点的余弦相似度
    feat_norm = F.normalize(feat_dst, p=2, dim=1)
    recon_norm = F.normalize(reconstructed_feat, p=2, dim=1)
    node_sim = 1 - torch.sum(feat_norm * recon_norm, dim=1)  # [N]

    return node_mse, node_sim

def compute_contrastive_loss(self, h_view1, h_view2, temperature=0.5, k=5):
    """
    多负样本对比损失（lightweight）
    正样本：h1[i] vs h2[i]
    负样本：h1[i] vs h2[perm[j]], j ≠ i, 取 k 个
    """
    h1 = F.normalize(h_view1, dim=1)  # [B, D]
    h2 = F.normalize(h_view2, dim=1)  # [B, D]

    batch_size = h1.size(0)
    device = h1.device

    # === 正样本打分 ===
    pos_score = torch.sum(h1 * h2, dim=1, keepdim=True) / temperature  # [B, 1]

    # === 构造 k 个负样本 ===
    neg_scores = []
    for _ in range(k):
        perm = torch.randperm(batch_size, device=device)
        h2_neg = h2[perm]
        neg_score = torch.sum(h1 * h2_neg, dim=1, keepdim=True) / temperature  # [B, 1]
        neg_scores.append(neg_score)

    neg_scores = torch.cat(neg_scores, dim=1)  # [B, k]

    # === 拼接 logit 和标签 ===
    logits = torch.cat([pos_score, neg_scores], dim=1)  # [B, 1 + k]
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 正样本为第 0 类

    loss = F.cross_entropy(logits, labels)
    return loss

def analyze_spectral_energy_distribution(
    low_pass_filter,
    high_pass_filter,
    denoised_low_pass_filter,
    denoised_high_pass_filter,
    ano_label,
    show_plot=True,
    save_fig=False,
    fig_path="spectral_energy_distribution.png",
    dpi=300
):
    """
    分析正常节点与异常节点在去噪前后谱能量分布及高频能量占比变化，并可保存图像。

    Args:
        low_pass_filter: Tensor [N, d]，原始低频谱嵌入
        high_pass_filter: Tensor [N, d]，原始高频谱嵌入
        denoised_low_pass_filter: Tensor [N, d]，去噪后的低频谱嵌入
        denoised_high_pass_filter: Tensor [N, d]，去噪后的高频谱嵌入
        ano_label: Tensor or np.ndarray [N]，异常标签，1表示异常，0表示正常
        show_plot: 是否显示图像
        save_fig: 是否保存图像
        fig_path: 保存图像路径
        dpi: 图像分辨率
    """
    # 数据转为 numpy
    def to_np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else x
    low_pass = to_np(low_pass_filter)
    high_pass = to_np(high_pass_filter)
    d_low_pass = to_np(denoised_low_pass_filter)
    d_high_pass = to_np(denoised_high_pass_filter)
    label_np = to_np(ano_label)

    is_normal = label_np == 0
    is_anomaly = label_np == 1

    # 能量计算
    def energy(x): return (x ** 2).sum(axis=1)
    e_low = energy(low_pass)
    e_high = energy(high_pass)
    e_d_low = energy(d_low_pass)
    e_d_high = energy(d_high_pass)

    # 高频能量占比
    r_high = e_high / (e_low + e_high + 1e-6)
    r_d_high = e_d_high / (e_d_low + e_d_high + 1e-6)

    # 输出统计
    def stat(name, val):
        return f"{name:<20} | Normal: mean={val[is_normal].mean():.4f}, std={val[is_normal].std():.4f} | Anomaly: mean={val[is_anomaly].mean():.4f}, std={val[is_anomaly].std():.4f}"

    print("📊 能量分布统计：")
    print(stat("原始低频能量", e_low))
    print(stat("原始高频能量", e_high))
    print(stat("去噪低频能量", e_d_low))
    print(stat("去噪高频能量", e_d_high))
    print(stat("原始高频占比", r_high))
    print(stat("去噪高频占比", r_d_high))

    if show_plot or save_fig:
        plt.figure(figsize=(18, 12))

        def plot_kde(idx, title, normal_val, anomaly_val):
            plt.subplot(3, 2, idx)
            sns.kdeplot(normal_val, label='Normal', linestyle='--', linewidth=2)
            sns.kdeplot(anomaly_val, label='Anomaly', color='red', linewidth=2)
            plt.title(title)
            plt.xlabel('Energy')
            plt.ylabel('Density')
            plt.legend()

        plot_kde(1, "原始低频能量", e_low[is_normal], e_low[is_anomaly])
        plot_kde(2, "原始高频能量", e_high[is_normal], e_high[is_anomaly])
        plot_kde(3, "去噪低频能量", e_d_low[is_normal], e_d_low[is_anomaly])
        plot_kde(4, "去噪高频能量", e_d_high[is_normal], e_d_high[is_anomaly])
        plot_kde(5, "高频能量占比", r_high[is_normal], r_high[is_anomaly])
        plot_kde(6, "去噪后高频占比", r_d_high[is_normal], r_d_high[is_anomaly])

        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # fig_path = f"{'spectral_energy_distribution'}_{timestamp}.png"
        # 保存图像
        if save_fig:
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ 图像已保存至: {fig_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()



def plot_score_distribution_combined(score_low, score_high, labels, show_kde=True, show_hist=False,
                                     save_dir=None, file_prefix="score_distribution_combined"):
    """
    可视化低频和高频异常分数的分布（异常vs正常），将多条曲线画到同一张图上，并保存图片到指定目录。
    """


    assert len(score_low) == len(score_high) == len(labels), "Score与label长度不一致"

    # 设置Seaborn风格和色调
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    is_anomaly = (labels == 1)
    is_normal = (labels == 0)

    plt.figure(figsize=(12, 7))

    # 配置颜色
    color_map = {
        'LowFreq-Normal': '#1f77b4',
        'LowFreq-Anomaly': '#d62728',
        'HighFreq-Normal': '#1f77b4',
        'HighFreq-Anomaly': '#d62728',
        'LowFreq-All': 'green',
        'HighFreq-All': 'green'
    }

    # KDE 曲线
    if show_kde:
        sns.kdeplot(score_low[is_normal], label='LowFreq - Normal', linestyle='--', linewidth=2,
                    color=color_map['LowFreq-Normal'], bw_adjust=0.8)
        sns.kdeplot(score_low[is_anomaly], label='LowFreq - Anomaly', linestyle='--', linewidth=2,
                    color=color_map['LowFreq-Anomaly'], bw_adjust=0.8)
        sns.kdeplot(score_high[is_normal], label='HighFreq - Normal', linewidth=2,
                    color=color_map['HighFreq-Normal'], bw_adjust=0.8)
        sns.kdeplot(score_high[is_anomaly], label='HighFreq - Anomaly', linewidth=2,
                    color=color_map['HighFreq-Anomaly'], bw_adjust=0.8)
        # sns.kdeplot(score_low, label='LowFreq - All', linestyle='--', linewidth=1.5,
        #             color=color_map['LowFreq-All'], bw_adjust=0.8)
        # sns.kdeplot(score_high, label='HighFreq - All', linewidth=1.5,
        #             color=color_map['HighFreq-All'], bw_adjust=0.8)

    # 直方图
    if show_hist:
        plt.hist(score_low[is_normal], bins=40, alpha=0.2, label='LowFreq - Normal', color=color_map['LowFreq-Normal'])
        plt.hist(score_low[is_anomaly], bins=40, alpha=0.2, label='LowFreq - Anomaly', color=color_map['LowFreq-Anomaly'])
        plt.hist(score_high[is_normal], bins=40, alpha=0.2, label='HighFreq - Normal', color=color_map['HighFreq-Normal'])
        plt.hist(score_high[is_anomaly], bins=40, alpha=0.2, label='HighFreq - Anomaly', color=color_map['HighFreq-Anomaly'])

    plt.title("Score Distribution by Frequency and Label", fontsize=16)
    plt.xlabel("Score (1 - Cosine Similarity)", fontsize=13)
    plt.ylabel("Density" if show_kde else "Count", fontsize=13)
    plt.legend(loc='upper right', fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    # 保存图像
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_prefix}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"图像已保存到：{filepath}")

    plt.close()

def local_contrastive_loss(original_embed, denoised_embed, temperature=0.5):
    original_embed = F.normalize(original_embed, dim=1)
    denoised_embed = F.normalize(denoised_embed, dim=1)
    logits = torch.matmul(original_embed, denoised_embed.T) / temperature
    labels = torch.arange(original_embed.shape[0]).to(original_embed.device)
    return F.cross_entropy(logits, labels)

def local_contrastive_loss_with_neighbors(node_embed, denoised_embed, adj_matrix, temperature=0.5):
    """
    node_embed: 原始图滤波后的节点表示 (N, D)
    denoised_embed: 去噪图的表示 (N, D)
    adj_matrix: 稀疏邻接矩阵（仅包含 batch 中节点）(N, N)
    """
    node_embed = F.normalize(node_embed, dim=1)
    denoised_embed = F.normalize(denoised_embed, dim=1)

    sim_matrix = torch.matmul(node_embed, denoised_embed.T)  # [N, N]
    sim_matrix = sim_matrix / temperature

    # 正样本掩码：邻居为正样本（包括自己）
    pos_mask = (adj_matrix > 0).float()
    pos_mask.fill_diagonal_(1.0)  # 自己也是正样本

    # 计算正样本平均相似度
    pos_sim = (F.log_softmax(sim_matrix, dim=1) * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

    # 损失是负 log-likelihood，越大表示正样本相似度越低
    loss = -pos_sim.mean()
    return loss



def visualize_tsne_with_anomaly(original, denoised, ano_label, title_prefix):
    import torch
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import os

    # 创建保存路径
    save_dir = 'tsne_visualizations'
    os.makedirs(save_dir, exist_ok=True)
    original = original.cpu()
    denoised = denoised.cpu()


    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

    # 分别降维
    original_2d = tsne.fit_transform(original)
    denoised_2d = tsne.fit_transform(denoised)

    # 画图 & 保存函数
    def plot_tsne(features_2d, label, title, filename):
        plt.figure(figsize=(6, 5))
        normal = features_2d[label == 0]
        abnormal = features_2d[label == 1]
        plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal', alpha=0.6, s=10)
        plt.scatter(abnormal[:, 0], abnormal[:, 1], c='red', label='Anomaly', alpha=0.6, s=10)
        plt.legend()
        plt.title(title)
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # 可视化并保存原始特征图
    plot_tsne(original_2d, ano_label, f'{title_prefix} - Original', f'{title_prefix.lower().replace(" ", "_")}_original.png')

    # 可视化并保存去噪后特征图
    plot_tsne(denoised_2d, ano_label, f'{title_prefix} - Denoised', f'{title_prefix.lower().replace(" ", "_")}_denoised.png')

def cosine_similarity_matrix(x, y):
    # 计算所有样本两两之间的余弦相似度矩阵
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return torch.mm(x, y.T)

def frequency_contrastive_loss(low_freq, high_freq, temperature=0.5):
    """NT-Xent-style contrastive loss between low and high frequency features"""
    sim_matrix = cosine_similarity_matrix(low_freq, high_freq)  # N x N
    sim_exp = torch.exp(sim_matrix / temperature)

    # 正样本在对角线
    positives = torch.diag(sim_exp)
    negatives = sim_exp.sum(dim=1) - positives

    loss = -torch.log(positives / (negatives + 1e-8)).mean()
    return loss

def compute_contrastive_loss(z1, z2, temperature=0.5):
    """
    z1: [N, d]  (low pass特征)
    z2: [N, d]  (high pass特征)
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)   # L2归一化
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # [2N, d]
    similarity_matrix = torch.matmul(representations, representations.T)  # [2N, 2N]

    # 构建正样本对
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)  # 避免自己跟自己比

    # 相似度除以温度
    similarity_matrix = similarity_matrix / temperature

    # 把对角线（自己跟自己）mask掉
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # 用cross-entropy，输入是[logits]和[target]
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def compute_structure_score(block, batch_features):
    """
    计算每个目标节点与其所有邻居节点特征的平均余弦相似度，反转为异常度得分（越大越异常）。

    参数：
        block: DGLBlock，当前 mini-batch 的最后一层 block。
        batch_features: Tensor，源节点的特征张量 [src_nodes, feat_dim]。

    返回：
        structure_score: Tensor，每个目标节点的结构异常得分 [dst_nodes]。
    """
    dst_num = block.num_dst_nodes()
    dst_feat = batch_features[:dst_num]  # [dst_num, feat_dim]

    src_ids, dst_ids = block.edges()  # [E], [E]
    src_feat = batch_features[src_ids]     # [E, feat_dim]
    dst_feat_expanded = dst_feat[dst_ids]  # [E, feat_dim]

    # 计算所有边上的余弦相似度，并反转为相异度 (0 表示相似，1 表示不相似)
    sim = F.cosine_similarity(dst_feat_expanded, src_feat, dim=1)  # [E]
    diff = (1 - sim) / 2.0  # 归一化到 [0, 1]，越大越不相似

    # 聚合到每个 dst 节点上（取平均）
    score_sum = torch.zeros(dst_num, device=batch_features.device)
    deg = torch.zeros(dst_num, device=batch_features.device)
    score_sum = score_sum.index_add(0, dst_ids, diff)
    deg = deg.index_add(0, dst_ids, torch.ones_like(diff))

    structure_score = score_sum / (deg + 1e-8)  # 平均相异度（异常得分）

    return structure_score

def structure_reconstruction_loss(H, A_true, threshold=0.7):
    """
    H: [N, d] 节点表示
    A_true: [N, N] 原始邻接矩阵（二值 0/1）
    threshold: 相似度阈值

    返回：结构重建损失
    """
    # 计算余弦相似度
    H_norm = F.normalize(H, p=2, dim=1)
    sim = torch.matmul(H_norm, H_norm.T)  # [N, N]

    # 二值化重建邻接矩阵
    A_pred = (sim > threshold).float()

    # Binary Cross Entropy Loss
    loss = F.binary_cross_entropy(A_pred, A_true.float())

    return loss

def get_adj_true_from_block(block):
    """
    从 DGL Block 构造目标节点之间的邻接矩阵（对结构重建用）
    返回：adj ∈ [num_dst_nodes, num_dst_nodes]
    """
    # 获取目标节点 ID（全局）
    dst_nid = block.dstdata[dgl.NID]  # [N_dst]
    dst_nid_set = set(dst_nid.tolist())
    dst_id_map = {nid.item(): i for i, nid in enumerate(dst_nid)}

    # 获取 Block 的边：全局 src 和 dst ID
    src, dst = block.edges()
    src_nid_all = block.srcdata[dgl.NID][src]
    dst_nid_all = block.dstdata[dgl.NID][dst]

    # 筛选：src 和 dst 都是目标节点
    mask = [(s.item() in dst_nid_set) and (d.item() in dst_nid_set)
            for s, d in zip(src_nid_all, dst_nid_all)]

    src_filtered = src_nid_all[mask]
    dst_filtered = dst_nid_all[mask]

    # 映射为目标节点索引
    src_idx = torch.tensor([dst_id_map[n.item()] for n in src_filtered], device=block.device)
    dst_idx = torch.tensor([dst_id_map[n.item()] for n in dst_filtered], device=block.device)

    # 构建邻接矩阵
    N = block.num_dst_nodes()
    adj = torch.zeros((N, N), device=block.device)
    adj[src_idx, dst_idx] = 1.0
    adj[dst_idx, src_idx] = 1.0  # 若为无向图
    return adj