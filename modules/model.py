import torch
import torch.nn as nn
import sympy
import scipy.special
import dgl.function as fn
import torch.nn.functional as F
from torch.nn import init
import math
from dgl.nn import GraphConv

class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt
        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=True):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.d = d

    def batch(self, block, feat, start=None, end=None, agg="mean"):
        if not isinstance(feat, dict):
            if block.is_homogeneous:
                feat = {"_N": feat}
            else:
                ntypes = block.srctypes
                assert len(ntypes) == 1, "多节点类型必须传 dict"
                feat = {ntypes[0]: feat}

        h_dict = {}
        for ntype, h in feat.items():
            h = self.linear(h)
            h = self.act(h)
            h = self.dropout(h)
            h = self.linear2(h)
            h = self.act(h)
            h = self.dropout(h)
            h_dict[ntype] = h

        if block.is_homogeneous:
            h_final = torch.zeros([len(h_dict["_N"]), 0], device=h_dict["_N"].device)
            for conv in self.conv:
                h0 = conv(block, h_dict["_N"])
                h_final = torch.cat([h_final, h0], -1)

            if start is not None and end is not None:
                h_feats = h0.shape[1]
                return h_final[:, start * h_feats : end * h_feats]
            else:
                return self.linear3(h_final)

        rel_outputs = []
        for etype in block.canonical_etypes:
            sub_block = block[etype]
            src_type, _, _ = etype
            h_src = h_dict[src_type]

            h0 = None
            h_final = torch.zeros([len(h_src), 0], device=h_src.device)
            for conv in self.conv:
                h0 = conv(sub_block, h_src)
                h_final = torch.cat([h_final, h0], -1)

            if start is not None and end is not None:
                h_feats = h0.shape[1]
                rel_out = h_final[:, start * h_feats : end * h_feats]
            else:
                rel_out = self.linear3(h_final)

            rel_outputs.append(rel_out)

        if agg == "mean":
            h_final = torch.mean(torch.stack(rel_outputs, dim=0), dim=0)
        elif agg == "sum":
            h_final = torch.sum(torch.stack(rel_outputs, dim=0), dim=0)
        elif agg == "max":
            h_final = torch.max(torch.stack(rel_outputs, dim=0), dim=0)[0]
        else:
            raise ValueError("Unknown agg type")

        return h_final

class FusedProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.PReLU()  

    def forward(self, h):
        out = self.linear1(h)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out  

class LinearEncoder(nn.Module):
    def __init__(self, n_in, n_h, n_out, n_layers, dropout=0.5):
        super(LinearEncoder, self).__init__()
        self.mlp = nn.Sequential()
        assert n_layers > 0
        self.dropout = dropout

        if n_layers == 1:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_out, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
            if dropout > 0:
                self.mlp.add_module("dropout1", nn.Dropout(dropout))

        elif n_layers == 2:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_h, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
            if dropout > 0:
                self.mlp.add_module("dropout1", nn.Dropout(dropout))

            self.mlp.add_module("dense2", nn.Linear(n_h, n_out, bias=True))

        else:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_h, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
            if dropout > 0:
                self.mlp.add_module("dropout1", nn.Dropout(dropout))

            for i in range(n_layers - 2):
                self.mlp.add_module(f"dense{i + 2}", nn.Linear(n_h, n_h, bias=True))
                self.mlp.add_module(f"act{i + 2}", nn.PReLU())
                if dropout > 0:
                    self.mlp.add_module(f"dropout{i + 2}", nn.Dropout(dropout))

            self.mlp.add_module(f"dense{n_layers}", nn.Linear(n_h, n_out, bias=True))
            self.mlp.add_module(f"act{n_layers}", nn.PReLU())
            if dropout > 0:
                self.mlp.add_module(f"dropout{n_layers}", nn.Dropout(dropout))

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, block, feat):
        emb = self.mlp(feat)
        return emb

class RelGraphConv(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, agg="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.agg = agg
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True))

    def forward(self, block, h):
        if block.is_homogeneous:
            for conv in self.gnn_layers:
                h = conv(block, h)
            return h

        rel_outputs = []
        for etype in block.canonical_etypes:
            sub_block = block[etype]
            h0 = h
            for conv in self.gnn_layers:
                h0 = conv(sub_block, h0)
            rel_outputs.append(h0)

        h_stack = torch.stack(rel_outputs, dim=0)  
        if self.agg == "mean":
            h_final = torch.mean(h_stack, dim=0)
        elif self.agg == "sum":
            h_final = torch.sum(h_stack, dim=0)
        elif self.agg == "max":
            h_final = torch.max(h_stack, dim=0)[0]
        else:
            raise ValueError("Unknown agg type")
        return h_final

class PANPModel(nn.Module):
    def __init__(self, in_feats, hidden_dim, timesteps=1000, dropout=0,
                 num_layers=1, t_emb_dim=None, decoder_hidden=None,
                 schedule='linear', min_alpha_bar=0.05):
        super().__init__()
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.in_feats = in_feats

        if t_emb_dim is None:
            t_emb_dim = max(16, hidden_dim // 2)
        self.t_emb_dim = t_emb_dim
        if decoder_hidden is None:
            decoder_hidden = hidden_dim

        self.gnn_layers = RelGraphConv(hidden_dim=hidden_dim, num_layers=num_layers, agg="mean")
        self.input_proj = nn.Linear(in_feats, hidden_dim) if in_feats != hidden_dim else None
        self.dropout = nn.Dropout(dropout)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_emb_dim, max(self.t_emb_dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(max(self.t_emb_dim, hidden_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        def ResBlock(dim):
            return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.dec_block1 = ResBlock(decoder_hidden)
        self.dec_block2 = ResBlock(decoder_hidden)
        self.dec_norm = nn.LayerNorm(decoder_hidden)
        self.decoder_head = nn.Linear(decoder_hidden, in_feats)
        if decoder_hidden != hidden_dim:
            self.dec_proj = nn.Linear(hidden_dim, decoder_hidden)
            self.dec_unproj = nn.Linear(decoder_hidden, hidden_dim)
        else:
            self.dec_proj = None
            self.dec_unproj = None
        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self._build_schedule(timesteps, schedule=schedule, min_alpha_bar=min_alpha_bar)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def _build_schedule(self, timesteps, schedule='linear', min_alpha_bar=0.05):
        device = torch.device('cpu')
        if schedule == 'linear':
            beta_start, beta_end = 1e-4, 0.02
            beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
            alpha = 1.0 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
        elif schedule == 'cosine':
            s = 0.008
            t = torch.linspace(0, 1, timesteps + 1, device=device)
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f[1:] / f[0]
            alpha = torch.empty_like(alpha_bar)
            alpha[0] = alpha_bar[0]
            alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
            beta = 1.0 - alpha
        elif schedule == 'sigmoid':
            t = torch.linspace(0, 1, timesteps, device=device)
            logsnr_min, logsnr_max = -20.0, 20.0
            logsnr = logsnr_max + (logsnr_min - logsnr_max) * torch.sigmoid(10*(t-0.5))
            alpha_bar = torch.sigmoid(logsnr)
            alpha = torch.empty_like(alpha_bar)
            alpha[0] = alpha_bar[0]
            alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
            beta = 1.0 - alpha
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        valid = (alpha_bar >= min_alpha_bar).nonzero().flatten()
        self.t_cap = int(valid.max().item()) + 1 if len(valid) > 0 else timesteps

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def sinusoidal_embedding(self, t, dim=None):
        if dim is None:
            dim = self.t_emb_dim
        half = dim // 2
        device = t.device
        if half <= 0:
            return torch.zeros((t.size(0), dim), device=device)
        freq = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / max(1, half - 1)))
        x = t[:, None].float() * freq[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        if emb.size(1) < dim:
            pad = torch.zeros((t.size(0), dim - emb.size(1)), device=device)
            emb = torch.cat([emb, pad], dim=1)
        elif emb.size(1) > dim:
            emb = emb[:, :dim]
        return emb

    def forward_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.to(device=x0.device, dtype=torch.long)
        a_bar = self.alpha_bar[t].unsqueeze(-1).to(x0.device)  
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_om = torch.sqrt(1.0 - a_bar)
        x_t = sqrt_a_bar * x0 + sqrt_om * noise
        return x_t, noise

    def reverse_denoise(self, block, x_t, t):
        device = x_t.device
        block = block.to(device)  
        N = x_t.shape[0]
        dst_idx = torch.arange(block.num_dst_nodes())

        h = self.input_proj(x_t) if self.input_proj is not None else x_t
        h = F.layer_norm(h, h.shape[1:])

        for i, mlp in enumerate(self.mlp_layers):
            h_mlp = F.relu(mlp(h))
            h_mlp = F.dropout(h_mlp, p=0.2, training=self.training)

            h_gnn = self.gnn_layers(block, h)
            h_gnn = F.relu(h_gnn)
            h_gnn = F.dropout(h_gnn, p=0.2, training=self.training)

            h = (h[dst_idx] + h_mlp[dst_idx] + h_gnn[dst_idx]) / 3

        t = t.to(device=device, dtype=torch.long)
        t_emb = self.t_mlp(self.sinusoidal_embedding(t, self.t_emb_dim))
        if t_emb.size(0) == 1:
            t_emb = t_emb.expand(N, -1)

        scale = torch.tanh(self.film_scale(t_emb))[dst_idx]
        shift = torch.tanh(self.film_shift(t_emb))[dst_idx]

        h = h * (1.0 + scale) + shift

        dec = h
        if self.dec_proj is not None:
            dec = self.dec_proj(dec)
        dec = dec + F.relu(self.dec_block1(dec))
        dec = dec + F.relu(self.dec_block2(dec))
        dec = self.dec_norm(dec)
        if self.dec_unproj is not None:
            dec = self.dec_unproj(dec)

        pred_noise = self.decoder_head(dec)
        return pred_noise


    def forward(self, block, feat, t=None, sample_t_per_node=True, t_max_ratio=0.8):
        device = feat.device
        N = feat.shape[0]
        t_max = max(1, int(self.timesteps * t_max_ratio))

        if sample_t_per_node:
            if t is None:
                t = torch.randint(0, t_max, (N,), device=feat.device)
        else:
            t0 = torch.randint(0, t_max, (1,), device=device, dtype=torch.long)
            t = t0.expand(N).to(device)

        x_t, noise = self.forward_noise(feat, t, None)
        pred_noise = self.reverse_denoise(block, x_t, t)

        return {
            "pred_noise": pred_noise,
            "noise": noise,
        }

class AND_ONE(nn.Module):
    def __init__(self, n_in, n_h, d, label, w1, w2, w3, timesteps=1000, batch_size=8192):
        super().__init__()
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.norm = nn.LayerNorm(n_h)
        self.batch_size = batch_size
        self.mlp = LinearEncoder(n_in, n_h, n_h, 2)
        self.bwgnn = BWGNN(n_in, n_h, n_h, None, d=d, batch=True)
        self.proj = FusedProjector(2 * n_h, n_h, n_in)
        self.noise = PANPModel(n_in, n_h, timesteps, dropout=0.5)
        self.label = label
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def compute_contrastive_loss(self, h_view1, h_view2, temperature=0.3, k=10): 
        h1 = F.normalize(h_view1, dim=1)  
        h2 = F.normalize(h_view2, dim=1)  
        batch_size = h1.size(0)
        device = h1.device
        pos_score = torch.sum(h1 * h2, dim=1, keepdim=True) / temperature  
        neg_scores = []

        for _ in range(k):
            perm = torch.randperm(batch_size, device=device)
            h2_neg = h2[perm]
            neg_score = torch.sum(h1 * h2_neg, dim=1, keepdim=True) / temperature  
            neg_scores.append(neg_score)

        neg_scores = torch.cat(neg_scores, dim=1)
        logits = torch.cat([pos_score, neg_scores], dim=1)  
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)  
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss


    def forward(self, blocks, feat, t=None):
        device = next(self.parameters()).device  
        feat = feat.to(device)
        block = blocks[-1]

        dst_idx = torch.arange(block.num_dst_nodes())
        h_str = self.bwgnn.batch(block, feat)[dst_idx]
        h_att = self.mlp(block, feat)[dst_idx]
        reconstructed_feat = self.proj(torch.cat([h_str, h_att], dim=1))
        node_mse = ((reconstructed_feat[dst_idx] - feat[dst_idx]) ** 2).mean(dim=1)  # [num_dst_nodes]
        loss_rec = node_mse.mean()  
        loss_ctr = self.compute_contrastive_loss(h_str, h_att)           

        attr = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)   
        output = self.noise(block, attr, t)
        noise = output['noise']
        pred_noise = output['pred_noise']

        node_noise_mse = ((pred_noise[dst_idx] - noise[dst_idx]) ** 2).mean(dim=1)  # [num_dst_nodes]
        loss_moise = node_noise_mse.mean()

        w_ctr = self.w1
        w_rec = self.w2
        w_noise = self.w3

        loss = w_ctr * loss_ctr + w_rec * loss_rec + w_noise * loss_moise
        
        return {
            'loss': loss,
            'h_str': h_str,
            'h_att': h_att,
            'reconstructed_feat': reconstructed_feat,
            "noise": output['noise'],
            "pred_noise": output['pred_noise']
        }


