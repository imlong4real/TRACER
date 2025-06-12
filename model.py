#####################################################################################
# GENESIS – Graph-Embedded Network for Expressing Spatial and Interaction Signals  ##
# Author: Long Yuan                                                                ##
# Email: lyuan13@jhmi.edu                                                          ##
#####################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from collections import defaultdict
import numpy as np
import math
from scipy.spatial import cKDTree

#
class SpatialAwareAggregator(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads=2, attention_threshold=0.2, positional_sigma=0.5):
        super().__init__(aggr='add')
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.attention_threshold = attention_threshold
        self.positional_sigma   = positional_sigma

        # one gate per head 
        self.gate_param = nn.Parameter(torch.zeros(num_heads))

        self.query = nn.Linear(in_dim, out_dim)
        self.key   = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, positions, gene_embeddings, return_edges=False):
        src, tgt = edge_index

        # feature Q·K
        q = self.query(x[src]).view(-1, self.num_heads, self.head_dim)
        k = self.key  (x[tgt]).view(-1, self.num_heads, self.head_dim)
        feat_attn = (q * k).sum(-1) / math.sqrt(self.head_dim)                 # [E, heads]

        # positional attenuation
        with torch.no_grad():
            d = (positions[src] - positions[tgt]).norm(dim=1)
            dist_attn = torch.exp(-d**2/(2*self.positional_sigma**2))
            dist_attn = dist_attn.unsqueeze(1).expand(-1, self.num_heads)      # [E, heads]
        attn_phys = feat_attn * dist_attn

        # gene‐based attention
        gene_sim = F.cosine_similarity(
            gene_embeddings[src], gene_embeddings[tgt], dim=-1
        )
        attn_gene = gene_sim.unsqueeze(1).expand(-1, self.num_heads)          # [E, heads]

        # per‐head gating
        w = torch.sigmoid(self.gate_param).view(1, -1)                         # [1, heads]
        attn_raw = w * attn_gene + (1 - w) * attn_phys                         # [E, heads]

        attn = F.leaky_relu(attn_raw, negative_slope=0.2)
        attn_mean = attn.mean(dim=1)                                           # [E]

        # prune
        keep = attn_mean > self.attention_threshold
        pruned_idx  = edge_index[:, keep]
        pruned_attn = attn[keep]

        # aggregate
        out = self.propagate(pruned_idx, x=x, attn=pruned_attn)
        if return_edges:
            return out, pruned_idx, attn_mean[keep]
        return out

    def message(self, x_j, attn):
        v = self.value(x_j).view(-1, self.num_heads, self.head_dim)
        return (v * attn.unsqueeze(-1)).view(-1, self.num_heads * self.head_dim)
#
class BioRandomWalker:
    def __init__(self, edge_index, walk_params):
        src, dst = edge_index.cpu().tolist()
        self.adj_list = defaultdict(list)
        for u, v in zip(src, dst):
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
        
        self.hard_neg_ratio = walk_params.get('hard_neg_ratio', 0.5)
        self.hard_topk      = walk_params.get('hard_topk', 10)

    def sample_negatives(self, pos_pairs, batch_nodes, gene_embeddings, momentum_embeddings):
        """
        gene_embeddings: [B, H] embeddings for the B root nodes
        batch_nodes:    [B] global IDs of those root nodes
        """
        device = pos_pairs.device

        # Detach before any CPU or numpy conversion
        gene_tensor = momentum_embeddings.detach()
        gene_emb = gene_tensor / (gene_tensor.norm(dim=1, keepdim=True) + 1e-8)
        batch_nodes_np = batch_nodes.detach().cpu().numpy()
        gene_emb_np    = gene_emb.cpu().numpy()

        batch_set = set(batch_nodes_np)
        neg_pairs = []

        for src_gid, _ in pos_pairs.detach().cpu().tolist():
            if src_gid not in batch_set:
                continue
            src_idx = int(np.where(batch_nodes_np == src_gid)[0][0])
            src_vec = gene_emb_np[src_idx]

            forbid = set(self.adj_list.get(src_gid, [])) | {src_gid}
            if np.random.rand() < self.hard_neg_ratio:
                sims = gene_emb_np @ src_vec
                mask = np.array([(nid not in forbid) for nid in batch_nodes_np])
                sims[~mask] = -np.inf
                topk = np.argpartition(-sims, self.hard_topk)[:self.hard_topk]
                valid = topk[sims[topk] > -np.inf]
                if len(valid):
                    pick = np.random.choice(valid)
                    neg_gid = int(batch_nodes_np[pick])
                    neg_pairs.append([src_gid, neg_gid])
                    continue

            # fallback random
            pool = list(batch_set - forbid) or list(batch_set)
            neg_pairs.append([src_gid, int(np.random.choice(pool))])

        return torch.tensor(neg_pairs, dtype=torch.long, device=device)

######### Noise Injection #########
def corrupt_features(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Given a float‐tensor x of shape [N, F], randomly replace ~noise_level fraction of entries
    with random integers in {-1, 0, +1}. Returns a new tensor of the same shape.
    """
    corrupted = x.clone()
    # Build a mask (same shape as corrupted) that is True with probability = noise_level
    noise_mask = (torch.rand_like(corrupted) < noise_level)
    if noise_mask.any():
        # Sample random integers in {-1, 0, +1} for all True positions in noise_mask
        rnd = torch.randint(-1, 2, (noise_mask.sum(),),
                            device=x.device, dtype=x.dtype)
        corrupted[noise_mask] = rnd
    return corrupted

#
class GENESIS(nn.Module):
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 128,
                 walk_params: dict = None,
                 attention_threshold: float = 0.2,
                 lambda_ortho: float = 1e-3,
                 lambda_lap:   float = 1e-3):
        super().__init__()

        # FIFO memory queue
        self.register_buffer('memory_queue', torch.zeros(4096, hidden_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.temperature = nn.Parameter(torch.tensor(1.0))
        #self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature.requires_grad_(True)

        # projector
        self.projector = nn.Sequential(
        nn.Linear(hidden_dim, 2*hidden_dim),
        nn.LayerNorm(2*hidden_dim), 
        nn.ReLU(),
        nn.Linear(2*hidden_dim, hidden_dim)  
        )

        # encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Momentum encoder
        self.momentum_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.momentum_ema = 0.999  # EMA decay rate


        # Three aggregators
        self.agg1 = SpatialAwareAggregator(hidden_dim, hidden_dim, num_heads=2, attention_threshold=attention_threshold)
        self.agg2 = SpatialAwareAggregator(hidden_dim, hidden_dim, num_heads=2, attention_threshold=attention_threshold)
        self.agg3 = SpatialAwareAggregator(hidden_dim, hidden_dim, num_heads=2, attention_threshold=attention_threshold)
        
        # 
        self.lambda_ortho = lambda_ortho
        self.lambda_lap   = lambda_lap

        # Walk params
        self.walk_params = walk_params or {
            'length': 3,
            'hard_neg_ratio': 0.5,
            'hard_topk': 50
        }

        self.attention_threshold = attention_threshold

    def update_momentum(self):
        with torch.no_grad():
            for param, mom_param in zip(self.node_encoder.parameters(), self.momentum_encoder.parameters()):
                mom_param.data = self.momentum_ema * mom_param.data + (1 - self.momentum_ema) * param.data

    def forward(self, x, edge_index, positions):
        x = x.float()
        positions = positions.float()
        h = self.node_encoder(x)
        h = self.agg1(h, edge_index, positions, h)
        h = self.agg2(h, edge_index, positions, h)
        h = self.agg3(h, edge_index, positions, h)
        return h

    def compute_contrastive_loss(self, data, loader, device, optimizer=None, noise_level: float = 0.0):
        walker = BioRandomWalker(data.edge_index, self.walk_params)
        total_loss = 0.0
        count = 0

        for batch in loader:
            batch = batch.to(device)
            B = batch.batch_size

            # Inject feature noise
            if noise_level > 0.0:
                # replace a frac=noise_level of entries with random {-1,0,1}.
                batch.x = corrupt_features(batch.x, noise_level=noise_level)

            roots = batch.n_id[:B]  # Global IDs of root nodes

            # Get embeddings
            x_online = batch.x.float()
            pos_online = batch.pos.float()
            z      = self(x_online, batch.edge_index, pos_online)
            #z_root = F.normalize(z[:B], dim=1)
            proj_online = self.projector(z[:B])  
            z_root      = F.normalize(proj_online, dim=1)

            # Positive pairs
            src, dst = batch.edge_index
            mask = (src < B) & (dst < B)
            pos_pairs_global = torch.stack([roots[src[mask]], roots[dst[mask]]], dim=1)

            # Always ensure 2D shape, even if empty
            root_to_local = {gid.item(): i for i, gid in enumerate(roots)}
            pos_pairs_local = torch.zeros((0, 2), dtype=torch.long, device=device)  # Initialize as 2D
            
            if pos_pairs_global.size(0) > 0:
                # Convert global IDs to local indices
                pos_list = []
                for s, d in pos_pairs_global:
                    s_local = root_to_local[s.item()]
                    d_local = root_to_local[d.item()]
                    pos_list.append([s_local, d_local])
                pos_pairs_local = torch.tensor(pos_list, device=device)

            self.update_momentum()
            with torch.no_grad():
                x_mom   = batch.x.float()
                pos_mom = batch.pos.float()
                mom_h   = self.momentum_encoder(x_mom)         
                #mom_z_root = F.normalize(mom_h[:B], dim=1)     
                proj_mom    = self.projector(mom_h[:B])  
                mom_z_root  = F.normalize(proj_mom, dim=1)
            
            # Enqueue momentum embeddings into FIFO queue
            with torch.no_grad():
                batch_size = mom_z_root.size(0)         # B
                ptr        = int(self.queue_ptr)        # current pointer
                end        = ptr + batch_size
                buffer_size = self.memory_queue.size(0)

                if end <= buffer_size:
                    # fits without wrapping
                    self.memory_queue[ptr:end] = mom_z_root
                else:
                    # wrap-around write
                    first = buffer_size - ptr
                    self.memory_queue[ptr:] = mom_z_root[:first]
                    self.memory_queue[: batch_size - first] = mom_z_root[first:]

                # advance pointer modulo buffer_size
                self.queue_ptr[0] = end % buffer_size

            neg_pairs_global = walker.sample_negatives(
                pos_pairs_global,
                roots,
                z_root,
                mom_z_root
            )
            # Negative pairs with same 2D guarantee
            #neg_pairs_global = walker.sample_negatives(pos_pairs_global, roots, z_root)
            neg_pairs_local = torch.zeros((0, 2), dtype=torch.long, device=device)
            
            if neg_pairs_global.size(0) > 0:
                neg_list = []
                for s, d in neg_pairs_global:
                    s_local = root_to_local[s.item()]
                    d_local = root_to_local[d.item()]
                    neg_list.append([s_local, d_local])
                neg_pairs_local = torch.tensor(neg_list, device=device)

            #neg_s_batch = (  # batch‐only negatives
            #    z_root[neg_pairs_local[:,0]] * z_root[neg_pairs_local[:,1]]
            #).sum(-1) / self.temperature  if neg_pairs_local.size(0)>0 else torch.tensor([], device=device)

            # Compute similarities
            pos_s = (z_root[pos_pairs_local[:, 0]] * 
                     z_root[pos_pairs_local[:, 1]]).sum(-1) / self.temperature \
                        if pos_pairs_local.size(0) > 0 else torch.tensor([], device=device)
            
            neg_s_batch = (z_root[neg_pairs_local[:, 0]] * 
                           z_root[neg_pairs_local[:, 1]]).sum(-1) / self.temperature \
                            if neg_pairs_local.size(0) > 0 else torch.tensor([], device=device)
            
            # Sample from memory queue
            valid_size = min(self.memory_queue.size(0), int(self.queue_ptr))
            if valid_size > 0 and epoch > 5:
                idx          = torch.randint(0, valid_size, (B,), device=device)
                queue_embeds = self.memory_queue[idx]           # [B, H]
                neg_s_queue  = torch.mm(z_root, queue_embeds.T).flatten()  # [B*B]
            else:
                neg_s_queue = torch.tensor([], device=device)

            neg_s = torch.cat([neg_s_batch, neg_s_queue], dim=0)
            
            # Loss calculation 
            if pos_s.numel() == 0 or (neg_s_batch.numel() + neg_s_queue.numel()) == 0:
                continue # Skip if truly no pairs, although highly unlikely based on my observation
            
            # Clamp logits to prevent overflow
            pos_s = torch.clamp(pos_s, min=-50, max=50) 
            neg_s = torch.clamp(neg_s, min=-50, max=50)
            
            # InfoNCE
            #loss_pos = -torch.log(torch.sigmoid(pos_s)).mean() if pos_s.numel() > 0 else 0
            #loss_neg = -torch.log(torch.sigmoid(-neg_s)).mean() if neg_s.numel() > 0 else 0
            labels_pos = torch.ones_like(pos_s)
            labels_neg = torch.zeros_like(neg_s)
            loss_pos   = F.binary_cross_entropy_with_logits(pos_s, labels_pos)
            loss_neg   = F.binary_cross_entropy_with_logits(neg_s, labels_neg)

            # Hinge‐margin 
            margin = 0.2
            margin_loss = F.relu(margin - pos_s.unsqueeze(1) + neg_s.unsqueeze(0)).mean()

            # Approxiamte orthogonality on the projections z_root
            H = z_root.size(1)
            G = z_root.T @ z_root            # [H,H] - 64×64
            G_sq_sum = G.pow(2).sum()        
            ortho_loss = (G_sq_sum - H).clamp(min=0).add(1e-6).sqrt()

            # Approximate Laplacian smoothness on positive edges
            src_idx, dst_idx = pos_pairs_local.unbind(1)
            P = src_idx.size(0)
            MAX_S = 128   # sample at most 128 edges
            if P > MAX_S:
                perm = torch.randperm(P, device=device)[:MAX_S]
                src_idx = src_idx[perm]
                dst_idx = dst_idx[perm]
                if hasattr(batch, 'attn_mean') and batch.attn_mean.numel():
                    edge_w = batch.attn_mean[mask][perm]
                else:
                    edge_w = torch.ones(MAX_S, device=device)
            else:
                if hasattr(batch, 'attn_mean') and batch.attn_mean.numel():
                    edge_w = batch.attn_mean[mask]
                else:
                    edge_w = torch.ones(P, device=device)

            diffs = z_root[src_idx] - z_root[dst_idx]      # [<=MAX_S, H]
            lap_loss = (edge_w * diffs.pow(2).sum(-1)).mean()


            loss = loss_pos + loss_neg + 0.1 * margin_loss + self.lambda_ortho * ortho_loss + self.lambda_lap * lap_loss

            total_loss += loss.item()
            count += 1

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

        return total_loss / count if count > 0 else 0.0