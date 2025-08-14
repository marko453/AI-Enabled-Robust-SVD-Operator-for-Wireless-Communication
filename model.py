
import os, math, argparse
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

# -------------------- IO & Dataset --------------------
def parse_cfg(path: str) -> Tuple[int, int, int, int, int]:
    with open(path) as f:
        vals = [int(x) for x in f if x.strip()]
    if len(vals) != 5:
        raise ValueError("Cfg must have exactly 5 lines: Nsamp, M, N, Q, r")
    return tuple(vals)

class ChannelDataset(Dataset):
    """Per-sample Frobenius normalization ."""
    def __init__(self, cfg_path, x, y, sid, train=False, nz=0., dr=0., normalize: bool = True):
        super().__init__()
        _, self.M, self.N, self.Q, self.r = parse_cfg(cfg_path)
        self.x = self._to_complex(x)
        self.y = self._to_complex(y) if y is not None else None
        self.sid, self.train, self.nz, self.dr = sid, train, nz, dr
        self.normalize = normalize
        self.eps = 1e-8

    @staticmethod
    def _to_complex(a: np.ndarray) -> torch.Tensor:
        if a is None: return None
        if np.iscomplexobj(a): return torch.from_numpy(a).astype(np.complex64)
        if a.shape[-1] == 2:
            t = torch.from_numpy(a).float()
            return torch.view_as_complex(t)
        raise ValueError("Array must be complex or have a final dimension of 2.")

    def __len__(self): return self.x.size(0)

    def _augment(self, h: torch.Tensor) -> torch.Tensor:
        # antenna dropout
        if self.dr > 0 and torch.rand(1) < self.dr:
            h[torch.randint(0, h.size(0), (1,)), :] = 0
        if self.dr > 0 and torch.rand(1) < self.dr:
            h[:, torch.randint(0, h.size(1), (1,))] = 0
        # noise injection
        if self.nz > 0:
            p = torch.mean(torch.abs(h) ** 2)
            if p > 1e-12:
                sg = math.sqrt(p.item()) * self.nz / math.sqrt(2)
                n = (torch.randn_like(h.real) + 1j * torch.randn_like(h.real)) * sg
                h = h + n
        return h

    def __getitem__(self, i: int):
        h_raw = self.x[i].clone()
        scale = torch.linalg.matrix_norm(h_raw, ord='fro') + self.eps
        if self.normalize:
            h = h_raw / scale
            y = (self.y[i] if self.y is not None else h_raw).clone() / scale
        else:
            h = h_raw
            y = (self.y[i] if self.y is not None else h_raw).clone()
        h = self._augment(h) if self.train else h
        return h, y, self.sid, scale.float()

# -------------------- Backbone Blocks --------------------
class GatedConv1D(nn.Module):
    """Depthwise-separable gated conv (低MACs)."""
    def __init__(self, dim: int, k: int = 5):
        super().__init__()
        self.dim = dim
        self.dw = nn.Conv1d(dim, dim * 2, k, groups=dim, padding=k // 2, bias=True)
        self.pw = nn.Conv1d(dim, dim, 1, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        y = self.dw(x)
        a, g = y[:, :self.dim, :], y[:, self.dim:, :]
        y = a * torch.sigmoid(g)
        y = self.pw(y)
        return y.permute(0, 2, 1)

class GroupedProjectedAttention(nn.Module):
    """ O(T * k_len)."""
    def __init__(self, dim: int, att_dim: int, groups: int, seq_len: int, k_len: int):
        super().__init__()
        assert dim % groups == 0
        self.groups = groups
        self.gdim = dim // groups
        self.att_dim = att_dim
        self.k_len = k_len
        self.scale = 1.0 / math.sqrt(att_dim)

        self.qs = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.ks = nn.ModuleList([nn.Linear(self.gdim, att_dim, bias=False) for _ in range(groups)])
        self.vs = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])
        self.os = nn.ModuleList([nn.Linear(self.gdim, self.gdim, bias=False) for _ in range(groups)])

        
        self.Pk = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                    for _ in range(groups)])
        self.Pv = nn.ParameterList([nn.Parameter(torch.randn(seq_len, k_len) * (1.0 / math.sqrt(seq_len)))
                                    for _ in range(groups)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        G = self.groups
        xg = x.view(B, T, G, self.gdim).permute(0, 2, 1, 3)
        outs = []
        for g in range(G):
            q = self.qs[g](xg[:, g])                       # (B,T,att_dim)
            k = self.ks[g](xg[:, g])                       # (B,T,att_dim)
            v = self.vs[g](xg[:, g])                       # (B,T,gdim)
            k_red = torch.einsum('bta,tk->bka', k, self.Pk[g])  # (B,k_len,att_dim)
            v_red = torch.einsum('btg,tk->bkg', v, self.Pv[g])  # (B,k_len,gdim)
            att = torch.matmul(q, k_red.transpose(1, 2)) * self.scale
            att = torch.softmax(att, dim=-1)
            out = torch.matmul(att, v_red)                 # (B,T,gdim)
            outs.append(self.os[g](out))
        return torch.cat(outs, dim=-1)

# -------------------- Axial Low-Rank Frequency Gate --------------------
class AxialLowRankFreqGate(nn.Module):
    def __init__(self, M: int, N: int, hidden: int = 32, temperature: float = 0.2):
        super().__init__()
        self.M, self.N, self.T = M, N, temperature
        self.hidden = hidden
        self.row_mlp = nn.Sequential(
            nn.Linear(M, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, M, bias=False)
        )
        self.col_mlp = nn.Sequential(
            nn.Linear(N, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, N, bias=False)
        )
    def forward(self, Hf: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(Hf)
        r_stat = mag.mean(dim=2)  # (B,M)
        c_stat = mag.mean(dim=1)  # (B,N)
        r_w = self.row_mlp(r_stat)
        c_w = self.col_mlp(c_stat)
        gate = torch.sigmoid((r_w.unsqueeze(2) + c_w.unsqueeze(1)) / self.T)
        return gate

# -------------------- NEW: Neural Ortho Refiner  --------------------
class NeuralOrthoRefiner(nn.Module):
    """One-step learned orthogonality refinement (no QR/SVD; rule-compliant).
    U' = U - a * U * sym(U^H U - I);  V' = V - b * V * sym(V^H V - I)
    a,b in (0, 0.5] learned via sigmoid.
    """
    def __init__(self, init_scale: float = 0.1, max_scale: float = 0.5):
        super().__init__()
        self.max_scale = float(max_scale)
        self.a_raw = nn.Parameter(torch.tensor(math.log(init_scale/(max_scale-init_scale))))
        self.b_raw = nn.Parameter(torch.tensor(math.log(init_scale/(max_scale-init_scale))))
    def _scale(self, p):
        return torch.sigmoid(p) * self.max_scale
    def _sym(self, G):
        return 0.5 * (G + G.conj().transpose(-2, -1))
    def forward(self, U: torch.Tensor, V: torch.Tensor):
        a = self._scale(self.a_raw)
        b = self._scale(self.b_raw)
        Br = U.shape[-1]
        I = torch.eye(Br, device=U.device, dtype=U.dtype)
        Gu = U.conj().transpose(-2, -1) @ U
        Gv = V.conj().transpose(-2, -1) @ V
        Du = self._sym(Gu - I)
        Dv = self._sym(Gv - I)
        U2 = U - a * (U @ Du)
        V2 = V - b * (V @ Dv)
        # column normalization (allowed by rules)
        U2 = U2 / (torch.linalg.norm(U2, dim=1, keepdim=True) + 1e-8)
        V2 = V2 / (torch.linalg.norm(V2, dim=1, keepdim=True) + 1e-8)
        return U2, V2

# -------------------- SVD Predictor  --------------------
class LowMACS_SVDNet(nn.Module):
    def __init__(self, M: int, N: int, r: int, n_scene: int,
                 dim: int = 64, depth: int = 2, groups: int = 4,
                 kernel_size: int = 3, temperature: float = 0.2,
                 k_len: int = 32, tau_s: float = 0.9, gate_hidden: int = 32):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        self.temperature = temperature
        self.tau_s = tau_s
        self.k_len = k_len
        self.gate_hidden = gate_hidden
        self.groups = groups
        self.depth = depth
        self.dim = dim

        # (Step-1) axial low-rank frequency gate (prunable)
        self.freq_gate = AxialLowRankFreqGate(M, N, hidden=gate_hidden, temperature=temperature)

        # backbone
        self.input_proj = nn.Linear(4 * N, dim)
        self.pos = nn.Parameter(torch.zeros(1, M, dim))
        self.scene_emb = nn.Embedding(n_scene, dim)

        att_dim = min(r, 32)
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    GroupedProjectedAttention(dim, att_dim, groups, seq_len=M, k_len=k_len)
                ))
            else:
                blocks.append(nn.Sequential(nn.LayerNorm(dim), GatedConv1D(dim, kernel_size)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

        # heads
        self.u_head = nn.Linear(dim, 2 * M * r)
        self.v_head = nn.Linear(dim, 2 * N * r)
        self.s_head = nn.Sequential(nn.Linear(dim, r), nn.Softplus())

        # (Step-2) neural orthogonal refinement
        self.ortho_refine = NeuralOrthoRefiner(init_scale=0.1, max_scale=0.5)

        self.apply(self._init)
        with torch.no_grad():
            if isinstance(self.s_head[0], nn.Linear):
                self.s_head[0].bias.fill_(-5.0)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None: nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_tau_s(self, tau: float): self.tau_s = float(max(0.0, min(1.0, tau)))

    def forward(self, H, sid):
        B = H.size(0)
        if H.dim() == 4 and H.shape[-1] == 2:
            H = torch.view_as_complex(H.to(torch.float32))

        # frequency gating
        Hf = torch.fft.fft2(H, norm='ortho')
        gate = self.freq_gate(Hf)
        Hf_gated = Hf * gate
        Hd = torch.fft.ifft2(Hf_gated, norm='ortho')

        # dual representation
        Hr, Hi = Hd.real, Hd.imag
        rc = nn.functional.avg_pool2d(Hr.unsqueeze(1), 2).squeeze(1)
        ic = nn.functional.avg_pool2d(Hi.unsqueeze(1), 2).squeeze(1)
        rc = nn.functional.interpolate(rc.unsqueeze(1), size=(self.M, self.N), mode='bilinear', align_corners=False).squeeze(1)
        ic = nn.functional.interpolate(ic.unsqueeze(1), size=(self.M, self.N), mode='bilinear', align_corners=False).squeeze(1)
        Hc = rc + 1j * ic

        f1 = torch.view_as_real(Hd).reshape(B, self.M, -1)   # (B,M,2N)
        f2 = torch.view_as_real(Hc).reshape(B, self.M, -1)   # (B,M,2N)
        x = self.input_proj(torch.cat([f1, f2], dim=-1)) + self.pos

        for blk in self.blocks:
            x = x + blk(x)
        feat = self.norm(x.mean(1)) + self.scene_emb(sid.to(x.device))

        # heads
        U = torch.view_as_complex(self.u_head(feat).view(B, self.M, self.r, 2))
        V = torch.view_as_complex(self.v_head(feat).view(B, self.N, self.r, 2))
        S_pred = self.s_head(feat) + 1e-6

        # column normalization + sorting
        U = U / (torch.linalg.norm(U, dim=1, keepdim=True) + 1e-8)
        V = V / (torch.linalg.norm(V, dim=1, keepdim=True) + 1e-8)
        S_pred, idx = torch.sort(S_pred, dim=-1, descending=True)
        idx_u = idx.unsqueeze(1).expand(-1, self.M, -1)
        idx_v = idx.unsqueeze(1).expand(-1, self.N, -1)
        U = U.gather(2, idx_u)
        V = V.gather(2, idx_v)

        # (Step-2) orthogonal refinement (no QR/SVD)
        U, V = self.ortho_refine(U, V)

        # spectral self-calibration
        M_ = U.conj().transpose(-2, -1) @ Hd @ V
        S_best = torch.abs(torch.diagonal(M_, dim1=-2, dim2=-1))
        S = (1.0 - self.tau_s) * S_pred + self.tau_s * S_best + 1e-6
        return U, S, V, Hd

    # ------------ Step-3: Export a structurally pruned model ------------
    @torch.no_grad()
    def export_pruned_model(self, keep_k_len: int, keep_gate_hidden: int):
        keep_k_len = int(max(8, min(self.k_len, keep_k_len)))
        keep_gate_hidden = int(max(8, min(self.gate_hidden, keep_gate_hidden)))

        # 1) new model with smaller k_len & gate hidden
        new_model = LowMACS_SVDNet(self.M, self.N, self.r,
                                   n_scene=self.scene_emb.num_embeddings,
                                   dim=self.dim, depth=self.depth, groups=self.groups,
                                   kernel_size=3, temperature=self.temperature,
                                   k_len=keep_k_len, tau_s=self.tau_s, gate_hidden=keep_gate_hidden)

        def copy_like(a, b):
            if a.shape == b.shape: b.data.copy_(a.data)

        # 2) shared weights
        copy_like(self.input_proj.weight, new_model.input_proj.weight)
        copy_like(self.input_proj.bias,   new_model.input_proj.bias)
        new_model.pos.data.copy_(self.pos.data)
        new_model.scene_emb.weight.data.copy_(self.scene_emb.weight.data)

        # 3) blocks (LN + GPA / GatedConv)
        for blk_old, blk_new in zip(self.blocks, new_model.blocks):
            mods_old = list(blk_old.children())
            mods_new = list(blk_new.children())
            # LayerNorm
            copy_like(mods_old[0].weight, mods_new[0].weight)
            copy_like(mods_old[0].bias,   mods_new[0].bias)
            if isinstance(mods_old[1], GroupedProjectedAttention):
                gpa_old: GroupedProjectedAttention = mods_old[1]
                gpa_new: GroupedProjectedAttention = mods_new[1]
                # Q/K/V/O identical
                for (qo, qn) in zip(gpa_old.qs, gpa_new.qs): copy_like(qo.weight, qn.weight)
                for (ko, kn) in zip(gpa_old.ks, gpa_new.ks): copy_like(ko.weight, kn.weight)
                for (vo, vn) in zip(gpa_old.vs, gpa_new.vs): copy_like(vo.weight, vn.weight)
                for (oo, on) in zip(gpa_old.os, gpa_new.os): copy_like(oo.weight, on.weight)
                # column selection for Pk/Pv by energy
                alpha = 1.0
                for g in range(gpa_old.groups):
                    Pk = gpa_old.Pk[g].data
                    Pv = gpa_old.Pv[g].data
                    score = (Pk.pow(2).sum(dim=0) + alpha * Pv.pow(2).sum(dim=0)).cpu().numpy()
                    keep_idx = np.argsort(-score)[:keep_k_len]
                    keep_idx = np.sort(keep_idx)
                    gpa_new.Pk[g].data.copy_(Pk[:, keep_idx])
                    gpa_new.Pv[g].data.copy_(Pv[:, keep_idx])
            else:
                gc_old: GatedConv1D = mods_old[1]
                gc_new: GatedConv1D = mods_new[1]
                copy_like(gc_old.dw.weight, gc_new.dw.weight); copy_like(gc_old.dw.bias, gc_new.dw.bias)
                copy_like(gc_old.pw.weight, gc_new.pw.weight); copy_like(gc_old.pw.bias, gc_new.pw.bias)

        # 4) heads
        copy_like(self.u_head.weight, new_model.u_head.weight); copy_like(self.u_head.bias, new_model.u_head.bias)
        copy_like(self.v_head.weight, new_model.v_head.weight); copy_like(self.v_head.bias, new_model.v_head.bias)
        copy_like(self.s_head[0].weight, new_model.s_head[0].weight); copy_like(self.s_head[0].bias, new_model.s_head[0].bias)

        # 5) freq gate hidden pruning
        if self.freq_gate.hidden == keep_gate_hidden:
            for i in [0,2]:
                copy_like(self.freq_gate.row_mlp[i].weight, new_model.freq_gate.row_mlp[i].weight)
                copy_like(self.freq_gate.col_mlp[i].weight, new_model.freq_gate.col_mlp[i].weight)
        else:
            W1r = self.freq_gate.row_mlp[0].weight.data      # (hidden, M)
            W2r = self.freq_gate.row_mlp[2].weight.data      # (M, hidden)
            W1c = self.freq_gate.col_mlp[0].weight.data      # (hidden, N)
            W2c = self.freq_gate.col_mlp[2].weight.data      # (N, hidden)
            s_row = W1r.pow(2).sum(dim=1).sqrt() * W2r.pow(2).sum(dim=0).sqrt()
            s_col = W1c.pow(2).sum(dim=1).sqrt() * W2c.pow(2).sum(dim=0).sqrt()
            score = (s_row + s_col).cpu().numpy()
            keep = np.argsort(-score)[:keep_gate_hidden]
            keep = np.sort(keep)
            new_model.freq_gate.row_mlp[0].weight.data.copy_(W1r[keep, :])
            new_model.freq_gate.col_mlp[0].weight.data.copy_(W1c[keep, :])
            new_model.freq_gate.row_mlp[2].weight.data.copy_(W2r[:, keep])
            new_model.freq_gate.col_mlp[2].weight.data.copy_(W2c[:, keep])

        # 6) copy ortho_refine params
        new_model.ortho_refine.a_raw.data.copy_(self.ortho_refine.a_raw.data)
        new_model.ortho_refine.b_raw.data.copy_(self.ortho_refine.b_raw.data)

        return new_model

# -------------------- Metric & Loss --------------------
class OfficialMetric:
    @staticmethod
    def ae(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        device = H.device
        Sigma = torch.diag_embed(S.float()).to(device).to(torch.cfloat)
        H_hat = U @ Sigma @ V.conj().transpose(-2, -1)
        rel_err = torch.linalg.matrix_norm(H - H_hat, ord="fro", dim=(-2, -1)) / \
                  (torch.linalg.matrix_norm(H, ord="fro", dim=(-2, -1)) + 1e-8)
        r = S.size(-1)
        I = torch.eye(r, device=device, dtype=H.dtype)
        ortho_u_err = torch.linalg.matrix_norm(U.conj().transpose(-2, -1) @ U - I, ord="fro", dim=(-2, -1))
        ortho_v_err = torch.linalg.matrix_norm(V.conj().transpose(-2, -1) @ V - I, ord="fro", dim=(-2, -1))
        return (rel_err + ortho_u_err + ortho_v_err).mean()

class AEPlusLoss(nn.Module):
    def forward(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor,
                H_gt: torch.Tensor, wE: float, wD: float, wS: float, lam: float) -> torch.Tensor:
        device = H_gt.device
        Sigma = torch.diag_embed(S.float()).to(device).to(torch.cfloat)
        H_hat = U @ Sigma @ V.conj().transpose(-2, -1)
        normH = torch.linalg.matrix_norm(H_gt, ord="fro", dim=(-2, -1)) + 1e-8
        L_rec = torch.linalg.matrix_norm(H_gt - H_hat, ord="fro", dim=(-2, -1)) / normH
        r = S.size(-1)
        I = torch.eye(r, device=device, dtype=H_gt.dtype)
        L_ortho = torch.linalg.matrix_norm(U.conj().transpose(-2, -1) @ U - I, ord="fro", dim=(-2, -1)) + \
                  torch.linalg.matrix_norm(V.conj().transpose(-2, -1) @ V - I, ord="fro", dim=(-2, -1))
        M = U.conj().transpose(-2, -1) @ H_gt @ V
        diagM = torch.diagonal(M, dim1=-2, dim2=-1); offM = M - torch.diag_embed(diagM)
        L_energy = 1.0 - (torch.linalg.matrix_norm(M, ord="fro", dim=(-2, -1)) / normH)
        L_diag   = torch.linalg.matrix_norm(offM, ord="fro", dim=(-2, -1)) / normH
        Sm = torch.abs(diagM)
        L_smatch = torch.mean(torch.abs((S / normH.unsqueeze(-1)) - (Sm / normH.unsqueeze(-1))), dim=-1)
        return (L_rec + lam * L_ortho + wE * L_energy + wD * L_diag + wS * L_smatch).mean()

# -------------------- Complexity (C) --------------------
def get_avg_flops(model: nn.Module, input_data: torch.Tensor) -> float:
    """Average MACs/sample using PyTorch Profiler """
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")
    b = input_data.size(0)
    model = model.eval().cpu()
    input_data = input_data.cpu()
    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            sid = torch.zeros(x.shape[0], dtype=torch.long)
            U,S,V,_ = self.m(x, sid)
            return U,S,V
    wrap = _Wrap(model)
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
            wrap(input_data)
    total_flops = sum(e.flops for e in prof.events())
    avg_flops = total_flops / max(1, b)
    return avg_flops * 1e-6 / 2  # FLOPs -> Mega MACs

# -------------------- Train / Prune / Finetune / Infer --------------------
class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        os.makedirs(cfg["CKPT_DIR"], exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _make_loaders(self) -> Tuple[DataLoader, DataLoader]:
        tr_sets, val_sets = [], []
        scen2idx = {s: i for i, s in enumerate(self.cfg["SCENARIOS"])}
        round_num = self.cfg["ROUND_NUM"]
        cfg0 = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}CfgData{self.cfg['SCENARIOS'][0]}.txt")
        _, self.M, self.N, self.Q, self.r = parse_cfg(cfg0)
        print("Loading and splitting data...")
        for s in self.cfg["SCENARIOS"]:
            cfg_p = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}CfgData{s}.txt")
            x_p = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}TrainData{s}.npy")
            y_p = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}TrainLabel{s}.npy")
            X, Y = np.load(x_p), np.load(y_p)
            ids = np.random.permutation(len(X)); cut = int(len(X) * (1 - self.cfg["VAL_SPLIT"]))
            tr_sets.append(ChannelDataset(cfg_p, X[ids[:cut]], Y[ids[:cut]], scen2idx[s],
                                          train=True, nz=self.cfg["NOISE"], dr=self.cfg["DROPOUT"], normalize=True))
            val_sets.append(ChannelDataset(cfg_p, X[ids[cut:]], Y[ids[cut:]], scen2idx[s],
                                           train=False, normalize=True))
        tr_loader = DataLoader(ConcatDataset(tr_sets), batch_size=self.cfg["BATCH"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(ConcatDataset(val_sets), batch_size=self.cfg["BATCH"], shuffle=False, num_workers=4, pin_memory=True)
        return tr_loader, val_loader

    def _sched_weights(self, ep: int, E: int):
        lam = float(np.interp(ep, [0, self.cfg["LAM_RAMP"]], [self.cfg["LAM0"], self.cfg["LAM1"]]))
        def decay(w0):
            p1, p2 = int(0.4 * E), int(0.9 * E)
            if ep <= p1: return w0
            if ep >= p2: return 0.05 * w0
            t = (ep - p1) / max(1, (p2 - p1)); return (1 - t) * w0 + t * (0.2 * w0)
        wE = decay(self.cfg["W_ENERGY"]); wD = decay(self.cfg["W_DIAG"]); wS = decay(self.cfg["W_SMATCH"])
        tau = float(np.interp(ep, [0, E], [self.cfg["TAU_S_START"], self.cfg["TAU_S_END"]]))
        return wE, wD, wS, lam, tau

    def _eval_valAE(self, model: nn.Module, val_loader: DataLoader) -> float:
        model.eval(); total = 0.0
        with torch.no_grad():
            for H, H_gt, sid, _ in val_loader:
                H, H_gt, sid = H.to(self.device), H_gt.to(self.device), sid.to(self.device)
                U, S, V, _ = model(H, sid)
                total += OfficialMetric.ae(U, S, V, H_gt).item() * H.size(0)
        return total / len(val_loader.dataset)

    def _finetune(self, model: nn.Module, tr_loader: DataLoader, val_loader: DataLoader,
                  epochs: int, lr: float):
        criterion = AEPlusLoss()
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.cfg["WD"])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
        best = float('inf')
        for ep in range(epochs):
            wE, wD, wS, lam, tau = self._sched_weights(self.cfg["EPOCHS"] + ep, self.cfg["EPOCHS"] + epochs)
            model.set_tau_s(tau)
            model.train()
            for H, H_gt, sid, _ in tqdm(tr_loader, desc=f"[Finetune] Ep {ep+1}/{epochs}"):
                H, H_gt, sid = H.to(self.device), H_gt.to(self.device), sid.to(self.device)
                opt.zero_grad(set_to_none=True)
                U, S, V, _ = model(H, sid)
                loss = criterion(U, S, V, H_gt, wE, wD, wS, lam)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), self.cfg["CLIP"]); opt.step()
            sch.step()
            vae = self._eval_valAE(model, val_loader)
            best = min(best, vae); print(f"[Finetune] val_AE={vae:.4f} (best {best:.4f})")

    def run(self) -> Tuple[str, float]:
        tr_loader, val_loader = self._make_loaders()
        # --- Train step1+step2 baseline ---
        base = LowMACS_SVDNet(self.M, self.N, self.r, len(self.cfg["SCENARIOS"]),
                              dim=self.cfg["DIM"], depth=self.cfg["DEPTH"], groups=self.cfg["GROUPS"],
                              kernel_size=self.cfg["KERNEL"], temperature=self.cfg["TEMPERATURE"],
                              k_len=self.cfg["K_LEN"], tau_s=self.cfg["TAU_S_START"], gate_hidden=self.cfg["GATE_HIDDEN"])

        # unified MACs
        sample_H, _, _, _ = next(iter(tr_loader))
        macs = get_avg_flops(base, sample_H[:min(sample_H.size(0), 8)])
        print(f"[Init] MACs={macs:.4f} M")

        base = base.to(self.device)
        criterion = AEPlusLoss()
        opt = optim.AdamW(base.parameters(), lr=self.cfg["LR"], weight_decay=self.cfg["WD"])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg["EPOCHS"], 1e-6)

        best_score = float('inf'); best_ckpt = os.path.join(self.cfg["CKPT_DIR"], "best_preprune.pth")
        for ep in range(self.cfg["EPOCHS"]):
            wE, wD, wS, lam, tau = self._sched_weights(ep, self.cfg["EPOCHS"]); base.set_tau_s(tau)
            base.train()
            pbar = tqdm(tr_loader, desc=f"Ep {ep+1}/{self.cfg['EPOCHS']} λ={lam:.3f} τ={tau:.2f} wE={wE:.2f} wD={wD:.2f} wS={wS:.2f}")
            for H, H_gt, sid, _ in pbar:
                H, H_gt, sid = H.to(self.device), H_gt.to(self.device), sid.to(self.device)
                opt.zero_grad(set_to_none=True)
                U, S, V, _ = base(H, sid)
                loss = criterion(U, S, V, H_gt, wE, wD, wS, lam)
                loss.backward(); nn.utils.clip_grad_norm_(base.parameters(), self.cfg["CLIP"]); opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            sch.step()
            vae = self._eval_valAE(base, val_loader); score = 100 * vae + macs
            print(f" >> val_AE={vae:.4f} | MACs={macs:.4f} | Score={score:.4f}")
            if score < best_score:
                best_score = score
                torch.save(base.state_dict(), best_ckpt)
                print(f"    (New best pre-prune score! Saved to {best_ckpt})")

        # --- Step-3: structured pruning ---
        print("\n[Pruning] Exporting pruned model...")
        base.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
        base = base.cpu()
        pruned = base.export_pruned_model(self.cfg["PRUNE_KEEP_KLEN"], self.cfg["PRUNE_KEEP_HIDDEN"])

        # MACs after pruning
        pruned_macs = get_avg_flops(pruned, sample_H[:min(sample_H.size(0), 8)])
        print(f"[Pruned] MACs={pruned_macs:.4f} M (was {macs:.4f} M)")

        # --- Finetune ---
        pruned = pruned.to(self.device)
        self._finetune(pruned, tr_loader, val_loader, epochs=self.cfg["FT_EPOCHS"], lr=self.cfg["FT_LR"])
        final_ae = self._eval_valAE(pruned, val_loader)
        final_score = 100 * final_ae + pruned_macs
        print(f"[Final] val_AE={final_ae:.4f} | MACs={pruned_macs:.4f} | Score={final_score:.4f}")

        # save pruned ckpt with arch
        final_ckpt = os.path.join(self.cfg["CKPT_DIR"], "best.pth")
        torch.save({"state_dict": pruned.state_dict(),
                    "arch": {"K_LEN": pruned.k_len, "GATE_HIDDEN": pruned.gate_hidden}}, final_ckpt)
        print(f"(Final model saved to {final_ckpt})")
        return final_ckpt, pruned_macs

class PruneFineTuneRunner:
    """Load external step1+2 ckpt -> prune (k_len & gate_hidden) -> finetune -> save."""
    def __init__(self, cfg: Dict[str, Any], src_ckpt: str,
                 keep_klen: int, keep_hidden: int,
                 ft_epochs: int, ft_lr: float):
        self.cfg = cfg
        self.src_ckpt = src_ckpt
        self.keep_klen = keep_klen
        self.keep_hidden = keep_hidden
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        trainer = Trainer(self.cfg)
        tr_loader, val_loader = trainer._make_loaders()
        # build baseline skeleton (step1+2)
        base = LowMACS_SVDNet(trainer.M, trainer.N, trainer.r,
                              n_scene=len(self.cfg["SCENARIOS"]),
                              dim=self.cfg["DIM"], depth=self.cfg["DEPTH"], groups=self.cfg["GROUPS"],
                              kernel_size=self.cfg["KERNEL"], temperature=self.cfg["TEMPERATURE"],
                              k_len=self.cfg["K_LEN"], tau_s=self.cfg["TAU_S_START"], gate_hidden=self.cfg["GATE_HIDDEN"])
        payload = torch.load(self.src_ckpt, map_location="cpu")
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        missing, unexpected = base.load_state_dict(state, strict=False)
        if len(unexpected) > 0: print(f"[Warn] Unexpected keys: {unexpected}")
        if len(missing) > 0:    print(f"[Warn] Missing keys filled by init: {missing}")

        # base metrics
        sample_H, _, _, _ = next(iter(tr_loader))
        macs0 = get_avg_flops(base, sample_H[:min(sample_H.size(0), 8)])
        base_ae = trainer._eval_valAE(base.to(self.device).eval(), val_loader)
        print(f"[Base] val_AE={base_ae:.4f} | MACs={macs0:.4f} | Score={100*base_ae+macs0:.4f}")

        # prune
        print(f"[Pruning] keep_klen={self.keep_klen} keep_hidden={self.keep_hidden}")
        base = base.cpu()
        pruned = base.export_pruned_model(self.keep_klen, self.keep_hidden)
        macs1 = get_avg_flops(pruned, sample_H[:min(sample_H.size(0), 8)])
        print(f"[After Prune] MACs={macs1:.4f} M (was {macs0:.4f} M)")

        # finetune
        pruned = pruned.to(self.device)
        trainer._finetune(pruned, tr_loader, val_loader, epochs=self.ft_epochs, lr=self.ft_lr)
        final_ae = trainer._eval_valAE(pruned, val_loader)
        final_score = 100 * final_ae + macs1
        print(f"[Final] val_AE={final_ae:.4f} | MACs={macs1:.4f} | Score={final_score:.4f}")

        # save
        os.makedirs(self.cfg["CKPT_DIR"], exist_ok=True)
        out_ckpt = os.path.join(self.cfg["CKPT_DIR"], "best_pruned.pth")
        torch.save({"state_dict": pruned.state_dict(),
                    "arch": {"K_LEN": pruned.k_len, "GATE_HIDDEN": pruned.gate_hidden}}, out_ckpt)
        print(f"[Saved] {out_ckpt}")
        return out_ckpt, macs1

class Inferencer:
    def __init__(self, cfg: Dict[str, Any], ckpt_path: str):
        self.cfg = cfg
        round_num = self.cfg["ROUND_NUM"]
        cfg0 = os.path.join(cfg["DATA_DIR"], f"Round{round_num}CfgData{cfg['SCENARIOS'][0]}.txt")
        _, self.M, self.N, self.Q, self.r = parse_cfg(cfg0)

        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload and "arch" in payload:
            k_len = int(payload["arch"].get("K_LEN", cfg["K_LEN"]))
            gate_hidden = int(payload["arch"].get("GATE_HIDDEN", cfg["GATE_HIDDEN"]))
            state = payload["state_dict"]
        else:
            # fallback for non-pruned ckpt
            k_len = cfg["K_LEN"]; gate_hidden = cfg["GATE_HIDDEN"]; state = payload

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LowMACS_SVDNet(self.M, self.N, self.r, len(cfg["SCENARIOS"]),
                                    dim=self.cfg["DIM"], depth=self.cfg["DEPTH"], groups=self.cfg["GROUPS"],
                                    kernel_size=self.cfg["KERNEL"], temperature=self.cfg["TEMPERATURE"],
                                    k_len=k_len, tau_s=self.cfg["TAU_S_END"], gate_hidden=gate_hidden)
        self.model.load_state_dict(state)
        os.makedirs(cfg["OUT_DIR"], exist_ok=True)

        # compute C value consistently
        test_cfg = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}CfgData{self.cfg['SCENARIOS'][0]}.txt")
        test_np  = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}TestData{self.cfg['SCENARIOS'][0]}.npy")
        ds = ChannelDataset(test_cfg, np.load(test_np), None, 0, train=False, normalize=True)
        dl = DataLoader(ds, batch_size=self.cfg["BATCH"], shuffle=False)
        H, _, _, _ = next(iter(dl))
        self.macs = get_avg_flops(self.model, H[:min(H.size(0), 8)])
        self.model.to(self.device).eval()
        print(f"[Inference] MACs (C value) = {self.macs:.4f} M")

    def _save_npz(self, idx: str, U: np.ndarray, S: np.ndarray, V: np.ndarray):
        out_p = os.path.join(self.cfg["OUT_DIR"], f"{idx}.npz")
        U_out = np.stack((U.real, U.imag), axis=-1)
        V_out = np.stack((V.real, V.imag), axis=-1)
        np.savez(out_p, U=U_out.astype(np.float32), S=S.astype(np.float32), V=V_out.astype(np.float32), C=float(self.macs))
        print(f"Saved {out_p}  |  U{U_out.shape}  S{S.shape}  V{V_out.shape}")

    def run(self):
        round_num = self.cfg["ROUND_NUM"]
        scen2idx = {s: i for i, s in enumerate(self.cfg["SCENARIOS"])}
        for s_idx, s in enumerate(self.cfg["SCENARIOS"], start=1):
            cfg_p  = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}CfgData{s}.txt")
            test_p = os.path.join(self.cfg["DATA_DIR"], f"Round{round_num}TestData{s}.npy")
            X = np.load(test_p)
            ds = ChannelDataset(cfg_p, X, None, scen2idx[s], train=False, normalize=True)
            dl = DataLoader(ds, batch_size=self.cfg["BATCH"], shuffle=False)
            U_acc, S_acc, V_acc = [], [], []
            with torch.no_grad():
                for H, _, sid, scale in tqdm(dl, desc=f"Inferring Scenario {s}"):
                    H, sid, scale = H.to(self.device), sid.to(self.device), scale.to(self.device)
                    U, S, V, _ = self.model(H, sid)
                    S = S * scale.unsqueeze(-1)
                    U_acc.append(U.cpu().numpy()); S_acc.append(S.cpu().numpy()); V_acc.append(V.cpu().numpy())
            U_np = np.concatenate(U_acc, 0); S_np = np.concatenate(S_acc, 0); V_np = np.concatenate(V_acc, 0)
            self._save_npz(str(s_idx), U_np, S_np, V_np)

# -------------------- Config & Main --------------------

def make_cfg(args) -> Dict[str, Any]:
    return dict(
        # General
        ROUND_NUM=2,
        DATA_DIR=args.data_dir,
        CKPT_DIR="./ckpts_step3_s12",
        OUT_DIR=args.out_dir,
        SCENARIOS=["1", "2", "3", "4"],

        # Pretrain 
        BATCH=256,
        LR=4e-4,
        WD=1e-4,
        EPOCHS=400,         
        VAL_SPLIT=0.1,
        CLIP=1.0,
        NOISE=0.03,
        DROPOUT=0.1,

        # Loss schedules
        LAM0=0.0, LAM1=0.35, LAM_RAMP=220,
        W_ENERGY=0.40, W_DIAG=0.30, W_SMATCH=0.30,

        # Architecture (step-1/2)
        DIM=64, DEPTH=2, GROUPS=4, KERNEL=3,
        TEMPERATURE=0.2, K_LEN=32, GATE_HIDDEN=32,

        # Pruning targets (defaults)
        PRUNE_KEEP_KLEN=args.keep_klen if hasattr(args, 'keep_klen') else 28,
        PRUNE_KEEP_HIDDEN=args.keep_hidden if hasattr(args, 'keep_hidden') else 28,

        # Finetune after pruning 
        FT_EPOCHS=args.ft_epochs if hasattr(args, 'ft_epochs') else 60,
        FT_LR=args.ft_lr if hasattr(args, 'ft_lr') else 2e-4,

        # τ schedule
        TAU_S_START=0.90, TAU_S_END=0.60
    )

def main():
    ap = argparse.ArgumentParser(description="Unified (Step1+2 baseline) + Step3: Train / Prune+Finetune / Infer")
    ap.add_argument("--mode", choices=["train", "infer", "prune_ft"], required=True)
    ap.add_argument("--data_dir", default="./data2")
    ap.add_argument("--ckpt", default="./ckpts_step3_s12/best.pth", help="for --mode infer")
    ap.add_argument("--out_dir", default="./submission_step3_s12")
    # ↓↓↓ used for train and prune_ft 
    ap.add_argument("--keep_klen", type=int, default=28)
    ap.add_argument("--keep_hidden", type=int, default=28)
    ap.add_argument("--ft_epochs", type=int, default=60)
    ap.add_argument("--ft_lr", type=float, default=2e-4)
    # ↓↓↓ only prune_ft 
    ap.add_argument("--src_ckpt", default="", help="path to your existing best ckpt (step-1+2)")
    args = ap.parse_args()

    cfg = make_cfg(args)

    if args.mode == "train":
        trainer = Trainer(cfg)
        final_ckpt, macs = trainer.run()
        print(f"\nTraining finished! Final @ {final_ckpt} | MACs={macs:.4f}M")
    elif args.mode == "prune_ft":
        if not os.path.exists(args.src_ckpt):
            raise FileNotFoundError(f"src_ckpt not found: {args.src_ckpt}")
        runner = PruneFineTuneRunner(cfg, args.src_ckpt,
                                     keep_klen=args.keep_klen,
                                     keep_hidden=args.keep_hidden,
                                     ft_epochs=args.ft_epochs,
                                     ft_lr=args.ft_lr)
        out_ckpt, macs = runner.run()
        print(f"\nPrune+Finetune finished! Final @ {out_ckpt} | MACs={macs:.4f}M")
    else:  # infer
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Train/prune_ft first or pass a valid path.")
        infer = Inferencer(cfg, args.ckpt)
        infer.run()
        import shutil
        os.makedirs(cfg["OUT_DIR"], exist_ok=True)
        shutil.copy(__file__, os.path.join(cfg["OUT_DIR"], "model.py"))
        print(f"\nInference complete. Results and model files are in '{cfg['OUT_DIR']}'.")

if __name__ == "__main__":
    main()
