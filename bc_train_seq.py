# -*- coding: utf-8 -*-
"""
Sequence Behavior Cloning with memory (GRU/LSTM) for obs64 -> act16.
- 以“每个 ep_XXXX.csv”为一段序列，按 t 升序拼接
- 输入特征 = [obs_*] + 时间特征 [t_norm, sin(2π t_norm), cos(2π t_norm)]
- 模型 = RNN( GRU/LSTM ) + 双头 (cont/bin)，逐时刻监督
- 保存 best: seq_policy.pt + seq_meta.json
"""
from datetime import datetime
import os, glob, json, argparse, random, math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pad_and_pack(batch, device):
    # batch: list of (X[T, Din], Yc[T, Cc] or None, Yb[T, Cb] or None, mask[T])
    lengths = [x[0].shape[0] for x in batch]
    T_max = max(lengths)
    Xs, Ycs, Ybs, Ms = [], [], [], []
    for X, Yc, Yb, M in batch:
        pad = T_max - X.shape[0]
        if pad > 0:
            X = torch.cat([X, torch.zeros(pad, X.shape[1])], dim=0)
            if Yc is not None: Yc = torch.cat([Yc, torch.zeros(pad, Yc.shape[1])], dim=0)
            if Yb is not None: Yb = torch.cat([Yb, torch.zeros(pad, Yb.shape[1])], dim=0)
            M = torch.cat([M, torch.zeros(pad)], dim=0)
        Xs.append(X); Ms.append(M)
        Ycs.append(Yc if Yc is not None else torch.zeros(T_max, 1))
        Ybs.append(Yb if Yb is not None else torch.zeros(T_max, 1))
    X = torch.stack(Xs, 0).to(device)           # B,T,Din
    M = torch.stack(Ms, 0).to(device)           # B,T
    Yc = torch.stack(Ycs, 0).to(device)         # B,T,Cc (可能为 size=1 的占位)
    Yb = torch.stack(Ybs, 0).to(device)         # B,T,Cb (可能为 size=1 的占位)
    return X, Yc, Yb, M, torch.tensor(lengths, device=device)

def parse_hidden(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [256,256]

# ---------------- dataset ----------------
class SeqDataset(Dataset):
    """
    把每个 ep_*.csv 作为一个序列样本，按 t 升序。
    输入：X[t] = concat(obs[t], time_feats[t])
    监督：Y_cont[t], Y_bin[t]（逐时刻）
    """
    def __init__(self, files: List[str], time_feat="norm+sin2pi+cos2pi"):
        assert files, "No files matched."
        self.seqs = []           # list of dict: df, obs_cols, act_cols, t_max
        self.time_feat = time_feat

        # 先读出全部文件，检查列
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if len(df) == 0: continue
                if "t" not in df.columns: raise RuntimeError(f"{f} has no 't' column")
                df = df.sort_values("t").reset_index(drop=True)
                df["__src__"] = f
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] read {f} failed: {e}")
        assert dfs, "All files failed or empty."

        # 列集合
        any_df = dfs[0]
        obs_cols = [c for c in any_df.columns if c.startswith("obs_")]
        act_cols = [c for c in any_df.columns if c.startswith("act_")]
        assert obs_cols and act_cols, "Need obs_* and act_* columns."

        # 动作划分
        def is_binary(col: str, df_list: List[pd.DataFrame]) -> bool:
            vals = np.concatenate([d[col].dropna().values for d in df_list])
            uniq = np.unique(vals)
            return np.all(np.isin(uniq, [0,1]))
        bin_cols = [c for c in act_cols if is_binary(c, dfs)]
        cont_cols = [c for c in act_cols if c not in bin_cols]

        # 统计量（obs 和 cont 归一化）
        all_obs = np.concatenate([d[obs_cols].values for d in dfs], axis=0).astype(np.float32)
        obs_mean = all_obs.mean(0); obs_std = all_obs.std(0); obs_std[obs_std==0] = 1.0

        if cont_cols:
            all_cont = np.concatenate([d[cont_cols].values for d in dfs], axis=0).astype(np.float32)
            cont_mean = all_cont.mean(0); cont_std = all_cont.std(0); cont_std[cont_std==0] = 1.0
        else:
            cont_mean = np.zeros(1, np.float32); cont_std = np.ones(1, np.float32)

        # 构建每个序列缓存（numpy，训练更快）
        self.obs_cols = obs_cols
        self.act_cols = act_cols
        self.cont_cols = cont_cols
        self.bin_cols = bin_cols
        self.obs_mean = obs_mean.tolist()
        self.obs_std  = obs_std.tolist()
        self.cont_mean = {c: float(cont_mean[i]) for i,c in enumerate(cont_cols)}
        self.cont_std  = {c: float(cont_std[i])  for i,c in enumerate(cont_cols)}

        for df in dfs:
            arr_obs = df[obs_cols].values.astype(np.float32)  # T, Dobs
            # 归一化 obs
            arr_obs = (arr_obs - obs_mean) / obs_std

            # 时间特征
            t = df["t"].values.astype(np.float32)
            t0, t1 = float(t.min()), float(t.max())
            Tspan = max(1.0, t1 - t0)
            t_norm = (t - t0) / Tspan
            feats = [t_norm]
            if "sin2pi" in time_feat or "norm+sin2pi+cos2pi" in time_feat:
                feats.append(np.sin(2*np.pi*t_norm))
            if "cos2pi" in time_feat or "norm+sin2pi+cos2pi" in time_feat:
                feats.append(np.cos(2*np.pi*t_norm))
            t_feat = np.stack(feats, axis=1).astype(np.float32)  # T, Dtime
            X = np.concatenate([arr_obs, t_feat], axis=1)        # T, Din

            # 目标
            Yc = None; Yb = None
            if cont_cols:
                Vc = df[cont_cols].values.astype(np.float32)
                for i,c in enumerate(cont_cols):
                    mu = cont_mean[i]; sd = cont_std[i]
                    if sd == 0: sd = 1.0
                    Vc[:, i] = (Vc[:, i] - mu) / sd
                Yc = Vc
            if bin_cols:
                Vb = df[bin_cols].values.astype(np.float32)
                Yb = Vb

            self.seqs.append({
                "X": X, "Yc": Yc, "Yb": Yb,
                "t_feat_dim": t_feat.shape[1],
                "src": df["__src__"].iloc[0],
                "T": X.shape[0]
            })

        # 记录 time_feat 维度
        self.t_feat_dim = self.seqs[0]["t_feat_dim"]
        self.in_dim = len(self.obs_cols) + self.t_feat_dim
        self.cont_dim = len(self.cont_cols)
        self.bin_dim = len(self.bin_cols)

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        X = torch.from_numpy(s["X"])                          # T,Din
        Yc = torch.from_numpy(s["Yc"]) if s["Yc"] is not None else None  # T,Cc
        Yb = torch.from_numpy(s["Yb"]) if s["Yb"] is not None else None  # T,Cb
        M  = torch.ones(X.shape[0])                           # T, 1=有效
        return X, Yc, Yb, M

# ---------------- model ----------------
class RNNPolicy(nn.Module):
    def __init__(self, in_dim, cont_dim, bin_dim,
                 rnn_type="gru", hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn_type = rnn_type.lower()
        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
        )
        self.head_cont = nn.Linear(hidden_size, cont_dim) if cont_dim>0 else None
        self.head_bin  = nn.Linear(hidden_size, bin_dim)  if bin_dim>0  else None

    def forward(self, X, h=None):
        # X: B,T,D
        Z, h_new = self.rnn(X, h)  # Z: B,T,H
        out = {}
        if self.head_cont is not None:
            out["cont"] = self.head_cont(Z)          # B,T,Cc (normalized space)
        if self.head_bin is not None:
            out["bin_logits"] = self.head_bin(Z)     # B,T,Cb
        return out, h_new

# ---------------- train ----------------训练命令：python bc_train_seq.py --data_glob "runs/20250913_003049/*.csv"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True)
    ap.add_argument("--out_dir", default="./bc_out_seq")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    # RNN
    ap.add_argument("--rnn_type", choices=["gru","lstm"], default="gru")
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    # loss/regularization
    ap.add_argument("--loss_w_cont", type=float, default=1.0)
    ap.add_argument("--loss_w_bin", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=0.0)
    ap.add_argument("--smooth_lambda", type=float, default=0.0, help="时序平滑：惩罚 Δa_t")
    # misc
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    files = sorted(glob.glob(args.data_glob))
    if not files and os.path.isfile(args.data_glob):
        files = [args.data_glob]
    assert files, f"No files matched: {args.data_glob}"

    # 用时间戳创建唯一输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(args.out_dir, exist_ok=True)

    # 划分 train/val（按文件划分，避免泄露）
    n = len(files); n_val = max(1, int(n*args.val_ratio))
    random.shuffle(files)
    val_files = files[:n_val]; train_files = files[n_val:]

    train_ds = SeqDataset(train_files)
    val_ds   = SeqDataset(val_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: pad_and_pack(b, device))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: pad_and_pack(b, device))

    model = RNNPolicy(
        in_dim=train_ds.in_dim,
        cont_dim=train_ds.cont_dim,
        bin_dim=train_ds.bin_dim,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    pos_weight = None
    if train_ds.bin_dim>0 and args.pos_weight>0:
        pos_weight = torch.full((train_ds.bin_dim,), float(args.pos_weight), device=device)

    best = float("inf"); best_path = os.path.join(args.out_dir, "seq_policy.pt")
    no_imp = 0

    def step_epoch(dloader, train=True):
        if train: model.train()
        else: model.eval()
        loss_sum=0.0; steps=0
        cont_mse=0.0; bce_sum=0.0; acc_sum=0.0; n_cont=0; n_bin=0
        for X, Yc, Yb, M, L in dloader:
            with torch.amp.autocast("cuda", enabled=args.amp):
                out,_ = model(X)
                loss = 0.0
                if train_ds.cont_dim>0:
                    # MSE（只在有效时间步上计算）
                    diff = (out["cont"] - Yc)
                    mse = (diff.pow(2).mean(dim=-1) * M).sum() / (M.sum()+1e-8)
                    loss = loss + args.loss_w_cont * mse
                    cont_mse += float(mse.item()); n_cont += 1
                if train_ds.bin_dim>0:
                    logits = out["bin_logits"]
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits, Yb, reduction="none", pos_weight=pos_weight
                    ).mean(dim=-1)    # B,T
                    bce = (bce * M).sum() / (M.sum()+1e-8)
                    loss = loss + args.loss_w_bin * bce
                    bce_sum += float(bce.item()); n_bin += 1
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    acc = (pred.eq(Yb).float().mean(dim=-1) * M).sum() / (M.sum()+1e-8)
                    acc_sum += float(acc.item())
                # 时序平滑（对连续动作） L_smooth = λ * ||a_t - a_{t-1}||
                if args.smooth_lambda>0.0 and train_ds.cont_dim>0:
                    a = out["cont"]
                    da = (a[:,1:,:] - a[:,:-1,:]).pow(2).mean(dim=-1)  # B,T-1
                    M2 = M[:,1:] * M[:,:-1]
                    smooth = (da * M2).sum() / (M2.sum()+1e-8)
                    loss = loss + args.smooth_lambda * smooth

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()); steps += 1
        metrics = {}
        if n_cont>0: metrics["cont_mse"] = cont_mse/max(n_cont,1)
        if n_bin>0:
            metrics["bin_bce"] = bce_sum/max(n_bin,1)
            metrics["bin_acc"] = acc_sum/max(n_bin,1)
        return loss_sum/max(steps,1), metrics

    def score_fn(met):
        s=0.0
        if train_ds.cont_dim>0: s += met.get("cont_mse",0.0)
        if train_ds.bin_dim>0:  s += (1.0 - met.get("bin_acc",1.0))
        return s

    for ep in range(1, args.epochs+1):
        tr_loss, _    = step_epoch(train_dl, train=True)
        _,   val_met  = step_epoch(val_dl, train=False)
        sc = score_fn(val_met)
        print(f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val={val_met} | score={sc:.6f}")

        if sc < best:
            best = sc; no_imp = 0
            torch.save(model.state_dict(), best_path)
            meta = {
                "obs_cols": train_ds.obs_cols,
                "act_cols": train_ds.act_cols,
                "cont_cols": train_ds.cont_cols,
                "binary_cols": train_ds.bin_cols,
                "obs_mean": {c: float(train_ds.obs_mean[i]) for i,c in enumerate(train_ds.obs_cols)},
                "obs_std":  {c: float(train_ds.obs_std[i])  for i,c in enumerate(train_ds.obs_cols)},
                "cont_mean": train_ds.cont_mean,
                "cont_std":  train_ds.cont_std,
                "time_feat": {"type": "norm+sin2pi+cos2pi", "dim": train_ds.t_feat_dim},
                "model": {
                    "arch": "rnn",
                    "rnn_type": args.rnn_type,
                    "in_dim": train_ds.in_dim,
                    "cont_dim": train_ds.cont_dim,
                    "bin_dim": train_ds.bin_dim,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout
                }
            }
            with open(os.path.join(args.out_dir, "seq_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            no_imp += 1
            if no_imp >= args.early_stop_patience:
                print(f"[EarlyStop] no improvement for {no_imp} epochs.")
                break
    print("Done. Best:", best_path)

if __name__ == "__main__":
    main()
