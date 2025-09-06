# -*- coding: utf-8 -*-
"""
Behavior Cloning (Imitation Learning) from expert logs (obs_* -> act_*).

增强点：
- --hidden "256,256"     自定义隐藏层
- --dropout 0.1          Dropout
- --val_ratio 0.1        验证集占比
- --seed 42              随机种子
- --epochs 50            训练轮数
- --batch_size 256       批大小
- --lr 1e-3              学习率
- --weight_decay 1e-4    权重衰减
- --loss_w_cont 1.0      连续动作损失权重
- --loss_w_bin 1.0       二值动作损失权重
- --grad_clip 1.0        梯度裁剪
- --scheduler plateau    学习率调度器：none/plateau/cosine
- --early_stop_patience 10 早停耐心
- --num_workers 0        DataLoader 线程数
- --pin_memory           DataLoader pin_memory
- --amp                  混合精度
- --save_every 0         每 N 轮保存一次 checkpoint（0=只保存最佳）
- 自动识别 obs_* / act_*，区分连续/二值动作；保存 policy.pt + meta.json
"""

import os, json, glob, random, argparse, math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_hidden(s: str) -> List[int]:
    if not s:
        return [256, 256]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]

# ---------------- Dataset ----------------
class ExpertDataset(Dataset):
    def __init__(self, files: List[str]):
        assert len(files) > 0, "No data files found."
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if len(df) == 0:
                    continue
                df["__source__"] = f
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read {f}: {e}")
        assert len(dfs) > 0, "All files failed to load or were empty."
        self.df = pd.concat(dfs, ignore_index=True)

        self.obs_cols = [c for c in self.df.columns if c.startswith("obs_")]
        self.act_cols = [c for c in self.df.columns if c.startswith("act_")]
        assert len(self.obs_cols) > 0, "No obs_* columns found."
        assert len(self.act_cols) > 0, "No act_* columns found."

        # split actions into continuous vs binary
        self.binary_cols, self.cont_cols = [], []
        for c in self.act_cols:
            vals = self.df[c].dropna().values
            uniq = np.unique(vals)
            if np.all(np.isin(uniq, [0, 1])):
                self.binary_cols.append(c)
            else:
                self.cont_cols.append(c)

        # normalization stats on obs + continuous actions
        self.obs_mean = self.df[self.obs_cols].mean().to_dict()
        self.obs_std = self.df[self.obs_cols].std().replace(0, 1).to_dict()

        self.cont_mean = {c: float(self.df[c].mean()) for c in self.cont_cols} if self.cont_cols else {}
        self.cont_std = {
            c: float(self.df[c].std() if self.df[c].std() != 0 else 1.0)
            for c in self.cont_cols
        } if self.cont_cols else {}

        # cache tensors
        X = self.df[self.obs_cols].astype(np.float32).values
        X = (X - np.array([self.obs_mean[c] for c in self.obs_cols], dtype=np.float32)) / \
            np.array([self.obs_std[c] for c in self.obs_cols], dtype=np.float32)
        self.X = torch.from_numpy(X)

        self.y_cont = None
        self.y_bin = None
        if self.cont_cols:
            Yc = self.df[self.cont_cols].astype(np.float32).values
            if len(self.cont_cols) > 0:
                Yc = (Yc - np.array([self.cont_mean[c] for c in self.cont_cols], dtype=np.float32)) / \
                     np.array([self.cont_std[c] for c in self.cont_cols], dtype=np.float32)
            self.y_cont = torch.from_numpy(Yc)
        if self.binary_cols:
            Yb = self.df[self.binary_cols].astype(np.float32).values
            self.y_bin = torch.from_numpy(Yb)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        x = self.X[idx]
        out = {"obs": x}
        if self.y_cont is not None: out["y_cont"] = self.y_cont[idx]
        if self.y_bin is not None: out["y_bin"] = self.y_bin[idx]
        return out

# ---------------- Model ----------------
class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, cont_dim: int, bin_dim: int,
                 hidden: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        self.cont_head = nn.Linear(last, cont_dim) if cont_dim > 0 else None
        self.bin_head = nn.Linear(last, bin_dim) if bin_dim > 0 else None

    def forward(self, x):
        z = self.shared(x)
        out = {}
        if self.cont_head is not None:
            out["cont"] = self.cont_head(z)  # normalized space
        if self.bin_head is not None:
            out["bin_logits"] = self.bin_head(z)
        return out

# ---------------- Train/Eval ----------------
def evaluate(model, dl, device, has_cont, has_bin, pos_weight=None):
    model.eval()
    mse_sum = 0.0; n_c = 0
    bce_sum = 0.0; n_b = 0
    acc_sum = 0.0; acc_n = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["obs"].to(device)
            out = model(x)
            if has_cont:
                yc = batch["y_cont"].to(device)
                predc = out["cont"]
                mse = nn.functional.mse_loss(predc, yc, reduction="mean")
                mse_sum += float(mse.item()); n_c += 1
            if has_bin:
                yb = batch["y_bin"].to(device)
                logits = out["bin_logits"]
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="mean",
                    pos_weight=pos_weight.to(device) if pos_weight is not None else None
                )
                bce_sum += float(bce.item()); n_b += 1
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred.eq(yb).float().mean())
                acc_sum += float(acc.item()); acc_n += 1
    res = {}
    if n_c > 0: res["cont_mse"] = mse_sum / max(n_c,1)
    if n_b > 0:
        res["bin_bce"] = bce_sum / max(n_b,1)
        res["bin_acc"] = acc_sum / max(acc_n,1)
    return res

def build_scheduler(optimizer, name: str, max_epochs: int):
    name = (name or "none").lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return None

def main():
    p = argparse.ArgumentParser()
    # IO
    p.add_argument("--data_glob", type=str, required=True, help='例如 "runs/20250829_230010/ep_*.csv"')
    p.add_argument("--out_dir", type=str, default="./bc_out")
    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--hidden", type=str, default="256,256")
    p.add_argument("--dropout", type=float, default=0.0)
    # Loss/opt tricks
    p.add_argument("--loss_w_cont", type=float, default=1.0)
    p.add_argument("--loss_w_bin", type=float, default=1.0)
    p.add_argument("--pos_weight", type=float, default=0.0, help=">0 则对正样本加权（处理正负极不均衡）")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "plateau", "cosine"])
    p.add_argument("--early_stop_patience", type=int, default=10)
    # DataLoader
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    # Misc
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_every", type=int, default=0, help="每 N 轮保存一次 checkpoint；0 表示只保存最佳模型")
    args = p.parse_args()

    set_seed(args.seed)
    files = sorted(glob.glob(args.data_glob))
    if len(files) == 0 and os.path.isfile(args.data_glob):
        files = [args.data_glob]
    assert len(files) > 0, f"No files matched: {args.data_glob}"

    os.makedirs(args.out_dir, exist_ok=True)

    ds = ExpertDataset(files)
    n = len(ds)
    val_len = max(1, int(n * args.val_ratio))
    train_len = n - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    in_dim = len(ds.obs_cols)
    cont_dim = len(ds.cont_cols)
    bin_dim  = len(ds.binary_cols)
    hidden = parse_hidden(args.hidden)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(in_dim, cont_dim, bin_dim, hidden=hidden, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # pos_weight for BCE (imbalance handling)
    pos_weight = None
    if bin_dim > 0 and args.pos_weight > 0:
        pos_weight = torch.full((bin_dim,), float(args.pos_weight), dtype=torch.float32, device=device)

    best_metric = float("inf")
    best_path = os.path.join(args.out_dir, "policy.pt")
    no_improve = 0

    def composite_val_score(metrics):
        # 越小越好：连续用 MSE，二值用 (1-acc)
        score = 0.0
        if cont_dim > 0:
            score += metrics.get("cont_mse", 0.0)
        if bin_dim > 0:
            score += (1.0 - metrics.get("bin_acc", 1.0))
        return score

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; steps = 0
        for batch in train_dl:
            x = batch["obs"].to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x)
                loss = 0.0
                if cont_dim > 0:
                    yc = batch["y_cont"].to(device)
                    loss_c = nn.functional.mse_loss(out["cont"], yc, reduction="mean") * args.loss_w_cont
                    loss = loss + loss_c
                if bin_dim > 0:
                    yb = batch["y_bin"].to(device)
                    loss_b = nn.functional.binary_cross_entropy_with_logits(
                        out["bin_logits"], yb, reduction="mean",
                        pos_weight=pos_weight
                    ) * args.loss_w_bin
                    loss = loss + loss_b

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()); steps += 1

        # eval
        metrics = evaluate(model, val_dl, device, cont_dim > 0, bin_dim > 0, pos_weight=pos_weight)
        val_score = composite_val_score(metrics)

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={running/max(steps,1):.6f} | val={metrics} | score={val_score:.6f}")

        # save best
        if val_score < best_metric:
            best_metric = val_score
            torch.save(model.state_dict(), best_path)
            # also save meta whenever best
            meta = {
                "obs_cols": ds.obs_cols,
                "act_cols": ds.act_cols,
                "cont_cols": ds.cont_cols,
                "binary_cols": ds.binary_cols,
                "obs_mean": ds.obs_mean,
                "obs_std": ds.obs_std,
                "cont_mean": ds.cont_mean,
                "cont_std": ds.cont_std,
                "model": {
                    "in_dim": in_dim, "cont_dim": cont_dim, "bin_dim": bin_dim,
                    "hidden": hidden, "dropout": args.dropout
                }
            }
            with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            no_improve = 0
        else:
            no_improve += 1

        # periodic checkpoint
        if args.save_every and args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt = os.path.join(args.out_dir, f"policy_ep{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt)

        # early stopping
        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] no improvement for {no_improve} epochs. Stop.")
            break

    print("Training complete. Best model:", best_path)

if __name__ == "__main__":
    main()
