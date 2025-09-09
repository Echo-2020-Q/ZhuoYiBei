# exp_3_4_ppo_finetune_seq.py (final patched)
# -*- coding: utf-8 -*-
import os, json, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ortho_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class RecurrentActorCritic(nn.Module):
    """
    与 bc_out_seq 的 seq_meta.json 对齐的序列策略：
    输入：标准化后的 obs64 + time_feat(<=3)
    输出：
      - cont: 连续动作（4机*3 = 12) ; 顺序与 meta['cont_cols'] 一致
      - bin_logits: 二值动作（4机*1 = 4）; 顺序与 meta['binary_cols'] 一致
      - V: 状态价值
    """

    def __init__(self, in_dim, cont_dim, bin_dim, rnn_type="gru",
                 hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn_type = rnn_type
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(
            in_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_cont = nn.Linear(hidden_size, cont_dim) if cont_dim > 0 else None
        self.head_bin = nn.Linear(hidden_size, bin_dim) if bin_dim > 0 else None
        self.head_v = nn.Linear(hidden_size, 1)
        self.apply(
            lambda m: _ortho_init(m, gain=math.sqrt(2))
            if isinstance(m, nn.Linear)
            else None
        )
        _ortho_init(self.head_v, gain=1.0)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_state(self, B, device):
        if self.rnn_type == "gru":
            return torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        else:
            h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            return (h0, c0)

    def forward(self, x, h=None):
        # x: [B,T,D]
        z, h1 = self.rnn(x, h)
        v = self.head_v(z)  # [B,T,1]
        out = {}
        if self.head_cont is not None:
            out["cont"] = self.head_cont(z)  # [B,T,Cc]
        if self.head_bin is not None:
            out["bin_logits"] = self.head_bin(z)  # [B,T,Cb]
        return out, v.squeeze(-1), h1


class PPOAgent:
    def __init__(self, bc_dir, device="cpu",
                 gamma=0.995, lam=0.95, clip_ratio=0.2,
                 vf_coef=0.5, ent_coef=0.001, bc_kl_coef=0.05,
                 lr=3e-4, max_grad_norm=1.0, std_init=0.2,
                 v_clip=0.2, target_kl=0.02,
                 init_load_dir=None):
        self.device = torch.device(device)
        self.gamma, self.lam = gamma, lam
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.v_clip = v_clip
        self.target_kl = target_kl

        # === 保存目录 vs 加载目录 ===
        # bc_dir 作为“保存目录”；init_load_dir 作为“加载目录”（若为空则与保存目录一致）
        self.save_dir = bc_dir
        load_dir = init_load_dir or bc_dir

        # === 读取元信息与标准化统计（从 load_dir 读取）===
        meta_path = os.path.join(load_dir, "seq_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"seq_meta.json not found in {load_dir}")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        m = meta["model"]
        self.obs_cols = meta["obs_cols"]
        self.cont_cols = meta["cont_cols"]
        self.bin_cols = meta["binary_cols"]
        self.act_cols = meta["act_cols"]  # 动作列名全集
        self.tdim = int(meta["time_feat"]["dim"])
        self.obs_mean = np.array([meta["obs_mean"][c] for c in self.obs_cols], np.float32)
        self.obs_std = np.array([meta["obs_std"][c] for c in self.obs_cols], np.float32)
        self.cont_mean = (
            np.array([meta["cont_mean"].get(c, 0) for c in self.cont_cols], np.float32)
            if self.cont_cols else None
        )
        self.cont_std = (
            np.array([meta["cont_std"].get(c, 1) for c in self.cont_cols], np.float32)
            if self.cont_cols else None
        )

        # 为了方便断点续训，将 meta 复制到 save_dir（若尚不存在）
        try:
            meta_dst = os.path.join(self.save_dir, "seq_meta.json")
            if not os.path.exists(meta_dst):
                os.makedirs(self.save_dir, exist_ok=True)
                shutil.copyfile(meta_path, meta_dst)
        except Exception:
            pass

        # 依据列名建立索引映射（避免列顺序变动带来的错位）
        self.cont_idx = [self.act_cols.index(c) for c in self.cont_cols]
        self.bin_idx = [self.act_cols.index(c) for c in self.bin_cols]

        self.net = RecurrentActorCritic(
            in_dim=m["in_dim"],
            cont_dim=m["cont_dim"],
            bin_dim=m["bin_dim"],
            rnn_type=m["rnn_type"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
        ).to(self.device)

        # 载入（优先 ONLINE，其次 BC 预训练），均从 load_dir
        ckpt_path = os.path.join(load_dir, "seq_policy.pt.online")
        pt_path = os.path.join(load_dir, "seq_policy.pt")
        ckpt = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.net.load_state_dict(ckpt["net"], strict=False)
            print(f"[PPOAgent] Loaded ONLINE checkpoint from: {ckpt_path}")
        elif os.path.exists(pt_path):
            bc_state = torch.load(pt_path, map_location=self.device)
            self.net.load_state_dict(bc_state, strict=False)
            print(f"[PPOAgent] Loaded BC init from: {pt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found under {load_dir} (tried .online and .pt)")

        # 冻结一个 BC 参考网络用于 KL（初始为当前网络权重）
        self.ref = RecurrentActorCritic(
            in_dim=m["in_dim"],
            cont_dim=m["cont_dim"],
            bin_dim=m["bin_dim"],
            rnn_type=m["rnn_type"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=0.0,
        ).to(self.device)
        self.ref.load_state_dict(self.net.state_dict(), strict=False)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        # 可学习的 log_std（对连续头）
        self.log_std = nn.Parameter(
            torch.full((len(self.cont_cols),), math.log(std_init), device=self.device)
        )

        # 优化器 & 退火相关
        self.base_lr = lr
        self.opt = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=self.base_lr)
        self.ent_coef = ent_coef
        self.ent_coef_min = 0.0001
        self.total_updates_for_anneal = 200
        self.bc_kl_coef = bc_kl_coef
        self._update_steps = 0

        # 若 online ckpt 存在则继续加载优化器等状态
        if ckpt is not None:
            try:
                if "log_std" in ckpt:
                    with torch.no_grad():
                        self.log_std.copy_(ckpt["log_std"].to(self.device))
                if "opt" in ckpt:
                    self.opt.load_state_dict(ckpt["opt"])
                self.bc_kl_coef = ckpt.get("bc_kl_coef", self.bc_kl_coef)
                self._update_steps = ckpt.get("update_steps", self._update_steps)
            except Exception:
                pass

    # === 归一化/反归一化 ===
    def norm_obs(self, o):
        return (o - self.obs_mean) / (self.obs_std + 1e-8)

    def denorm_cont(self, y):
        if self.cont_cols:
            return y * (self.cont_std if self.cont_std is not None else 1.0) + (
                self.cont_mean if self.cont_mean is not None else 0.0
            )
        return y

    # === 行为采样 ===
    @torch.no_grad()
    def act(self, obs64, tfeat, h=None, explore=True):
        """
        obs64: np.float32[64]  (整帧)
        tfeat: np.float32[tdim]
        返回: a4_flat(16), V(float), logp(float), h1
        """
        x = np.concatenate([self.norm_obs(obs64), tfeat], axis=0)[None, None, :]  # [1,1,D]
        xt = torch.from_numpy(x).float().to(self.device)
        out, V, h1 = self.net(xt, h)
        cont_mu = out["cont"][0, 0]  # [Cc]
        bin_logits = out["bin_logits"][0, 0]  # [Cb]
        std = self.log_std.exp()
        if explore:
            cont = cont_mu + std * torch.randn_like(cont_mu)
        else:
            cont = cont_mu
        # fire 二值：伯努利
        bernoulli = torch.distributions.Bernoulli(logits=bin_logits)
        fire = bernoulli.sample()
        logp_fire = bernoulli.log_prob(fire).sum()
        # 连续多元正态
        normal = torch.distributions.Normal(cont_mu, std)
        logp_cont = normal.log_prob(cont).sum()
        logp = (logp_cont + logp_fire).item()

        # 反归一化连续
        cont_denorm = self.denorm_cont(cont.cpu().numpy())
        # 按列映射拼回 act16
        a_full = np.zeros((len(self.act_cols),), np.float32)
        for i, col_idx in enumerate(self.cont_idx):
            a_full[col_idx] = float(cont_denorm[i])
        fire_np = fire.cpu().numpy().astype(np.float32)
        for i, col_idx in enumerate(self.bin_idx):
            a_full[col_idx] = float(fire_np[i])
        return a_full, float(V[0, 0].item()), float(logp), h1

    # === 把 rollout(list) 组装成 batch(张量) + 计算 GAE/回报 ===
    def _build_batch_from_traj(self, traj):
        """
        traj: list of dicts with keys:
          - obs: np.float32[64]
          - tfeat: np.float32[tdim]
          - act: np.float32[16]
          - val: float
          - logp: float
          - rew: float
          - done: float(0/1)
        """
        if len(traj) == 0:
            raise ValueError("Empty trajectory")
        D = len(self.obs_cols) + self.tdim
        Cc = len(self.cont_cols)
        N = len(traj)
        obs = np.zeros((N, 1, D), np.float32)
        act_c = np.zeros((N, 1, Cc), np.float32)
        act_b = np.zeros((N, 1, len(self.bin_cols)), np.float32)
        vals = np.zeros((N, 1), np.float32)
        rews = np.zeros((N, 1), np.float32)
        dones = np.zeros((N, 1), np.float32)
        logp_old = np.zeros((N, 1), np.float32)

        for i, step in enumerate(traj):
            o64 = np.asarray(step["obs"], np.float32)
            tf = np.asarray(step["tfeat"], np.float32)
            x = np.concatenate([self.norm_obs(o64), tf], axis=0)
            obs[i, 0, :] = x
            a16 = np.asarray(step["act"], np.float32)
            # 连续位按 cont_idx 抽取
            act_c[i, 0, :] = np.array([a16[j] for j in self.cont_idx], dtype=np.float32)
            # 二值位按 bin_idx 抽取
            act_b[i, 0, :] = np.array([a16[j] for j in self.bin_idx], dtype=np.float32)
            vals[i, 0] = float(step.get("val", 0.0))
            rews[i, 0] = float(step.get("rew", 0.0))
            dones[i, 0] = float(step.get("done", 0.0))
            logp_old[i, 0] = float(step.get("logp", 0.0))

        # === GAE ===
        adv = np.zeros_like(vals)
        ret = np.zeros_like(vals)
        lastgaelam = 0.0
        for t in reversed(range(N)):
            if t == N - 1:
                nextnonterm = 1.0 - float(dones[t, 0])
                nextv = float(vals[t, 0])
            else:
                nextnonterm = 1.0 - float(dones[t + 1, 0])
                nextv = float(vals[t + 1, 0])
            delta = float(rews[t, 0]) + self.gamma * nextv * nextnonterm - float(vals[t, 0])
            lastgaelam = delta + self.gamma * self.lam * nextnonterm * lastgaelam
            adv[t, 0] = lastgaelam
        ret = adv + vals

        # 归一化优势
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        adv = (adv - adv_mean) / adv_std

        # 张量化 + 旧 V 值（用于 value clip）
        batch = {
            "obs": torch.from_numpy(obs),
            "act_c": torch.from_numpy(act_c),
            "act_b": torch.from_numpy(act_b),
            "adv": torch.from_numpy(adv),
            "ret": torch.from_numpy(ret),
            "logp_old": torch.from_numpy(logp_old),
            "val_old": torch.from_numpy(vals),
        }
        # RNN 初始隐状态（零）
        B = N
        h0 = self.net.init_state(B, self.device)
        batch["h0"] = h0
        return batch

    # === 训练入口：既支持 list(traj) 也支持 现成 batch ===
    def update(self, data, bc_ref=None, epochs=4, minibatch=2):
        if isinstance(data, list):
            batch = self._build_batch_from_traj(data)
        else:
            batch = data
        out = self._update_from_batch(batch, bc_ref=bc_ref, epochs=epochs, minibatch=minibatch)

        # 退火（按更新步推进）
        self._update_steps += 1
        t = min(1.0, self._update_steps / max(1, self.total_updates_for_anneal))
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * (1.0 - 0.8 * t)  # lr: base -> 0.2*base
        # 熵系数从初值退到 ent_coef_min
        self.ent_coef = self.ent_coef * (1.0 - t) + self.ent_coef_min * t
        return out

    def _update_from_batch(self, batch, bc_ref=None, epochs=4, minibatch=2):
        N = batch["obs"].shape[0]
        idx_all = np.arange(N)

        # 统计 KL 以便自适应系数
        kl_vals = []

        for _ in range(epochs):
            np.random.shuffle(idx_all)
            for split in np.array_split(idx_all, minibatch):
                obs = batch["obs"][split].to(self.device)  # [B,1,D]
                act_c = batch["act_c"][split].to(self.device)  # [B,1,Cc]
                act_b = batch["act_b"][split].to(self.device)  # [B,1,Cb]
                adv = batch["adv"][split].to(self.device)  # [B,1]
                ret = batch["ret"][split].to(self.device)  # [B,1]
                logp_old = batch["logp_old"][split].to(self.device)  # [B,1]
                v_old = batch["val_old"][split].to(self.device)[:, 0]  # [B]

                h0 = batch["h0"]
                if isinstance(h0, tuple):  # LSTM
                    h0 = (h0[0][:, split].to(self.device), h0[1][:, split].to(self.device))
                else:  # GRU
                    h0 = h0[:, split].to(self.device)

                out, V, _ = self.net(obs, h0)
                mu = out["cont"][:, 0, :]  # [B,Cc]
                logits = out["bin_logits"][:, 0, :]  # [B,Cb]
                std = self.log_std.exp().unsqueeze(0)  # [1,Cc]

                # 连续对数似然
                normal = torch.distributions.Normal(mu, std)
                logp_cont = normal.log_prob(act_c[:, 0, :]).sum(dim=-1, keepdim=True)  # [B,1]

                # 二值对数似然
                bern = torch.distributions.Bernoulli(logits=logits)
                logp_fire = bern.log_prob(act_b[:, 0, :]).sum(dim=-1, keepdim=True)  # [B,1]

                logp = logp_cont + logp_fire
                ratio = (logp - logp_old).exp()

                # PPO-clip
                obj1 = ratio * adv
                obj2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                pg_loss = -torch.min(obj1, obj2).mean()

                # Value loss with clipping
                v_pred = V[:, 0]
                v_clipped = v_old + (v_pred - v_old).clamp(-self.v_clip, self.v_clip)
                v_loss1 = F.mse_loss(v_pred, ret[:, 0], reduction="none")
                v_loss2 = F.mse_loss(v_clipped, ret[:, 0], reduction="none")
                v_loss = torch.max(v_loss1, v_loss2).mean()

                # 熵（鼓励探索）：连续近似用 log_std，总熵加上 Bernoulli 熵
                ent = (torch.log(std).sum() + bern.entropy().mean())

                # 额外 KL 到 BC 参考（单步近似）
                with torch.no_grad():
                    ref_out, _, _ = (bc_ref or self.ref)(obs, h0)
                mu_ref = ref_out["cont"][:, 0, :]
                logits_ref = ref_out["bin_logits"][:, 0, :]
                std_ref = std.detach()
                # KL(N0||N1) = sum(log(s1/s0) + (s0^2+(m0-m1)^2)/(2*s1^2) - 1/2)
                kl_cont = (
                    torch.log(std / std_ref)
                    + (std_ref.pow(2) + (mu - mu_ref).pow(2)) / (2 * std.pow(2))
                    - 0.5
                ).sum(dim=-1).mean()
                # Bernoulli KL
                p = torch.sigmoid(logits)
                q = torch.sigmoid(logits_ref)
                kl_bin = (
                    p * torch.log((p + 1e-8) / (q + 1e-8))
                    + (1 - p) * torch.log(((1 - p) + 1e-8) / ((1 - q) + 1e-8))
                ).sum(dim=-1).mean()
                kl_total = kl_cont + kl_bin
                kl_vals.append(kl_total.detach().cpu())

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent + self.bc_kl_coef * kl_total

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.net.parameters()) + [self.log_std], self.max_grad_norm)
                self.opt.step()

                # 限制 std 范围
                with torch.no_grad():
                    self.log_std.clamp_(math.log(1e-3), math.log(1.0))

        # epoch 结束后根据 KL 自适应调节 BC KL 系数
        if len(kl_vals) > 0:
            kl_mean = torch.stack(list(kl_vals)).mean().item()
            if kl_mean > 1.5 * self.target_kl:
                self.bc_kl_coef *= 1.5
            elif kl_mean < 0.5 * self.target_kl:
                self.bc_kl_coef *= 0.7
            self.bc_kl_coef = float(np.clip(self.bc_kl_coef, 1e-4, 1.0))

        # 返回一些监控指标（可选）
        return {
            "kl_mean": float(kl_mean) if len(kl_vals) > 0 else 0.0,
            "bc_kl_coef": float(self.bc_kl_coef),
            "updates": int(self._update_steps),
        }

    def save(self, out_dir=None):
        out_dir = out_dir or self.save_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "log_std": self.log_std.data.detach().cpu(),
            "opt": self.opt.state_dict(),
            "bc_kl_coef": self.bc_kl_coef,
            "update_steps": self._update_steps,
        }, os.path.join(out_dir, "seq_policy.pt.online"))
