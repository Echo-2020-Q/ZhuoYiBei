# exp_3_4_ppo_finetune_seq_19obs.py (seq-enabled)
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
                 hidden_size=256, num_layers=2, dropout=0.0):
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

        # 先设置这些默认值，后面若 ckpt 有会覆盖
        self.bc_kl_coef = bc_kl_coef
        self._update_steps = 0
        self.ent_coef = ent_coef
        self.ent_coef_min = 0.0001
        self.total_updates_for_anneal = 200
        self.base_lr = lr

        # === 保存目录 vs 加载目录 ===
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
        self.act_cols = meta["act_cols"]
        self.tdim = int(meta["time_feat"]["dim"])

        # 读取 BC 里 64 维的统计
        self.obs_mean = np.array([meta["obs_mean"][c] for c in self.obs_cols], np.float32)
        self.obs_std = np.array([meta["obs_std"][c] for c in self.obs_cols], np.float32)

        # === 运行时观测维度（19*4=76）===
        self.runtime_obs_dim = 19 * 4
        if self.obs_mean.shape[0] != self.runtime_obs_dim:
            pad = self.runtime_obs_dim - self.obs_mean.shape[0]
            assert pad > 0, "runtime_obs_dim 小于 BC 统计，不应发生"
            self.obs_mean = np.concatenate([self.obs_mean, np.zeros(pad, np.float32)], axis=0)
            self.obs_std = np.concatenate([self.obs_std, np.ones(pad, np.float32)], axis=0)

        # 连续动作统计照常
        self.cont_mean = (np.array([meta["cont_mean"].get(c, 0) for c in self.cont_cols], np.float32)
                          if self.cont_cols else None)
        self.cont_std = (np.array([meta["cont_std"].get(c, 1) for c in self.cont_cols], np.float32)
                         if self.cont_cols else None)

        # 列索引映射
        self.cont_idx = [self.act_cols.index(c) for c in self.cont_cols]
        self.bin_idx = [self.act_cols.index(c) for c in self.bin_cols]

        # === 用新的输入维度建网 ===
        new_in_dim = self.runtime_obs_dim + self.tdim
        self.net = RecurrentActorCritic(
            in_dim=new_in_dim,
            cont_dim=m["cont_dim"],
            bin_dim=m["bin_dim"],
            rnn_type=m["rnn_type"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
        ).to(self.device)

        # 载入权重（自动识别 GRU/LSTM & 兼容加载）
        ckpt_path = os.path.join(load_dir, "seq_policy.pt.online")
        pt_path = os.path.join(load_dir, "seq_policy.pt")
        ckpt = None
        state = None
        src_path = None

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            src_path = ckpt_path
            state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
            print(f"[PPOAgent] Found ONLINE checkpoint: {ckpt_path}")
        elif os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location=self.device)
            src_path = pt_path
            state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
            print(f"[PPOAgent] Found BC checkpoint: {pt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found under {load_dir} (tried .online and .pt)")

        def _infer_rnn_type_from_state(state_dict, hidden_size):
            w = state_dict.get("rnn.weight_ih_l0", None)
            if w is None:
                return None
            gates = w.shape[0] // int(hidden_size)
            if gates == 3:
                return "gru"
            elif gates == 4:
                return "lstm"
            return None

        inferred_type = _infer_rnn_type_from_state(state, hidden_size=m["hidden_size"])
        if inferred_type is not None and inferred_type != getattr(self.net, "rnn_type", None):
            print(f"[PPOAgent] Detected ckpt RNN = {inferred_type} by shape; "
                  f"current model = {self.net.rnn_type}. Rebuilding to match ckpt...")
            self.net = RecurrentActorCritic(
                in_dim=new_in_dim,
                cont_dim=m["cont_dim"],
                bin_dim=m["bin_dim"],
                rnn_type=inferred_type,
                hidden_size=m["hidden_size"],
                num_layers=m["num_layers"],
                dropout=m["dropout"],
            ).to(self.device)

        curr = self.net.state_dict()
        compatible = {k: v for k, v in state.items() if (k in curr and v.shape == curr[k].shape)}
        skipped = [k for k in state.keys() if k not in compatible]
        curr.update(compatible)
        self.net.load_state_dict(curr, strict=True)
        print(f"[PPOAgent] Loaded {len(compatible)} tensors from {src_path}; "
              f"skipped {len(skipped)} due to shape mismatch.")

        # 参考网络（KL 用），结构与 self.net 一致
        self.ref = RecurrentActorCritic(
            in_dim=new_in_dim,
            cont_dim=m["cont_dim"],
            bin_dim=m["bin_dim"],
            rnn_type=self.net.rnn_type,
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=0.0,
        ).to(self.device)
        self.ref.load_state_dict(self.net.state_dict(), strict=True)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        # 如果 ckpt 里带有 log_std/opt/bc_kl_coef/update_steps，记录下来稍后恢复
        if isinstance(ckpt, dict):
            try:
                if "log_std" in ckpt:
                    self._restore_log_std_from_ckpt = ckpt["log_std"].to(self.device)
                if "opt" in ckpt:
                    self._restore_opt_state_dict = ckpt["opt"]
                # 这里会安全覆盖，因为我们前面已给了默认值
                self.bc_kl_coef = ckpt.get("bc_kl_coef", self.bc_kl_coef)
                self._update_steps = ckpt.get("update_steps", self._update_steps)
            except Exception as _e:
                print(f"[PPOAgent] Optional optimizer/log_std restore note: {_e}")

        # 可学习的 log_std（对连续头）
        self.log_std = nn.Parameter(
            torch.full((len(self.cont_cols),), math.log(std_init), device=self.device)
        )

        # 优化器
        self.opt = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=self.base_lr)

        # === 日志与曲线输出目录 ===
        self.plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.train_log_csv = os.path.join(self.save_dir, "train_log.csv")

        # 恢复 log_std / 优化器（若有）
        try:
            if hasattr(self, "_restore_log_std_from_ckpt"):
                with torch.no_grad():
                    if self._restore_log_std_from_ckpt.shape == self.log_std.shape:
                        self.log_std.copy_(self._restore_log_std_from_ckpt)
                del self._restore_log_std_from_ckpt
        except Exception:
            pass
        try:
            if hasattr(self, "_restore_opt_state_dict"):
                self.opt.load_state_dict(self._restore_opt_state_dict)
                del self._restore_opt_state_dict
        except Exception:
            pass



    def _save_epoch_plot(self, epoch_stats, step_idx: int):
        """
        把每个 epoch 的损失曲线画到一张图里：total / policy / value / entropy。
        返回保存的路径；若 matplotlib 不可用则返回 None。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np
        except Exception:
            return None

        if not epoch_stats:
            return None

        x = _np.arange(1, len(epoch_stats) + 1, dtype=_np.int32)
        tot = _np.array([s["loss_total"] for s in epoch_stats], dtype=float)
        pg  = _np.array([s["pg_loss"]    for s in epoch_stats], dtype=float)
        vl  = _np.array([s["v_loss"]     for s in epoch_stats], dtype=float)
        ent = _np.array([s["entropy"]    for s in epoch_stats], dtype=float)

        plt.figure(figsize=(8, 5), dpi=120)
        plt.plot(x, tot, label="total_loss")
        plt.plot(x, pg,  label="policy_loss")
        plt.plot(x, vl,  label="value_loss")
        plt.plot(x, ent, label="entropy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Entropy")
        plt.title(f"PPO Update #{step_idx} Loss Curves")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        save_path = os.path.join(self.plots_dir, f"update_{int(step_idx):05d}.png")
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(save_path)
        plt.close()
        return save_path

    # === 归一化/反归一化 ===
    def norm_obs(self, o):
        return (o - self.obs_mean) / (self.obs_std + 1e-8)

    def denorm_cont(self, y):
        if self.cont_cols:
            return y * (self.cont_std if self.cont_std is not None else 1.0) + (
                self.cont_mean if self.cont_mean is not None else 0.0
            )
        return y

    # === 便捷接口：初始化 RNN 隐状态 ===
    def init_hidden(self, B=1):
        return self.net.init_state(B, self.device)

    # === 行为采样（支持传入/返回隐状态） ===
    @torch.no_grad()
    def act(self, obs_vec, tfeat, h=None, explore=True):
        """
        obs76: np.float32[76]
        tfeat: np.float32[tdim]
        h:     RNN 隐状态（来自上一次 act 的返回），或 None
        返回: a4_flat(16), V(float), logp(float), h1
        """
        x = np.concatenate([self.norm_obs(obs_vec), tfeat], axis=0)[None, None, :]  # [1,1,D]
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

    # === 把整段 traj 组装成 [B=1, T, D] 的批次 + 计算 GAE ===
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
        返回：适配 RNN 的 batch（B=1, T=len(traj)）
        """
        if len(traj) == 0:
            raise ValueError("Empty trajectory")

        D = self.obs_mean.shape[0] + self.tdim  # 运行时 76 + tdim
        Cc = len(self.cont_cols)
        Cb = len(self.bin_cols)
        T = len(traj)
        B = 1

        obs = np.zeros((B, T, D), np.float32)
        act_c = np.zeros((B, T, Cc), np.float32)
        act_b = np.zeros((B, T, Cb), np.float32)
        vals = np.zeros((B, T), np.float32)
        rews = np.zeros((B, T), np.float32)
        dones = np.zeros((B, T), np.float32)
        logp_old = np.zeros((B, T), np.float32)

        for t in range(T):
            step = traj[t]
            o76 = np.asarray(step["obs"], np.float32)
            tf = np.asarray(step["tfeat"], np.float32)
            x = np.concatenate([self.norm_obs(o76), tf], axis=0)
            obs[0, t, :] = x
            a16 = np.asarray(step["act"], np.float32)
            act_c[0, t, :] = np.array([a16[j] for j in self.cont_idx], dtype=np.float32)
            act_b[0, t, :] = np.array([a16[j] for j in self.bin_idx], dtype=np.float32)
            vals[0, t] = float(step.get("val", 0.0))
            rews[0, t] = float(step.get("rew", 0.0))
            dones[0, t] = float(step.get("done", 0.0))
            logp_old[0, t] = float(step.get("logp", 0.0))

        # === GAE over time axis T ===
        adv = np.zeros_like(vals)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterm = 1.0 - float(dones[0, t])
                nextv = float(vals[0, t])
            else:
                nextnonterm = 1.0 - float(dones[0, t + 1])
                nextv = float(vals[0, t + 1])
            delta = float(rews[0, t]) + self.gamma * nextv * nextnonterm - float(vals[0, t])
            lastgaelam = delta + self.gamma * self.lam * nextnonterm * lastgaelam
            adv[0, t] = lastgaelam
            ret = adv + vals

        # 归一化优势
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        adv = (adv - adv_mean) / adv_std

        batch = {
            "obs": torch.from_numpy(obs),          # [1,T,D]
            "act_c": torch.from_numpy(act_c),      # [1,T,Cc]
            "act_b": torch.from_numpy(act_b),      # [1,T,Cb]
            "adv": torch.from_numpy(adv),          # [1,T]
            "ret": torch.from_numpy(ret),          # [1,T]
            "logp_old": torch.from_numpy(logp_old),# [1,T]
            "val_old": torch.from_numpy(vals),     # [1,T]
        }
        # 初始隐状态（零）
        batch["h0"] = self.net.init_state(B=1, device=self.device)
        return batch

    # === 训练入口 ===
    def update(self, data, bc_ref=None, epochs=4, minibatch=2):
        if isinstance(data, list):
            batch = self._build_batch_from_traj(data)
        else:
            batch = data
        out = self._update_from_batch(batch, bc_ref=bc_ref, epochs=epochs, minibatch=minibatch)

        # 退火
        self._update_steps += 1
        t = min(1.0, self._update_steps / max(1, self.total_updates_for_anneal))
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * (1.0 - 0.8 * t)
        self.ent_coef = self.ent_coef * (1.0 - t) + self.ent_coef_min * t
        return out

    def _update_from_batch(self, batch, bc_ref=None, epochs=4, minibatch=2):
        """
        支持 [B=1, T, D] 的整段序列训练；按时间轴把 T 均分为 `minibatch` 份做截断 BPTT。
        训练时：
          - 每个 epoch 统计/打印：pg_loss, v_loss, entropy, total_loss, approx_kl(old->new), clipfrac, ev, kl_ref
          - 保存一张损失曲线图到 ./bc_out_seq/plots/update_XXXXX.png
          - 追加数值到 ./bc_out_seq/train_log.csv
        """
        obs_all = batch["obs"].to(self.device)  # [1,T,D]
        act_c_all = batch["act_c"].to(self.device)  # [1,T,Cc]
        act_b_all = batch["act_b"].to(self.device)  # [1,T,Cb]
        adv_all = batch["adv"].to(self.device)  # [1,T]
        ret_all = batch["ret"].to(self.device)  # [1,T]
        logp_old_all = batch["logp_old"].to(self.device)  # [1,T]
        v_old_all = batch["val_old"].to(self.device)  # [1,T]

        B, T, D = obs_all.shape
        assert B == 1, "当前实现默认 B=1（整段 episode 作为一条序列）"

        # 将时间轴分成 minibatch 份
        splits = np.array_split(np.arange(T), max(1, minibatch))
        kl_vals = []
        epoch_stats = []

        def _explained_variance(y_pred, y_true):
            # y_*: [1,T]
            y_pred = y_pred.detach()
            y_true = y_true.detach()
            var_y = torch.var(y_true)
            if float(var_y) < 1e-12:
                return 0.0
            return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-12))

        for e in range(epochs):
            # 每个 epoch 打乱时间切块顺序
            np.random.shuffle(splits)

            agg = dict(pg=[], v=[], ent=[], tot=[], kl_ref=[], kl_approx=[], clipfrac=[])

            for idxs in splits:
                t0 = int(idxs[0]); t1 = int(idxs[-1]) + 1
                obs = obs_all[:, t0:t1, :]  # [1,τ,D]
                act_c = act_c_all[:, t0:t1, :]  # [1,τ,Cc]
                act_b = act_b_all[:, t0:t1, :]  # [1,τ,Cb]
                adv = adv_all[:, t0:t1]  # [1,τ]
                ret = ret_all[:, t0:t1]  # [1,τ]
                logp_old = logp_old_all[:, t0:t1]  # [1,τ]
                v_old = v_old_all[:, t0:t1]  # [1,τ]

                # 截断 BPTT：每个切块从零隐状态开始（简单稳妥）
                h0 = self.net.init_state(B=1, device=self.device)

                out, V, _ = self.net(obs, h0)  # out: [1,τ,*] ; V: [1,τ]
                mu = out["cont"][:, :, :]  # [1,τ,Cc]
                logits = out["bin_logits"][:, :, :]  # [1,τ,Cb]
                std = self.log_std.exp().unsqueeze(0).unsqueeze(0)  # [1,1,Cc]

                # 连续对数似然（逐时刻求和掉动作维）
                normal = torch.distributions.Normal(mu, std)
                logp_cont = normal.log_prob(act_c).sum(dim=-1)  # [1,τ]

                # 二值对数似然
                bern = torch.distributions.Bernoulli(logits=logits)
                logp_fire = bern.log_prob(act_b).sum(dim=-1)  # [1,τ]

                logp = logp_cont + logp_fire  # [1,τ]
                ratio = (logp - logp_old).exp()  # [1,τ]

                # PPO-clip（逐时刻）
                obj1 = ratio * adv
                obj2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                pg_loss = -torch.min(obj1, obj2).mean()

                # Value loss with clipping（逐时刻）
                v_pred = V
                v_clipped = v_old + (v_pred - v_old).clamp(-self.v_clip, self.v_clip)
                v_loss1 = F.mse_loss(v_pred, ret, reduction="none")
                v_loss2 = F.mse_loss(v_clipped, ret, reduction="none")
                v_loss = torch.max(v_loss1, v_loss2).mean()

                # 熵（连续 + Bernoulli）
                ent_cont = torch.log(std).sum()  # 常数项（按动作维），对每个切片仅计一次
                ent_bin = bern.entropy().mean()
                ent = ent_cont + ent_bin

                # KL 到参考策略（逐时刻平均）
                with torch.no_grad():
                    ref_out, _, _ = (bc_ref or self.ref)(obs, h0)
                mu_ref = ref_out["cont"][:, :, :]
                logits_ref = ref_out["bin_logits"][:, :, :]
                std_ref = std.detach()

                kl_cont = (
                        torch.log(std / std_ref)
                        + (std_ref.pow(2) + (mu - mu_ref).pow(2)) / (2 * std.pow(2))
                        - 0.5
                ).sum(dim=-1).mean()

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

                with torch.no_grad():
                    self.log_std.clamp_(math.log(1e-3), math.log(1.0))

                # ==== 统计信息（本 minibatch）====
                with torch.no_grad():
                    approx_kl = torch.mean(torch.abs(logp_old - logp)).item()
                    clipfrac = torch.mean(((ratio - 1.0).abs() > self.clip_ratio).float()).item()
                    agg["pg"].append(float(pg_loss.item()))
                    agg["v"].append(float(v_loss.item()))
                    agg["ent"].append(float(ent.item()))
                    agg["tot"].append(float(loss.item()))
                    agg["kl_ref"].append(float(kl_total.item()))
                    agg["kl_approx"].append(float(approx_kl))
                    agg["clipfrac"].append(float(clipfrac))

            # ==== epoch 级聚合 ====
            with torch.no_grad():
                # 用全序列估计 EV
                h0_all = self.net.init_state(B=1, device=self.device)
                _, V_all, _ = self.net(obs_all, h0_all)  # [1,T]
                ev = _explained_variance(V_all, ret_all)

            ep_stat = {
                "epoch": int(e + 1),
                "pg_loss": float(np.mean(agg["pg"])) if agg["pg"] else 0.0,
                "v_loss": float(np.mean(agg["v"])) if agg["v"] else 0.0,
                "entropy": float(np.mean(agg["ent"])) if agg["ent"] else 0.0,
                "loss_total": float(np.mean(agg["tot"])) if agg["tot"] else 0.0,
                "kl_ref": float(np.mean(agg["kl_ref"])) if agg["kl_ref"] else 0.0,
                "approx_kl": float(np.mean(agg["kl_approx"])) if agg["kl_approx"] else 0.0,
                "clipfrac": float(np.mean(agg["clipfrac"])) if agg["clipfrac"] else 0.0,
                "ev": float(ev),
            }
            epoch_stats.append(ep_stat)

            # 控制台打印每个 epoch 的指标
            print(
                "[PPO][epoch {}/{}] total={:.4f} | pg={:.4f} v={:.4f} ent={:.4f} | "
                "kl_ref={:.4f} kl≈={:.4f} clipfrac={:.2%} ev={:.3f}".format(
                    ep_stat["epoch"], epochs,
                    ep_stat["loss_total"], ep_stat["pg_loss"], ep_stat["v_loss"], ep_stat["entropy"],
                    ep_stat["kl_ref"], ep_stat["approx_kl"], ep_stat["clipfrac"], ep_stat["ev"]
                ),
                flush=True
            )

        # === 动态调 KL 系数 ===
        if len(kl_vals) > 0:
            kl_mean = torch.stack(list(kl_vals)).mean().item()
            if kl_mean > 1.5 * self.target_kl:
                self.bc_kl_coef *= 1.5
            elif kl_mean < 0.5 * self.target_kl:
                self.bc_kl_coef *= 0.7
            self.bc_kl_coef = float(np.clip(self.bc_kl_coef, 1e-4, 1.0))
        else:
            kl_mean = 0.0

        # === 画图 & 记录 CSV ===
        plot_path = self._save_epoch_plot(epoch_stats, self._update_steps + 1)

        try:
            need_head = not os.path.exists(self.train_log_csv)
            with open(self.train_log_csv, "a", newline="", encoding="utf-8") as f:
                import csv as _csv, time as _time
                w = _csv.writer(f)
                if need_head:
                    w.writerow([
                        "time", "update_idx", "epoch", "total_loss", "pg_loss", "v_loss", "entropy",
                        "kl_ref", "approx_kl", "clipfrac", "ev", "lr", "ent_coef", "bc_kl_coef"
                    ])
                lr_now = self.opt.param_groups[0]["lr"]
                for s in epoch_stats:
                    w.writerow([
                        int(_time.time()), int(self._update_steps + 1), int(s["epoch"]),
                        float(s["loss_total"]), float(s["pg_loss"]), float(s["v_loss"]), float(s["entropy"]),
                        float(s["kl_ref"]), float(s["approx_kl"]), float(s["clipfrac"]), float(s["ev"]),
                        float(lr_now), float(self.ent_coef), float(self.bc_kl_coef)
                    ])
        except Exception:
            pass

        # 返回更多信息，便于外部打印
        return {
            "kl_mean": float(kl_mean),
            "bc_kl_coef": float(self.bc_kl_coef),
            "updates": int(self._update_steps),
            "epoch_stats": epoch_stats,
            "plot_path": plot_path,
            "lr": float(self.opt.param_groups[0]["lr"]),
            "ent_coef": float(self.ent_coef),
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
