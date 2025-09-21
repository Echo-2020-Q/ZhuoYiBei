# -*- coding: utf-8 -*-
"""exp3.7_PPO19_reward.py
批量采集多轮（例如 1000 轮）对抗的 obs64+act16 数据，并在线更新 PPO。
- 每轮写 runs/<timestamp>/ep_XXXX.csv 与 manifest.json
- 攻击逻辑：PPO 只产出 (rate, direct, vz, fire intent)，fire 作为“意图位”
  后续仍用你两阶段打击流程（策略优先 -> 基线分配），必须确认打击成功才记一次
- 奖励：不用 get_score()；采用
    reward = Δ击毁蓝(+1) + Δ被毁红(-1) + 禁飞区内每秒(-0.05/架)
- PPO 在线更新：每轮结束用整轮 traj 调用 ppo.update()，并保存到 ./bc_out_seq/seq_policy.pt.online
"""
import random
import os
import csv
import json
import time
import re
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional
import math

import rflysim
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

# === PPO（使用你提供的实现） ===
from exp_3_4_ppo_finetune_seq_19obs import PPOAgent

# ======= BC 策略加载器（保留原实现以备参考/对齐列名等；动作由 PPO 决定）=======
import json as _json
import numpy as _np
import torch as _torch
from torch import nn as _nn

# ==================== 你的原始常量（可按需调整） ====================
DESTROYED_FLAG = "DAMAGE_STATE_DESTROYED"
RED_IDS = [10091, 10084, 10085, 10086]

AMMO_MAX = 8
ATTACK_COOLDOWN_SEC = 1.5
SCAN_INTERVAL_SEC = 0.1
LOCK_SEC = 55.0
BLUE_SIDE_CODE = 2
DESTROY_CODES = {1}

vel_0 = rflysim.Vel()
vel_0.vz = 150
vel_0.rate = 200
vel_0.direct = 90

BOUNDARY_RECT = {
    "min_x": 101.07326442811694,
    "max_x": 103.08242360888715,
    "min_y": 39.558295557025474,
    "max_y": 40.599429229677526,
}
JAM_CENTER = (101.96786384206956, 40.2325)
JAM_EDGE_PT = (101.84516211958464, 40.2325)
JAM_RADIUS_M = None
BLUE_DIST_CAP = 70000.0

# ==================== 通用工具函数 ====================


# ==================== 训练损失记录与绘图 ====================
# ==================== 训练损失记录与绘图（精简：只画 total/pg/v） ====================
class LossLogger:
    def __init__(self, out_root: str):
        import os
        self.out_root = os.path.abspath(out_root)
        os.makedirs(self.out_root, exist_ok=True)
        self.csv_path = os.path.join(self.out_root, "loss_history.csv")
        self.global_step = 0
        self._csv_inited = False
        # === NEW: per-episode summary ===
        self.ep_csv_path = os.path.join(self.out_root, "episode_summary.csv")
        self._ep_csv_inited = False

    def _ensure_header(self):
        import csv, os
        if self._csv_inited and os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["global_step", "episode", "epoch_in_update", "total_loss", "pg_loss", "v_loss"])
        self._csv_inited = True

    # === NEW: episode summary header ===
    def _ensure_ep_header(self):
        import csv, os
        if self._ep_csv_inited and os.path.exists(self.ep_csv_path):
            return
        with open(self.ep_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "return", "len_steps", "blue_kills", "red_losses"])
        self._ep_csv_inited = True

    def append_update_stats(self, episode_idx: int, out: dict):
        import csv
        stats = (out or {}).get("epoch_stats", []) or []
        if not stats:
            return
        self._ensure_header()
        rows = []
        for ei, s in enumerate(stats, start=1):
            self.global_step += 1
            rows.append([
                self.global_step, int(episode_idx), int(ei),
                float(s.get("loss_total", 0.0)),
                float(s.get("pg_loss", 0.0)),
                float(s.get("v_loss", 0.0)),
            ])
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        # 画损失曲线（里边会顺带尝试叠加 reward）
        try:
            self.plot_curves()
        except Exception as e:
            print("[LossLogger] plot_curves failed:", e, flush=True)

    # === NEW: 每回合结束调用，记录回报 ===
    def append_episode_summary(self, episode_idx: int, ep_return: float, ep_len: int, blue_kills: int, red_losses: int):
        import csv
        self._ensure_ep_header()
        with open(self.ep_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(episode_idx), float(ep_return), int(ep_len), int(blue_kills), int(red_losses)])
        # 画 reward 曲线
        try:
            self.plot_curves()
        except Exception as e:
            print("[LossLogger] plot_curves failed:", e, flush=True)

    def plot_curves(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except Exception:
            pass

        import csv, os
        import matplotlib.pyplot as plt

        xs, total, pg, v = [], [], [], []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(int(row["global_step"]))
                    total.append(float(row["total_loss"]))
                    pg.append(float(row["pg_loss"]))
                    v.append(float(row["v_loss"]))

        def smooth(arr, k=7):
            if k <= 1 or len(arr) < k:
                return arr
            out, win = [], []
            for x in arr:
                win.append(x)
                if len(win) > k: win.pop(0)
                out.append(sum(win) / len(win))
            return out

        if xs:
            plt.figure(figsize=(9, 5), dpi=120)
            ax = plt.gca()
            ax.plot(xs, smooth(total), label="total_loss")
            ax.plot(xs, smooth(pg),    label="pg_loss")
            ax.plot(xs, smooth(v),     label="v_loss")
            ax.set_xlabel("Global step (epoch count across episodes)")
            ax.set_ylabel("Loss")
            ax.set_title("PPO Training Loss (All Episodes)")
            ax.grid(True, alpha=0.3)

            # === NEW: 叠加回合回报（如有） ===
            import os as _os
            if _os.path.exists(self.ep_csv_path):
                ep_ids, ep_returns = [], []
                with open(self.ep_csv_path, "r", encoding="utf-8") as f:
                    r2 = csv.DictReader(f)
                    for row in r2:
                        ep_ids.append(int(row["episode"]))
                        ep_returns.append(float(row["return"]))
                if ep_ids:
                    ax2 = ax.twinx()
                    ax2.plot(ep_ids, smooth(ep_returns), label="episode return", alpha=0.6)
                    ax2.set_ylabel("Return")
                    # 做个简单的双图例
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc="best")

            out_png = os.path.join(self.out_root, "loss_over_episodes.png")
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
            print(f"[LossLogger] Updated plot: {out_png}", flush=True)

        # 再单独存一张纯 reward 图（可选）
        import os as _os
        if _os.path.exists(self.ep_csv_path):
            ep_ids, ep_returns = [], []
            with open(self.ep_csv_path, "r", encoding="utf-8") as f:
                r2 = csv.DictReader(f)
                for row in r2:
                    ep_ids.append(int(row["episode"]))
                    ep_returns.append(float(row["return"]))
            if ep_ids:
                plt.figure(figsize=(9, 4), dpi=120)
                plt.plot(ep_ids, smooth(ep_returns), label="episode return")
                plt.xlabel("Episode")
                plt.ylabel("Return")
                plt.title("Episode Return")
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_png2 = os.path.join(self.out_root, "reward_over_episodes.png")
                plt.tight_layout()
                plt.savefig(out_png2)
                plt.close()
                print(f"[LossLogger] Updated plot: {out_png2}", flush=True)


def _bearing_deg_from_A_to_B(lonA, latA, lonB, latB):
    import math
    if None in (lonA, latA, lonB, latB):
        return None
    φ1, φ2 = math.radians(latA), math.radians(latB)
    λ1, λ2 = math.radians(lonA), math.radians(lonB)
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def _geo_dist_haversine_m(lon1, lat1, lon2, lat2):
    import math
    if None in (lon1, lat1, lon2, lat2):
        return None
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def _direct_rate_from_vx_vy(vx, vy):
    from math import sqrt, atan2, degrees
    try:
        vxn = float(vx) if vx is not None else 0.0
        vye = float(vy) if vy is not None else 0.0
    except Exception:
        vxn, vye = 0.0, 0.0
    direct = sqrt(vxn*vxn + vye*vye)
    rate = (degrees(atan2(vye, vxn)) + 360.0) % 360.0
    return direct, rate

def _ang_norm(deg): return (float(deg) + 360.0) % 360.0

def _ang_diff_abs(a, b):
    d = abs(_ang_norm(a) - _ang_norm(b))
    return d if d <= 180.0 else 360.0 - d

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _cosd(deg):
    import math
    return math.cos(math.radians(float(deg or 0.0)))

def _bearing_deg_A_to_B(lonA, latA, lonB, latB):
    return _bearing_deg_from_A_to_B(lonA, latA, lonB, latB)

def _is_missile_track(tid):
    try:
        return int(tid) < 10000
    except Exception:
        return False

def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    out = {}
    for k in dir(obj):
        if k.startswith('_'): continue
        try: v = getattr(obj, k)
        except Exception: continue
        if callable(v): continue
        out[k] = v
    return out

def _dist_to_boundary_m(client, lon, lat):
    if lon is None or lat is None:
        return (0.0, 0.0, 0.0, 0.0)
    L = Position(x=BOUNDARY_RECT["min_x"], y=lat, z=0)
    R = Position(x=BOUNDARY_RECT["max_x"], y=lat, z=0)
    D = Position(x=lon, y=BOUNDARY_RECT["min_y"], z=0)
    U = Position(x=lon, y=BOUNDARY_RECT["max_y"], z=0)
    P = Position(x=lon, y=lat, z=0)
    try:
        d_left  = client.get_distance_by_lon_lat(L, P)
        d_right = client.get_distance_by_lon_lat(P, R)
        d_down  = client.get_distance_by_lon_lat(D, P)
        d_up    = client.get_distance_by_lon_lat(P, U)
    except Exception:
        from math import cos, radians, sqrt
        def _approx_m(dx_lon, dy_lat, lat_ref):
            dx = dx_lon * 111320.0 * cos(radians(lat_ref))
            dy = dy_lat * 110540.0
            return sqrt(dx*dx + dy*dy)
        d_left  = _approx_m(lon - BOUNDARY_RECT["min_x"], 0, lat)
        d_right = _approx_m(BOUNDARY_RECT["max_x"] - lon, 0, lat)
        d_down  = _approx_m(0, lat - BOUNDARY_RECT["min_y"], lat)
        d_up    = _approx_m(0, BOUNDARY_RECT["max_y"] - lat, lat)
    return (d_left, d_right, d_down, d_up)

def _signed_dist_to_jam_boundary_m(client, lon, lat):
    if lon is None or lat is None:
        return 0.0
    global JAM_RADIUS_M
    C = Position(x=JAM_CENTER[0], y=JAM_CENTER[1], z=0)
    if JAM_RADIUS_M is None:
        E = Position(x=JAM_EDGE_PT[0], y=JAM_EDGE_PT[1], z=0)
        try:
            JAM_RADIUS_M = client.get_distance_by_lon_lat(C, E)
        except Exception:
            JAM_RADIUS_M = 10416.0
    P = Position(x=lon, y=lat, z=0)
    try:
        d_c = client.get_distance_by_lon_lat(C, P)
    except Exception:
        d_c = 0.0
    return d_c - JAM_RADIUS_M

def _parse_track_from_string_rich(s):
    if not isinstance(s, str): return None
    m_id = re.search(r'\btarget_id\s*:\s*(\d+)', s) or re.search(r'\bid\s*:\s*(\d+)', s)
    if not m_id: return None
    tid = int(m_id.group(1))
    m_xy = re.search(r'target\s*_?pos.*?\{[^}]*?x\s*:\s*([-\d\.eE]+)\s*[,， ]+\s*y\s*:\s*([-\d\.eE]+)', s) \
        or re.search(r'\bx\s*:\s*([-\d\.eE]+)\s*[，, ]+\s*y\s*:\s*([-\d\.eE]+)', s)
    lon = float(m_xy.group(1)) if m_xy else None
    lat = float(m_xy.group(2)) if m_xy else None
    m_spd = re.search(r'target[_\s-]?speed\s*:\s*([-\d\.eE]+)', s, re.I)
    spd = float(m_spd.group(1)) if m_spd else None
    m_dir = re.search(r'(target[_\s-]?direction|direction|dir)\s*:\s*([-\d\.eE]+)', s, re.I)
    dire = float(m_dir.group(2)) if m_dir else None
    if dire is not None:
        dire = dire % 360.0
        if dire < 0: dire += 360.0
    return {"target_id": tid, "lon": lon, "lat": lat, "speed": spd, "direction": dire}

def _extract_track_fields_any(t_like):
    td = _as_dict(t_like)
    tid = td.get('target_id') or td.get('id')
    if tid is None: return None
    tid = int(tid)
    pos  = td.get('target_pos') or td.get('target pos') or {}
    posd = _as_dict(pos)
    lon  = posd.get('x'); lat = posd.get('y')
    spd  = td.get('target_speed') or td.get('speed')
    dire = td.get('target_direction') or td.get('target direction') or td.get('direction') or td.get('dir')
    try:
        if spd is not None: spd  = float(spd)
    except Exception: spd = None
    try:
        if dire is not None:
            dire = float(dire) % 360.0
            if dire < 0: dire += 360.0
    except Exception: dire = None
    return {"target_id": tid, "lon": lon, "lat": lat, "speed": spd, "direction": dire}

def normalize_visible(visible):
    out = {}
    if not isinstance(visible, dict): return out
    for detector_id, v in visible.items():
        try: detector_id = int(detector_id)
        except Exception: pass
        tracks = []
        if v is None:
            pass
        elif isinstance(v, list):
            for t in v:
                if isinstance(t, dict) or hasattr(t, '__dict__'):
                    item = _extract_track_fields_any(t)
                else:
                    item = _parse_track_from_string_rich(t)
                if item: tracks.append(item)
        elif isinstance(v, dict):
            if 'target_id' in v or 'id' in v:
                item = _extract_track_fields_any(v)
                if item: tracks.append(item)
            else:
                for _, t in v.items():
                    item = _extract_track_fields_any(t)
                    if item: tracks.append(item)
        else:
            item = _parse_track_from_string_rich(v)
            if item: tracks.append(item)
        out[int(detector_id)] = tracks
    return out

class _BC_MLP(_nn.Module):
    def __init__(self, in_dim, cont_dim, bin_dim, hidden=(256,256), dropout=0.0):
        super().__init__()
        layers=[]; last=in_dim
        for h in hidden:
            layers += [_nn.Linear(last, h), _nn.ReLU()]
            if dropout>0: layers += [_nn.Dropout(dropout)]
            last=h
        self.shared = _nn.Sequential(*layers) if layers else _nn.Identity()
        self.cont_head = _nn.Linear(last, cont_dim) if cont_dim>0 else None
        self.bin_head  = _nn.Linear(last, bin_dim)  if bin_dim>0 else None
    def forward(self, x):
        z = self.shared(x); out={}
        if self.cont_head is not None: out["cont"] = self.cont_head(z)
        if self.bin_head  is not None: out["bin_logits"] = self.bin_head(z)
        return out

class BCPredictor64x16:
    def __init__(self, out_dir: str):
        meta = _json.load(open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8"))
        self.obs_cols   = meta["obs_cols"]
        self.act_cols   = meta["act_cols"]
        self.cont_cols  = meta["cont_cols"]
        self.binary_cols= meta["binary_cols"]
        m = meta["model"]
        self.model = _BC_MLP(m["in_dim"], m["cont_dim"], m["bin_dim"],
                             hidden=tuple(m.get("hidden",[256,256])), dropout=m.get("dropout",0.0))
        self.model.load_state_dict(_torch.load(os.path.join(out_dir, "policy.pt"), map_location="cpu"))
        self.model.eval()
        self.obs_mean = _np.array([meta["obs_mean"][c] for c in self.obs_cols], dtype=_np.float32)
        self.obs_std  = _np.array([meta["obs_std"][c]  for c in self.obs_cols], dtype=_np.float32)
        if self.cont_cols:
            self.cont_mean = _np.array([meta["cont_mean"][c] for c in self.cont_cols], dtype=_np.float32)
            self.cont_std  = _np.array([meta["cont_std"][c]  for c in self.cont_cols], dtype=_np.float32)
        else:
            self.cont_mean = None; self.cont_std = None
        self._obs_idx = {c:i for i,c in enumerate(self.obs_cols)}
        self._act_idx = {c:i for i,c in enumerate(self.act_cols)}
        self._cont_idx= {c:i for i,c in enumerate(self.cont_cols)} if self.cont_cols else {}
        self._bin_idx = {c:i for i,c in enumerate(self.binary_cols)} if self.binary_cols else {}
    def predict_full(self, obs_dict: dict) -> _np.ndarray:
        x = _np.array([float(obs_dict[c]) for c in self.obs_cols], dtype=_np.float32)
        x = (x - self.obs_mean) / self.obs_std
        with _torch.no_grad():
            out = self.model(_torch.from_numpy(x).unsqueeze(0))
        result = _np.zeros((len(self.act_cols),), dtype=_np.float32)
        if self.cont_cols:
            cont_norm = out["cont"].cpu().numpy()[0]
            cont = cont_norm * self.cont_std + self.cont_mean
            for c, i in self._cont_idx.items():
                result[self._act_idx[c]] = cont[i]
        if self.binary_cols:
            logits = out["bin_logits"].cpu().numpy()[0]
            probs = 1.0 / (1.0 + _np.exp(-logits))
            for c, i in self._bin_idx.items():
                result[self._act_idx[c]] = 1.0 if probs[i] >= 0.5 else 0.0
        return result

# ======= Runtime: stateful seq predictor (保留用于列名/对齐；动作由 PPO 决定) =======
class _RNNRuntime(_nn.Module):
    def __init__(self, in_dim, cont_dim, bin_dim, rnn_type, hidden_size, num_layers, dropout):
        super().__init__()
        rnn_cls = {"gru": _nn.GRU, "lstm": _nn.LSTM}[rnn_type]
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(in_dim, hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head_c = _nn.Linear(hidden_size, cont_dim) if cont_dim > 0 else None
        self.head_b = _nn.Linear(hidden_size, bin_dim) if bin_dim > 0 else None
    def forward(self, x, h=None):
        z, h = self.rnn(x, h)
        out = {}
        if self.head_c is not None: out["cont"] = self.head_c(z)
        if self.head_b is not None: out["bin_logits"] = self.head_b(z)
        return out, h

class BCSeqPredictor64x16:
    def __init__(self, out_dir: str, device: str = "cpu"):
        meta_path = os.path.join(out_dir, "seq_meta.json")
        pol_path  = os.path.join(out_dir, "seq_policy.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"seq_meta.json not found in {out_dir}")
        meta = _json.load(open(meta_path, "r", encoding="utf-8"))
        m = meta["model"]
        self.obs_cols  = meta["obs_cols"]
        self.act_cols  = meta["act_cols"]
        self.cont_cols = meta["cont_cols"]
        self.bin_cols  = meta["binary_cols"]
        self.obs_mean = _np.array([meta["obs_mean"][c] for c in self.obs_cols], dtype=_np.float32)
        self.obs_std  = _np.array([meta["obs_std"][c]  for c in self.obs_cols], dtype=_np.float32)
        self.cont_mean = _np.array([meta["cont_mean"].get(c,0) for c in self.cont_cols], dtype=_np.float32) if self.cont_cols else None
        self.cont_std  = _np.array([meta["cont_std"].get(c,1)  for c in self.cont_cols], dtype=_np.float32) if self.cont_cols else None
        self.time_feat_dim = int(meta["time_feat"]["dim"])
        self.model = _RNNRuntime(in_dim=m["in_dim"], cont_dim=m["cont_dim"], bin_dim=m["bin_dim"],
                                 rnn_type=m["rnn_type"], hidden_size=m["hidden_size"],
                                 num_layers=m["num_layers"], dropout=m["dropout"])
        state = _torch.load(pol_path, map_location=device)
        new_state = {}
        for k, v in state.items():
            if k.startswith("head_cont."): new_state[k.replace("head_cont.", "head_c.")] = v
            elif k.startswith("head_bin."): new_state[k.replace("head_bin.", "head_b.")] = v
            else: new_state[k] = v
        self.model.load_state_dict(new_state, strict=True)
        self.model.eval()
        self.device = _torch.device(device)
        self.h = None
        self.t0 = None
        self.tspan = 300.0
    def reset_episode(self, t0: float = 0.0, tspan_hint: float = None):
        self.h = None
        self.t0 = float(t0)
        if tspan_hint and tspan_hint > 0: self.tspan = float(tspan_hint)
    def _time_feat(self, t: float):
        if self.t0 is None: self.t0 = float(t)
        t_norm = float(t - self.t0) / max(1.0, self.tspan)
        feats = [t_norm, math.sin(2*math.pi*t_norm), math.cos(2*math.pi*t_norm)]
        return _np.array(feats[:self.time_feat_dim], dtype=_np.float32)
    def step(self, obs_dict: dict, t: float):
        x = _np.array([float(obs_dict[c]) for c in self.obs_cols], dtype=_np.float32)
        x = (x - self.obs_mean) / self.obs_std
        tf = self._time_feat(t)
        x = _np.concatenate([x, tf], axis=0)[None, None, :]
        with _torch.no_grad():
            out, self.h = self.model(_torch.from_numpy(x).to(self.device), self.h)
        result = _np.zeros((len(self.act_cols),), dtype=_np.float32)
        if self.cont_cols:
            cont = out["cont"].cpu().numpy()[0,0,:]
            cont = cont * (self.cont_std if self.cont_std is not None else 1.0) + (self.cont_mean if self.cont_mean is not None else 0.0)
            for i,c in enumerate(self.cont_cols):
                idx = self.act_cols.index(c); result[idx] = float(cont[i])
        if self.bin_cols:
            logits = out["bin_logits"].cpu().numpy()[0,0,:]
            probs = 1.0/(1.0+_np.exp(-logits))
            for i,c in enumerate(self.bin_cols):
                idx = self.act_cols.index(c); result[idx] = 1.0 if probs[i] >= 0.5 else 0.0
        return result

# ==================== 记录器 ====================
class RLRecorder:
    def __init__(self):
        self.buffer = []
        self.last_action = {}
        self.attack_events = defaultdict(list)
        self.last_vel_cmd = {}
        self.rows_for_csv = []
        self.latest_obs_vec = None
        self.latest_act_vec = None
        self.has_dumped = False

    def pack_single_obs19(self, client, rid, obs, boundary_rect,
                          jam_signed_dist, blue4_dists,
                          nearest_blue_speed, nearest_blue_dir,
                          nearest_msl_dist=None, nearest_msl_dir=None,
                          rel_angle_to_msl=None, vel_meas=None):
        """
        输出 19 维单机观测：
        0-3   : 到边界的距离 (left, right, down, up)
        4-6   : vx, vy, vz
        7-10  : 最近4个蓝机的距离（不足补 cap）
        11    : 最近蓝机速度
        12    : 最近蓝机方向
        13    : 禁飞区边界的有符号距离
        14    : 剩余弹药数
        15    : 自身平面速度标量
        16    : 最近导弹的距离
        17    : 最近导弹的方向
        18    : 导弹相对角度（导弹方向 vs 飞机朝向）
        """
        left, right, down, up = _dist_to_boundary_m(client, obs["pos"]["lon"], obs["pos"]["lat"])
        vx, vy, vz_meas = obs["vel"]["vx"], obs["vel"]["vy"], obs["vel"]["vz"]
        ammo = obs["ammo"]

        # 基础 16 维
        obs16 = [
            left, right, down, up,
            vx if vx is not None else 0.0,
            vy if vy is not None else 0.0,
            vz_meas if vz_meas is not None else 0.0,
            *blue4_dists,
            nearest_blue_speed if nearest_blue_speed is not None else 0.0,
            nearest_blue_dir if nearest_blue_dir is not None else 0.0,
            jam_signed_dist if jam_signed_dist is not None else 0.0,
            ammo if ammo is not None else 0
        ]
        # 补 speed_scalar
        from math import sqrt
        speed_scalar = sqrt(float(vx or 0.0) ** 2 + float(vy or 0.0) ** 2)
        obs16.append(speed_scalar)

        # 加入导弹相关 3 维
        obs16.append(nearest_msl_dist if nearest_msl_dist is not None else 0.0)
        obs16.append(nearest_msl_dir if nearest_msl_dir is not None else 0.0)
        obs16.append(rel_angle_to_msl if rel_angle_to_msl is not None else 0.0)

        assert len(obs16) == 19, f"expect 19 obs, got {len(obs16)}"
        return obs16

    def pack_single_act4(self, rid, sim_sec, vel_meas, pos_meas):
        vcmd = self.last_vel_cmd.get(int(rid))
        att_times = self.attack_events.get(int(rid), [])
        attack_flag = 1 if any((sim_sec - 1) < t <= sim_sec for t in att_times) else 0
        def nz(x, default):
            try: return float(x)
            except Exception: return default

        if vcmd and (sim_sec - int(vcmd["t"])) <= 1:
            speed_cmd = nz(vcmd.get("rate"), 0.0)  # 速度
            heading_cmd = nz(vcmd.get("direct"), 0.0) % 360.0  # 航向角
            vz_cmd = nz(vcmd.get("vz"), nz(getattr(pos_meas, "z", None), 0.0))
            # 约定的 act4: [heading(角度), speed(速度), vz, fire]
            return [heading_cmd, speed_cmd, vz_cmd, int(attack_flag)]

        vx = getattr(vel_meas, "vx", None) if vel_meas else None
        vy = getattr(vel_meas, "vy", None) if vel_meas else None
        direct_calc, rate_calc = _direct_rate_from_vx_vy(vx, vy)
        vz_fallback = nz(getattr(pos_meas, "z", None), 0.0)
        return [rate_calc, direct_calc, vz_fallback, int(attack_flag)]
    def add_vector_row(self, sim_sec, obs_vec, act_vec):
        self.rows_for_csv.append([int(sim_sec)] + list(obs_vec) + list(act_vec))
    def mark_action(self, red_id, action_dict, sim_time):
        ad = dict(action_dict); ad["t"] = float(sim_time)
        self.last_action[int(red_id)] = ad
    def mark_attack_success(self, red_id, sim_time):
        rid = int(red_id)
        self.attack_events[rid].append(float(sim_time))
        self.attack_events[rid] = [t for t in self.attack_events[rid] if sim_time - t <= 10.0]
    def mark_vel_cmd(self, red_id, rate, direct, vz, sim_time):
        self.last_vel_cmd[int(red_id)] = {
            "rate": float(rate) if rate is not None else None,
            "direct": float(direct) if direct is not None else None,
            "vz": float(vz) if vz is not None else None,
            "t": float(sim_time)
        }
    def record_tick(self, sim_sec, red_id, obs_dict, extra_info=None):
        a = self.last_action.get(int(red_id))
        rec_action = (dict(a) | {"age_sec": int(sim_sec) - int(a.get("t", sim_sec))}) if a else None
        self.buffer.append({"t": int(sim_sec), "red_id": int(red_id), "obs": obs_dict, "action": rec_action, "info": (extra_info or {})})
    def dump_csv(self, path):
        if self.has_dumped: return
        if not self.rows_for_csv:
            print("[RL] No vector data to dump:", path, flush=True)
            self.has_dumped = True; return
        n_act = 16
        n_obs = len(self.rows_for_csv[0]) - 1 - n_act
        header = ["t"] + [f"obs_{i}" for i in range(n_obs)] + [f"act_{j}" for j in range(n_act)]
        abspath = os.path.abspath(path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(self.rows_for_csv)
        print(f"[RL] Dumped {len(self.rows_for_csv)} rows to {abspath}", flush=True)
        self.has_dumped = True

# ==================== 红方控制器 ====================
class RedForceController:
    def __init__(self, client, red_ids, out_csv_path):
        self.client = client
        self.red_ids = set(int(x) for x in red_ids)
        self.ammo = {int(r): AMMO_MAX for r in red_ids}
        self.last_fire_time = {int(r): 0.0 for r in red_ids}
        self.assigned_target = {}
        self.target_in_progress = set()
        self.destroyed_targets = set()
        self.destroyed_blue = set()
        self.destroyed_red = set()
        self._ended = False
        self.lock = threading.Lock()
        self.target_locks = {}
        self._last_visible = set()
        self._last_locked = {}
        self._last_destroyed = set()
        self.recorder = RLRecorder()
        self._last_logged_sec = -1
        self._score_cache = None
        self._score_counter = 0
        self.out_csv_path = out_csv_path

        # --- 调试/导弹 ---
        self.DEBUG_EVADE = True
        self.MISSILE_DEBUG_PRINT_INTERVAL = 0.5
        self._last_debug_print = {}
        self._last_approach_flag = {}
        self.MISSILE_THREAT_DIST_M = 50000.0
        self.MISSILE_BEARING_THRESH_DEG = 25.0
        self.EVASIVE_TURN_DEG = 90.0
        self.EVASIVE_SPEED_MIN = 160.0
        self.EVASIVE_DURATION_SEC = 30.0
        self.EVASIVE_COOLDOWN_SEC = 2.0
        self._missile_last_dist = {}
        self._missile_last = {}
        self._evasive_until = {}
        self._last_evasive_time = {}
        # --- 全蓝已被锁定 -> 让最东边红机后撤机动 ---
        self.EAST_MANEUVER_COOLDOWN_SEC = 10.0   # 触发后冷却，避免每秒重复触发
        self.EAST_MANEUVER_DURATION_SEC = 20.0   # （如需扩展为持续窗口，可用）
        self._east_maneuver_cooldown_until = 0.0 # 冷却时间戳（用 sim_time）
        self._east_maneuver_until = {}  # 新增：rid -> sim_time 截止时间
        # --- 结束延迟 ---
        self._end_grace_until = None
        self._end_reason = ""

        # ======= BC 标志位仍在，但不再用 BC 下发速度（动作交给 PPO）=======
        self._bc: Optional[BCPredictor64x16] = None
        self._bc_seq: Optional[BCSeqPredictor64x16] = None
        self._bc_mode = "none"
        self._bc_out_dir: Optional[str] = None

        # —— PPO 初始化 & 轨迹缓存 / 奖励权重 ——
        self.ppo = PPOAgent(bc_dir="./bc_out_seq", init_load_dir="./bc_out_seq", device="cpu")
        self.traj = []                       # list of dict(obs,tfeat,act,val,logp,rew,done)
        self._last_step_sec = None           # 上一条样本对应的 sim_sec
        self._last_counts = {"blue":0, "red":0}  # 用于计算“增量击毁”
        self.REW_BLUE_KILL = +1.0            # 击毁蓝
        self.REW_RED_LOSS = -1.0             # 自身被毁
        self.REW_NOFLY_PER_SEC = -0.05       # 每架红禁飞区内每秒惩罚
        # 策略攻击意图缓存（由 PPO 每秒产出，供两阶段打击用）
        self._policy_attack_intent = {int(r): 0 for r in red_ids}
        # 本秒是否已跑过策略（用于“每秒只下发一次”）
        self._last_policy_sec = -1
        # 速度上限（动作与规避都会用来 clamp）
        self.SPEED_CAP = 200.0

        # 作战速度带（用于奖励 shaping）
        self.SPEED_MIN = 80.0
        self.SPEED_MAX = 200.0

        # 规避时的最低提速目标（必须 ≤ 上限）
        self.EVASIVE_SPEED_MIN = 160.0

        # === 新增：初始化 GRU 隐状态（B=1） ===
        self._ppo_h = None


        # === Shaping & Regularization ===
        self.gamma = 0.99

        # 奖励权重（可按需微调；单步总 shaping 建议<=0.05~0.1）
        self.lambda_ttl      = -0.001   # 生存时间成本（越快结束越好）
        self.alpha_phi       = 0.6      # 势函数：距离权重
        self.beta_phi        = 0.4      # 势函数：指向权重（cos对准）
        self.lambda_lock_pos = +0.01    # 持续专注同一目标（每秒）
        self.lambda_switch   = -0.02    # 频繁换目标（每次）
        self.lambda_speed_ok = +0.005   # 速度落在作战带
        self.lambda_speed_ng = -0.005   # 过慢或过快
        self.lambda_vz_jitter= -0.002   # 垂直速度抖动罚
        self.lambda_act_smooth = -0.003 # 动作抖动罚（相邻秒动作差的L2）
        self.lambda_evade_good = +0.01  # 在威胁下朝远离/安全几何演化
        self.lambda_evade_bad  = -0.02  # 在威胁下仍迎向/几何恶化
        # === 探测网格覆盖奖励 ===
        self.grid_N_lon = 100   # 横向格数
        self.grid_N_lat = 100   # 纵向格数
        self.radar_radius = 0.05  # 度 (约5km，可调)
        self.visited_map = [[False for _ in range(self.grid_N_lat)] for _ in range(self.grid_N_lon)]
        self.reward_per_cell = 0.05
        self.max_reward_per_step = 2.0


        # 速度带（米/秒）——按你仿真调参区间改
        self.SPEED_MIN = 80.0
        self.SPEED_MAX = 200.0

        # shaping 运行时缓存
        self._phi_prev = 0.0                 # 上一步势函数
        self._last_act_full = None           # 上一步 PPO act 向量（用于动作平滑）
        self._last_vz = {}                   # 上一步各机 vz
        self._focus_target = {}              # rid -> (last_tid, consecutive_secs)
        self._switch_count_step = 0          # 本步换目标次数
        self._focused_secs_step = 0.0        # 本步累计专注秒数（归一到机数）
        self._evade_good_last = 0.0          # 上一步规避“好”秒数
        self._evade_bad_last  = 0.0          # 上一步规避“坏”秒数

        for rid in red_ids:
            try:
                uid = self.client.enable_radar(vehicle_id=rid, state=1)
                print(f"[Red] Radar ON for {rid}, uid={uid}", flush=True)
            except Exception as e:
                print(f"[Red] enable_radar({rid}) failed: {e}", flush=True)
            try:
                vuid = self.client.set_vehicle_vel(rid, vel_0)
                print(f"[Red] vel set for {rid}, uid={vuid}", flush=True)
                time.sleep(0.2)
                print(client.get_command_status(vuid))
            except Exception as e:
                print(f"[Red] set_vehicle_vel({rid}) failed: {e}", flush=True)





    def _coord_to_cell(self, lon, lat):
        if lon is None or lat is None:
            return None
        x_ratio = (lon - BOUNDARY_RECT["min_x"]) / (BOUNDARY_RECT["max_x"] - BOUNDARY_RECT["min_x"])
        y_ratio = (lat - BOUNDARY_RECT["min_y"]) / (BOUNDARY_RECT["max_y"] - BOUNDARY_RECT["min_y"])
        if not (0 <= x_ratio <= 1 and 0 <= y_ratio <= 1):
            return None
        ix = int(x_ratio * self.grid_N_lon)
        iy = int(y_ratio * self.grid_N_lat)
        return (ix, iy)
    def _exploration_reward(self, lon, lat):
        from math import radians, cos
        new_cells = []
        cell = self._coord_to_cell(lon, lat)
        if cell is None:
            return 0.0
        cx, cy = cell
        # 粗糙：取雷达半径对应的格子范围
        d_lon = int(self.radar_radius / ((BOUNDARY_RECT["max_x"] - BOUNDARY_RECT["min_x"]) / self.grid_N_lon))
        d_lat = int(self.radar_radius / ((BOUNDARY_RECT["max_y"] - BOUNDARY_RECT["min_y"]) / self.grid_N_lat))
        for dx in range(-d_lon, d_lon + 1):
            for dy in range(-d_lat, d_lat + 1):
                ix, iy = cx + dx, cy + dy
                if 0 <= ix < self.grid_N_lon and 0 <= iy < self.grid_N_lat:
                    if not self.visited_map[ix][iy]:
                        self.visited_map[ix][iy] = True
                        new_cells.append((ix, iy))
        if new_cells:
            reward = min(self.max_reward_per_step, self.reward_per_cell * len(new_cells))
            return reward
        return 0.0

    # ---- BC 装载（可选，保留不使用） ----
    def _resolve_latest_seq_dir(self, root):
        import glob
        candidates = []
        if os.path.exists(os.path.join(root, "seq_meta.json")) and os.path.exists(os.path.join(root, "seq_policy.pt")):
            candidates.append(root)
        for d in sorted(glob.glob(os.path.join(root, "*")), reverse=True):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "seq_meta.json")) and os.path.exists(os.path.join(d, "seq_policy.pt")):
                candidates.append(d)
        return candidates[0] if candidates else None

    def attach_bc_policy(self, bc_out_dir: str):
        import traceback
        self._bc_out_dir = bc_out_dir
        self._bc = None; self._bc_seq = None; self._bc_mode = "none"
        try:
            print(f"[BC] Try load from: {bc_out_dir}", flush=True)
            real_seq_dir = self._resolve_latest_seq_dir(bc_out_dir)
            if real_seq_dir is not None:
                print(f"[BC] Found sequence policy under: {real_seq_dir}", flush=True)
                self._bc_seq = BCSeqPredictor64x16(real_seq_dir, device="cpu")
                self._bc_mode = "seq"
                print(f"[BC] Loaded SEQ policy ok: {real_seq_dir}", flush=True)
                return
            if os.path.exists(os.path.join(bc_out_dir, "meta.json")):
                print("[BC] Found MLP meta (meta.json). Loading MLP policy...", flush=True)
                self._bc = BCPredictor64x16(bc_out_dir)
                self._bc_mode = "mlp"
                print(f"[BC] Loaded MLP policy ok from {bc_out_dir}", flush=True)
                return
            print(f"[BC-WARN] No seq_meta.json or meta.json under {bc_out_dir}.", flush=True)
        except Exception as e:
            print(f"[BC] load failed: {e}", flush=True)
            traceback.print_exc()
            self._bc = None; self._bc_seq = None; self._bc_mode = "none"

    def _estimate_msl_heading_speed(self, mid, lon_now, lat_now, t_now):
        prev = self._missile_last.get(int(mid))
        self._missile_last[int(mid)] = {"lon": float(lon_now), "lat": float(lat_now), "t": float(t_now)}
        if not prev: return (None, None)
        dt = float(t_now) - float(prev["t"])
        if dt <= 1e-3: return (None, None)
        mdir = _bearing_deg_from_A_to_B(prev["lon"], prev["lat"], lon_now, lat_now)
        dist = _geo_dist_haversine_m(prev["lon"], prev["lat"], lon_now, lat_now) or 0.0
        mspeed = dist / dt
        return (mdir, mspeed)

    def _is_blue_side(self, side):
        return side in (2, "SIDE_BLUE", "BLUE")

    def _is_destroyed_flag(self, dmg):
        if isinstance(dmg, (int, float)): return int(dmg) == 1
        s = str(dmg)
        return ("DESTROYED" in s.upper()) or (s.strip() == "1")

    def _is_fire_success(self, uid, max_tries=5, wait_s=0.1):
        if not uid: return False
        for _ in range(max_tries):
            time.sleep(wait_s)
            try: st = self.client.get_command_status(uid)
            except Exception: st = None
            if isinstance(st, dict):
                status = st.get('status', '')
                result = st.get('execute_result', '') or ''
                if status == 'EXECUTE_SUCCESS' or ('执行打击成功' in result): return True
                if status == 'EXECUTE_FAILED': return False
        return False

    def _fire_with_log(self, rid, tid):
        uid = self.client.set_target(vehicle_id=rid, target_id=tid)
        sim_t = self.client.get_sim_time()
        self.recorder.mark_action(rid, {"type": "fire", "target_id": int(tid), "uid": uid}, sim_t)
        return uid

    def _distance_m(self, lon1, lat1, lon2, lat2):
        d = _geo_dist_haversine_m(lon1, lat1, lon2, lat2)
        return float(d) if d is not None else 1e18

    def _get_target_pos(self, target_id, vis, all_pos):
        p = all_pos.get(target_id)
        if p is not None: return p.x, p.y
        for tracks in vis.values():
            for t in tracks:
                if t.get("target_id") == target_id:
                    return t.get("lon"), t.get("lat")
        return (None, None)

    def _maybe_maneuver_eastmost(self, visible_blue_targets, all_pos, vel_all, now_sim):
        """
        条件：所有“当前可见的蓝机”都已在 self.target_locks 里（即都被锁定）。
        行为：选择最靠东（lon 最大）的在存活/未规避的红机，立即反向加速到上限。
        """
        # 1) 条件判定：可见蓝机非空 且 全部在锁定表中
        if not visible_blue_targets:
            return
        all_locked = all(tid in self.target_locks for tid in visible_blue_targets)
        if not all_locked:
            return

        # 冷却限制
        if now_sim < float(self._east_maneuver_cooldown_until or 0.0):
            return

        # 2) 选最靠东的红机（lon 最大）
        east_rid, east_lon = None, None
        for rid in sorted(self.red_ids):
            if rid in self.destroyed_red:
                continue
            if self._evasive_until.get(rid, 0.0) > now_sim:
                continue
            rp = all_pos.get(rid)
            if not rp or getattr(rp, "x", None) is None:
                continue
            lon = float(rp.x)
            if (east_lon is None) or (lon > east_lon):
                east_lon, east_rid = lon, rid

        if east_rid is None:
            return

        # 3) 计算当前航向并反向（+180°），速度打到上限
        v_me = vel_all.get(east_rid) if vel_all else None
        _, heading = _direct_rate_from_vx_vy(
            getattr(v_me, "vx", 0.0) if v_me else 0.0,
            getattr(v_me, "vy", 0.0) if v_me else 0.0
        )
        reverse_heading = _ang_norm((heading if heading is not None else 0.0) + 180.0)
        speed_cmd = float(self.SPEED_CAP)
        vz_keep = float(getattr(v_me, "vz", 0.0) if v_me else 0.0)

        try:
            vcmd = rflysim.Vel()
            vcmd.rate   = speed_cmd              # 速度
            vcmd.direct = reverse_heading        # 航向：反向
            vcmd.vz     = vz_keep                # 保持当前升降
            uid = self.client.set_vehicle_vel(east_rid, vcmd)
            self._east_maneuver_until[east_rid] = float(now_sim) + float(self.EAST_MANEUVER_DURATION_SEC)
            self._east_maneuver_cooldown_until = float(now_sim) + float(self.EAST_MANEUVER_COOLDOWN_SEC)

            self.recorder.mark_vel_cmd(east_rid, rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=now_sim)
            print(f"[EAST-MANEUVER] rid={east_rid} lon={east_lon:.6f} -> reverse to {vcmd.direct:.1f}°, speed={vcmd.rate:.1f}", flush=True)
        except Exception as e:
            print(f"[EAST-MANEUVER] set_vehicle_vel({east_rid}) failed: {e}", flush=True)

        # 4) 设置冷却，避免下一秒又触发
        self._east_maneuver_cooldown_until = float(now_sim) + float(self.EAST_MANEUVER_COOLDOWN_SEC)


    def _fetch_score_with_retry(self, tries=20, wait=0.2, where="(unknown)"):
        score_obj = None
        for _ in range(tries):
            try: score_obj = self.client.get_score()
            except Exception: score_obj = None
            if score_obj:
                print(f"[Final Score {where}]", score_obj, flush=True)
                return score_obj
            time.sleep(wait)
        print(f"[Final Score {where}] still None after retry.", flush=True)
        return None

    def _end_simulation_once(self, reason=""):
        if self._ended: return
        self._ended = True
        if reason: print(f"[AutoStop] {reason}", flush=True)
        self._fetch_score_with_retry(tries=5, wait=0.2, where="pre-stop")
        try: self.client.stop()
        except Exception as e: print("[AutoStop] client.stop() failed:", e, flush=True)
        self._fetch_score_with_retry(tries=20, wait=0.2, where="post-stop")
        try: self.recorder.dump_csv(self.out_csv_path)
        except Exception as e: print("[RL] dump_csv failed:", e, flush=True)

    def _update_destroyed_from_situ(self, force=False):
        if self._ended: return
        if not force:
            self._situ_counter = getattr(self, "_situ_counter", 0) + 1
            if self._situ_counter % 5 != 0: return
        try: situ_raw = self.client.get_situ_info() or {}
        except Exception: return
        any_new = False
        for vid, info in situ_raw.items():
            if not info: continue
            side = getattr(info, "side", None) if not isinstance(info, dict) else info.get("side")
            dmg  = getattr(info, "damage_state", None) if not isinstance(info, dict) else info.get("damage_state")
            try:
                vvid = getattr(info, "id", None)
                if vvid is None and isinstance(info, dict): vvid = info.get("id")
                if vvid is None: vvid = int(vid)
                else: vvid = int(vvid)
            except Exception:
                continue
            if not self._is_destroyed_flag(dmg): continue
            if self._is_blue_side(side):
                if vvid not in self.destroyed_blue:
                    self.destroyed_blue.add(vvid); self.destroyed_targets.add(vvid); any_new = True
            else:
                if vvid not in self.destroyed_red:
                    self.destroyed_red.add(vvid); any_new = True
                    if vvid in self.ammo: self.ammo[vvid] = 0
                    self.target_locks.pop(vvid, None)
        if any_new:
            print(f"[Situ] BLUE destroyed: {sorted(self.destroyed_blue)} | RED destroyed: {sorted(self.destroyed_red)}", flush=True)
        import time as _time
        now_wall = _time.time()
        if (len(self.destroyed_blue) >= 4 or len(self.destroyed_red) >= 4) and self._end_grace_until is None:
            self._end_reason = "All 4 BLUE aircraft destroyed." if len(self.destroyed_blue) >= 4 else "All 4 RED aircraft destroyed."
            self._end_grace_until = now_wall + 1.0
            print(f"[Grace] {self._end_reason} Ending in 20s...", flush=True)
            return
        if self._end_grace_until is not None:
            if now_wall >= self._end_grace_until:
                self._end_simulation_once(self._end_reason)

    # ---- 主一步 ----
    def step(self):
        if self._ended: return
        try: raw_visible = self.client.get_visible_vehicles()
        except Exception as e: print("[Red] get_visible_vehicles() failed:", e, flush=True); return
        vis = normalize_visible(raw_visible)
        try: all_pos = self.client.get_vehicle_pos()
        except Exception: all_pos = {}
        try: vel_all = self.client.get_vehicle_vel()
        except Exception: vel_all = {}

        self._update_destroyed_from_situ()
        sim_t = self.client.get_sim_time()
        sim_sec = int(sim_t)
        self._score_counter = getattr(self, "_score_counter", 0) + 1
        if self._score_counter % 5 == 0:
            try: self._score_cache = self.client.get_score() or None
            except Exception: self._score_cache = None

        # === 记录 obs/act（act 为上一秒命令或实测兜底）===
        if sim_sec != getattr(self, "_last_logged_sec", -1):
            obs_concat = []; act_concat = []
            for rid in sorted(self.red_ids):
                my_p = all_pos.get(rid)
                my_lon = getattr(my_p, "x", None) if my_p else None
                my_lat = getattr(my_p, "y", None) if my_p else None
                tracks = vis.get(rid, []) or []
                dlist = []
                for t in tracks:
                    lonB, latB = t.get("lon"), t.get("lat")
                    if lonB is None or latB is None or my_lon is None or my_lat is None: continue
                    try:
                        d = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                Position(x=lonB, y=latB, z=0))
                    except Exception: d = None
                    if d is not None: dlist.append(float(d))
                dlist.sort()
                if len(dlist) < 4: dlist += [BLUE_DIST_CAP] * (4 - len(dlist))
                else: dlist = dlist[:4]

                nb_speed, nb_dir = None, None
                if tracks and my_lon is not None and my_lat is not None:
                    best, best_t = None, None
                    for t in tracks:
                        lonB, latB = t.get("lon"), t.get("lat")
                        if lonB is None or latB is None: continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                     Position(x=lonB, y=latB, z=0))
                        except Exception: dd = None
                        if dd is not None and (best is None or dd < best): best, best_t = dd, t
                    if best_t:
                        nb_speed = best_t.get("speed")
                        nb_dir = best_t.get("direction")
                        if nb_dir is None:
                            nb_dir = _bearing_deg_from_A_to_B(my_lon, my_lat, best_t.get("lon"), best_t.get("lat"))

                jam_signed = _signed_dist_to_jam_boundary_m(self.client, my_lon, my_lat)
                vel_meas = vel_all.get(rid)
                obs_raw = {
                    "pos": {"lon": my_lon, "lat": my_lat, "alt": getattr(my_p, "z", None) if my_p else None},
                    "vel": {"vx": getattr(vel_meas, "vx", None) if vel_meas else None,
                            "vy": getattr(vel_meas, "vy", None) if vel_meas else None,
                            "vz": getattr(vel_meas, "vz", None) if vel_meas else None},
                    "ammo": int(self.ammo.get(rid, 0)),
                }

                # === 最近导弹的 3 个特征：距离 / 导弹航向 / 我机与导弹航向的夹角 ===
                nearest_msl_dist, nearest_msl_dir, rel_angle_to_msl = None, None, None

                # 我机当前航向（由 vx, vy 计算）
                my_heading_deg = None
                if vel_meas is not None:
                    _, my_heading_deg = _direct_rate_from_vx_vy(getattr(vel_meas, "vx", 0.0),
                                                                getattr(vel_meas, "vy", 0.0))

                # 注意不要用变量名 tracks 作为元素名，避免遮蔽
                for tr in vis.get(rid, []):
                    tid = tr.get("target_id")
                    if tid is None or int(tid) >= 10000:  # 10000 以下按导弹
                        continue
                    mlon, mlat = tr.get("lon"), tr.get("lat")
                    if mlon is None or mlat is None or my_lon is None or my_lat is None:
                        continue

                    try:
                        d_msl = self.client.get_distance_by_lon_lat(
                            Position(x=my_lon, y=my_lat, z=0),
                            Position(x=mlon, y=mlat, z=0)
                        )
                    except Exception:
                        d_msl = None

                    if d_msl is None:
                        continue
                    if (nearest_msl_dist is None) or (d_msl < nearest_msl_dist):
                        nearest_msl_dist = float(d_msl)

                        # 1) 先取雷达给的航向
                        mdir = tr.get("direction")
                        # 2) 如果没有，就用历史两点估算导弹航向（你上面封装好的函数）
                        if mdir is None:
                            mdir, _ = self._estimate_msl_heading_speed(int(tid), float(mlon), float(mlat), float(sim_t))
                        if mdir is None:
                            # 3) 再不行，用导弹->我机的 LOS 方向（凑合一个）
                            mdir = _bearing_deg_from_A_to_B(mlon, mlat, my_lon, my_lat)
                        nearest_msl_dir = float(mdir) if mdir is not None else 0.0

                        # 相对角度：导弹航向 vs 我机航向
                        if (nearest_msl_dir is not None) and (my_heading_deg is not None):
                            rel_angle_to_msl = _ang_diff_abs(nearest_msl_dir, my_heading_deg)
                        else:
                            rel_angle_to_msl = 0.0

                # === 组 19 维观测
                obs19 = self.recorder.pack_single_obs19(
                    self.client, rid, obs_raw, BOUNDARY_RECT, jam_signed,
                    dlist, nb_speed, nb_dir,
                    nearest_msl_dist, nearest_msl_dir, rel_angle_to_msl,
                    vel_meas=vel_meas
                )
                obs_concat.extend(obs19)
                # === 为 shaping 收集：最近目标ID、角度误差、速度带、vz 抖动 ===
                # 最近可攻击（可见）蓝机ID
                nearest_tid, nearest_d = None, None
                if tracks and (my_lon is not None) and (my_lat is not None):
                    for t in tracks:
                        tid_c = t.get("target_id")
                        if tid_c is None or int(tid_c) < 10000:  # 跳过导弹
                            continue
                        lonB, latB = t.get("lon"), t.get("lat")
                        if lonB is None or latB is None:
                            continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                     Position(x=lonB, y=latB, z=0))
                        except Exception:
                            dd = None
                        if dd is not None and (nearest_d is None or dd < nearest_d):
                            nearest_d, nearest_tid = float(dd), int(tid_c)

                # 我机当前航向（由测得 vx,vy）
                my_heading_deg = None
                if vel_meas is not None:
                    _, my_heading_deg = _direct_rate_from_vx_vy(getattr(vel_meas, "vx", 0.0),
                                                                getattr(vel_meas, "vy", 0.0))
                # 最近蓝机的 LOS 方位（若有）
                bearing_to_nb = None
                if nearest_tid is not None and (my_lon is not None) and (my_lat is not None):
                    nb_lon, nb_lat = self._get_target_pos(nearest_tid, vis, all_pos)
                    if (nb_lon is not None) and (nb_lat is not None):
                        bearing_to_nb = _bearing_deg_from_A_to_B(my_lon, my_lat, nb_lon, nb_lat)

                # 角度误差（度）
                ang_err_deg = None
                if (my_heading_deg is not None) and (bearing_to_nb is not None):
                    ang_err_deg = _ang_diff_abs(my_heading_deg, bearing_to_nb)

                # 写入临时容器供本步整合（以 rid 为键）
                if not hasattr(self, "_shaping_cache"):
                    self._shaping_cache = {}
                self._shaping_cache[int(rid)] = {
                    "nearest_d": float(nearest_d) if nearest_d is not None else None,
                    "ang_err_deg": float(ang_err_deg) if ang_err_deg is not None else None,
                    "nearest_tid": int(nearest_tid) if nearest_tid is not None else None,
                }

                # 速度标量与速度带统计
                v_scalar = math.sqrt(float(getattr(vel_meas, "vx", 0.0) or 0.0)**2 + float(getattr(vel_meas, "vy", 0.0) or 0.0)**2)
                in_band = (self.SPEED_MIN <= v_scalar <= self.SPEED_MAX)
                band_key = "_speed_band_hits"
                setattr(self, band_key, getattr(self, band_key, 0) + (1 if in_band else 0))
                setattr(self, "_speed_total", getattr(self, "_speed_total", 0) + 1)

                # 垂直速度抖动
                vz_now = float(getattr(vel_meas, "vz", 0.0) or 0.0)
                last_vz = self._last_vz.get(rid, vz_now)
                setattr(self, "_vz_abs_diff_sum", getattr(self, "_vz_abs_diff_sum", 0.0) + abs(vz_now - last_vz))
                self._last_vz[rid] = vz_now

                # 专注/换目标统计（以“最近蓝机”作为轻量 proxy）
                last_pair = self._focus_target.get(rid, (None, 0.0))
                last_tid, consec = last_pair
                if nearest_tid is not None:
                    if last_tid == nearest_tid:
                        consec += 1.0 * (1)  # 每秒 +1
                    else:
                        self._switch_count_step += 1
                        consec = 1.0
                    self._focus_target[rid] = (nearest_tid, consec)
                # 归一化到“每机平均专注秒”
                # 注意：每秒最后统一平均，这里先累加


                # === 行为 4 维
                act4 = self.recorder.pack_single_act4(rid, sim_sec, vel_meas=vel_meas, pos_meas=my_p)
                act_concat.extend(act4)

                # === 记录，用 obs19 的下标（不要再用 obs16 了）
                self.recorder.record_tick(sim_sec, rid, {
                    "boundary_dists": {"left": obs19[0], "right": obs19[1], "down": obs19[2], "up": obs19[3]},
                    "vel": {"vx": obs19[4], "vy": obs19[5], "vz": obs19[6]},
                    "blue_dists": dlist,
                    "nearest_blue_speed": nb_speed,
                    "nearest_blue_dir": nb_dir,
                    "jam_signed_dist": obs19[13],
                    "ammo": obs19[14],
                    "speed_scalar": obs19[15],
                    "nearest_msl_dist": obs19[16],
                    "nearest_msl_dir": obs19[17],
                    "rel_angle_to_msl": obs19[18],
                })

            self.recorder.latest_obs_vec = obs_concat
            self.recorder.latest_act_vec = act_concat
            self.recorder.add_vector_row(sim_sec, obs_concat, act_concat)
            # —— 本秒统计：专注平均、速度带比例、垂直抖动均值（供奖励使用）——
            focus_total = 0.0; focus_cnt = 0
            for rid in sorted(self.red_ids):
                pair = self._focus_target.get(rid, (None, 0.0))
                focus_total += float(pair[1] if pair else 0.0)
                focus_cnt += 1
                # 本秒统计完，把“秒计数”只保留最后目标但不累计，方便下秒递增
                if pair and pair[0] is not None:
                    self._focus_target[rid] = (pair[0], 0.0)
            self._focused_secs_step = (focus_total / max(1, focus_cnt))

            self._speed_band_ratio = (getattr(self, "_speed_band_hits", 0) / max(1, getattr(self, "_speed_total", 1)))
            self._speed_off_ratio  = 1.0 - self._speed_band_ratio
            self._vz_abs_diff_mean = (getattr(self, "_vz_abs_diff_sum", 0.0) / max(1, focus_cnt))

            # 清零计数（供下一秒重新累计）
            setattr(self, "_speed_band_hits", 0)
            setattr(self, "_speed_total", 0)
            setattr(self, "_vz_abs_diff_sum", 0.0)

            self._last_logged_sec = sim_sec

        # ——【每秒一次】用 PPO 产生 4×act4，并补“上一秒”的奖励 —— #
        try:
            if sim_sec != getattr(self, "_last_policy_sec", -1) and self.recorder.latest_obs_vec is not None:
                # 先给上一条样本补奖励
                if self._last_step_sec is not None and self.traj:
                    blue_k, red_k = len(self.destroyed_blue), len(self.destroyed_red)
                    d_blue = blue_k - self._last_counts["blue"]
                    d_red  = red_k  - self._last_counts["red"]
                    rew = d_blue * self.REW_BLUE_KILL + d_red * self.REW_RED_LOSS
                    # ===== 新增：区域探索奖励 =====
                    for rid in sorted(self.red_ids):
                        rp = all_pos.get(rid)
                        if rp and getattr(rp, "x", None) is not None and getattr(rp, "y", None) is not None:
                            rew += self._exploration_reward(float(rp.x), float(rp.y))

                    # 禁飞区惩罚（原有）
                    nf_penalty = 0.0
                    for rid in sorted(self.red_ids):
                        rp = all_pos.get(rid)
                        if rp and getattr(rp, "x", None) is not None and getattr(rp, "y", None) is not None:
                            if _signed_dist_to_jam_boundary_m(self.client, float(rp.x), float(rp.y)) < 0.0:
                                nf_penalty += self.REW_NOFLY_PER_SEC
                    rew += nf_penalty

                    # ===== 新增：生存时间成本（鼓励更快结束，小量负值）=====
                    alive_reds = len([rid for rid in self.red_ids if rid not in self.destroyed_red])
                    rew += self.lambda_ttl * float(alive_reds)

                    # ===== 新增：Potential-based Shaping（距离+对准）
                    # 计算当前势函数 Phi(s)
                    phi_now = 0.0;
                    cnt = 0
                    for rid in sorted(self.red_ids):
                        meta = getattr(self, "_shaping_cache", {}).get(int(rid), {})
                        d = meta.get("nearest_d", None)
                        ang_err = meta.get("ang_err_deg", None)
                        if d is None or ang_err is None:
                            continue
                        # 距离归一/截断（越近越好）
                        d_cap = BLUE_DIST_CAP if BLUE_DIST_CAP and BLUE_DIST_CAP > 1.0 else 70000.0
                        d_norm = _clamp(d / d_cap, 0.0, 1.0)
                        term_d = self.alpha_phi * (1.0 / (1.0 + d_norm))
                        # 指向对准（cos最大为1）
                        term_ang = self.beta_phi * _cosd(ang_err)
                        phi_now += (term_d + term_ang);
                        cnt += 1
                    if cnt > 0:
                        phi_now /= float(cnt)
                        rew += (self.gamma * phi_now - (self._phi_prev or 0.0))
                        self._phi_prev = phi_now
                    else:
                        # 无法观测到蓝机时，不改变 _phi_prev 以避免抖动
                        pass

                    # ===== 新增：锁定/专注 & 换目标惩罚 =====
                    rew += (self.lambda_lock_pos * float(self._focused_secs_step))
                    rew += (self.lambda_switch * float(self._switch_count_step))
                    self._switch_count_step = 0  # 用一次清一次

                    # ===== 新增：速度带与垂直速度抖动 =====
                    rew += (self.lambda_speed_ok * float(getattr(self, "_speed_band_ratio", 0.0)))
                    rew += (self.lambda_speed_ng * float(getattr(self, "_speed_off_ratio", 1.0)))
                    rew += (self.lambda_vz_jitter * float(getattr(self, "_vz_abs_diff_mean", 0.0)))

                    # ===== 新增：动作平滑（相邻秒动作向量 L2 均值）=====
                    if self._last_act_full is not None and len(self._last_act_full) == len(self.traj[-1]["act"]):

                        a_prev = _np.asarray(self._last_act_full, dtype=_np.float32)
                        a_curr = _np.asarray(self.traj[-1]["act"], dtype=_np.float32)  # 注意：这是上条样本的动作
                        l2 = float(_np.linalg.norm(a_curr - a_prev)) / max(1.0, float(len(a_curr)))
                        rew += (self.lambda_act_smooth * l2)

                    # ===== 新增：反导规避几何（上一秒统计）=====
                    rew += (self.lambda_evade_good * float(self._evade_good_last))
                    rew += (self.lambda_evade_bad * float(self._evade_bad_last))
                    self._evade_good_last = 0.0
                    self._evade_bad_last = 0.0

                    # 落到上一条样本
                    self.traj[-1]["rew"] = float(rew)
                    self._last_counts = {"blue": blue_k, "red": red_k}

                # 本秒动作 —— 关键改动：传入/更新 GRU 隐状态
                obs76 = _np.asarray(self.recorder.latest_obs_vec, dtype=_np.float32)
                t_norm = (sim_sec - 0.0) / max(1.0, 300.0)
                tfeat_full = _np.array([t_norm, math.sin(2*math.pi*t_norm), math.cos(2*math.pi*t_norm)], dtype=_np.float32)
                tfeat = tfeat_full[: self.ppo.tdim]
                a_full, V, logp, h1 = self.ppo.act(obs76, tfeat, h=self._ppo_h, explore=True)
                # 记录本秒动作（供下一秒的平滑惩罚使用）
                self._last_act_full = list(a_full)

                self._ppo_h = h1

                # 下发速度 + 设置“意图位”
                red_list = sorted(self.red_ids)
                for idx, rid in enumerate(red_list):
                    a0, a1, a2, a3 = [float(a_full[idx * 4 + k]) for k in range(4)]
                    vcmd = rflysim.Vel()
                    vcmd.rate   = max(0.0, float(a1))       # direct(速度) -> 接口 rate
                    vcmd.direct = float(a0) % 360.0         # rate(角度) -> 接口 direct
                    vcmd.vz     = float(a2)
                    now_sim = self.client.get_sim_time()
                    if self._evasive_until.get(rid, 0.0) > now_sim:
                        self._policy_attack_intent[rid] = 0
                        continue
                    # 后撤持续窗口：屏蔽 PPO，并“按需刷新”上一条反向指令（避免过期）
                    if self._east_maneuver_until.get(rid, 0.0) > now_sim:
                        self._policy_attack_intent[rid] = 0  # 不让打
                        last = self.recorder.last_vel_cmd.get(int(rid))
                        try:
                            # 如果上一条速度命令时间已久，主动重发一次，避免控制器把速度慢慢拉回去
                            if (not last) or (now_sim - float(last.get("t", 0.0))) >= 0.8:
                                vcmd = rflysim.Vel()
                                # 兜底：保持上次航向/升降，速度顶到上限
                                vcmd.rate = min(self.SPEED_CAP,
                                                float(last.get("rate", self.SPEED_CAP)) if last else self.SPEED_CAP)
                                vcmd.direct = float(last.get("direct", 0.0) if last else 0.0) % 360.0
                                vcmd.vz = float(last.get("vz", 0.0) if last else 0.0)
                                uid = self.client.set_vehicle_vel(rid, vcmd)
                                self.recorder.mark_vel_cmd(rid, rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz,
                                                           sim_time=now_sim)
                        except Exception as e:
                            print(f"[EAST-MANEUVER] refresh cmd for {rid} failed: {e}", flush=True)
                        continue  # 关键：跳过 PPO 对该机的下发
                    try:
                        uid = self.client.set_vehicle_vel(rid, vcmd)
                        self.recorder.mark_vel_cmd(rid, rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=now_sim)
                    except Exception as e:
                        print(f"[PPO] set_vehicle_vel({rid}) failed: {e}", flush=True)
                    self._policy_attack_intent[rid] = 1 if int(a3) == 1 else 0

                # 记录样本占位（奖励下一秒补）
                self.traj.append({
                    "obs":   obs76.copy(),
                    "tfeat": tfeat.copy(),
                    "act":   _np.asarray(a_full, dtype=_np.float32).copy(),
                    "val":   float(V),
                    "logp":  float(logp),
                    "rew":   0.0,
                    "done":  0.0,
                })
                self._last_step_sec = sim_sec
                self._last_policy_sec = sim_sec
        except Exception as e:
            print("[PPO] policy step failed:", e, flush=True)

        if self._ended: return

        # —— 执行攻击逻辑（原样两阶段），仅使用“意图位”过滤 —— #
        try:
            raw_visible2 = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True); return
        vis2 = normalize_visible(raw_visible2)
        try: all_pos2 = self.client.get_vehicle_pos()
        except Exception: all_pos2 = {}
        self._update_destroyed_from_situ()
        now = self.client.get_sim_time()

        try: vel_all2 = self.client.get_vehicle_vel()
        except Exception: vel_all2 = {}

        # ===== 导弹调试（保留）=====
        if not hasattr(self, "_last_raw_msl_print_sec"):
            self._last_raw_msl_print_sec = -1
        if sim_sec != self._last_raw_msl_print_sec:
            self._last_raw_msl_print_sec = sim_sec
            try:
                missile_ids = set()
                for vid in (all_pos2 or {}):
                    try:
                        if int(vid) < 10000: missile_ids.add(int(vid))
                    except Exception: pass
                for tracks in (vis2 or {}).values():
                    for t in (tracks or []):
                        tid = t.get("target_id")
                        if tid is not None and int(tid) < 10000: missile_ids.add(int(tid))
                if missile_ids:
                    print("[RAW MISSILES @{}] ids={}".format(sim_sec, sorted(missile_ids)), flush=True)
                    for mid in sorted(missile_ids):
                        mpos = all_pos2.get(mid)
                        mm = vel_all2.get(mid) if vel_all2 else None
                        raw_track = None
                        for det_id, tracks in (raw_visible2 or {}).items():
                            if isinstance(tracks, list):
                                for item in tracks:
                                    td = _as_dict(item)
                                    if int(td.get("target_id", -1)) == int(mid): raw_track = td; break
                            elif isinstance(tracks, dict):
                                for _, item in tracks.items():
                                    td = _as_dict(item)
                                    if int(td.get("target_id", -1)) == int(mid): raw_track = td; break
                            if raw_track: break
                        print("[RAW] mid={} track={} pos=({:.6f},{:.6f}) vel={{vx:{}, vy:{}, vz:{}, direct:{}, rate:{}}}".format(
                            mid,
                            raw_track,
                            getattr(mpos, "x", float("nan")) if mpos else float("nan"),
                            getattr(mpos, "y", float("nan")) if mpos else float("nan"),
                            getattr(mm, "vx", None) if mm else None,
                            getattr(mm, "vy", None) if mm else None,
                            getattr(mm, "vz", None) if mm else None,
                            getattr(mm, "direct", None) if mm else None,
                            getattr(mm, "rate", None) if mm else None
                        ), flush=True)
            except Exception as e:
                print("[RAW MISSILES] dump failed:", e, flush=True)

        expired = [tid for tid, meta in self.target_locks.items() if now >= meta["until"]]
        for tid in expired: self.target_locks.pop(tid, None)

        # === 躲避判定（保留）===
        if not hasattr(self, "_missile_prev_xy_t"):
            self._missile_prev_xy_t = {}
        missiles = []
        now_sim = self.client.get_sim_time()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if not _is_missile_track(tid): continue
                mid = int(tid)
                mpos = all_pos2.get(mid)
                if mpos and getattr(mpos, "x", None) is not None and getattr(mpos, "y", None) is not None:
                    mlon, mlat = float(mpos.x), float(mpos.y)
                else:
                    mlon = t.get("lon"); mlat = t.get("lat")
                if mlon is None or mlat is None: continue

                mdir_calc, mspeed_calc = None, None
                prev = self._missile_prev_xy_t.get(mid)
                if prev and prev[0] is not None and prev[1] is not None:
                    prev_lon, prev_lat, prev_t = prev
                    dt = max(1e-3, float(now_sim - float(prev_t)))
                    mdir_calc = _bearing_deg_from_A_to_B(prev_lon, prev_lat, mlon, mlat)
                    d_m = _geo_dist_haversine_m(prev_lon, prev_lat, mlon, mlat)
                    if d_m is not None: mspeed_calc = d_m / dt
                self._missile_prev_xy_t[mid] = (mlon, mlat, now_sim)

                missiles.append({
                    "mid": mid, "lon": mlon, "lat": mlat,
                    "mdir": mdir_calc, "mspeed": mspeed_calc,
                    "mspeed_radar": t.get("speed", None)
                })

        nearest_red_of_missile = {}
        if missiles:
            for m in missiles:
                mid, mlon, mlat = m["mid"], m["lon"], m["lat"]
                rid_near, d_near = None, None
                for rid in self.red_ids:
                    rp = all_pos2.get(rid)
                    if not rp or getattr(rp, "x", None) is None or getattr(rp, "y", None) is None: continue
                    rlon, rlat = float(rp.x), float(rp.y)
                    d = _geo_dist_haversine_m(mlon, mlat, rlon, rlat)
                    if d is None: continue
                    if d_near is None or d < d_near: rid_near, d_near = rid, float(d)
                if rid_near is not None: nearest_red_of_missile[mid] = (rid_near, d_near)

        if missiles:
            now_wall = time.time()
            for rid in sorted(self.red_ids):
                my_p = all_pos2.get(rid)
                if not my_p or getattr(my_p, "x", None) is None or getattr(my_p, "y", None) is None: continue
                my_lon, my_lat = float(my_p.x), float(my_p.y)
                best_mid, best_dist, best_meta = None, None, None
                for m in missiles:
                    d = _geo_dist_haversine_m(m["lon"], m["lat"], my_lon, my_lat)
                    if d is None: continue
                    if best_dist is None or d < best_dist:
                        best_mid, best_dist, best_meta = m["mid"], float(d), m
                if best_mid is None: continue
                mdir = best_meta.get("mdir")
                los_m2red = _bearing_deg_from_A_to_B(best_meta["lon"], best_meta["lat"], my_lon, my_lat)
                ang_diff = _ang_diff_abs(mdir, los_m2red) if (mdir is not None and los_m2red is not None) else None

                approach_ok = False; reason = "unknown"
                if ang_diff is not None:
                    if ang_diff <= self.MISSILE_BEARING_THRESH_DEG:
                        approach_ok = True; reason = f"angle_ok({ang_diff:.1f}<={self.MISSILE_BEARING_THRESH_DEG})"
                    else:
                        reason = f"angle_large({ang_diff:.1f})"
                else:
                    key = (rid, best_mid, "dist")
                    last_d = self._missile_last_dist.get(key, None)
                    if last_d is not None and best_dist < last_d:
                        approach_ok = True; reason = "dist_decreasing"
                    else:
                        reason = "no_dir_and_no_decrease"
                    self._missile_last_dist[key] = best_dist

                in_threat = (best_dist is not None and best_dist <= self.MISSILE_THREAT_DIST_M)
                last_ev = self._last_evasive_time.get(rid, -1e9)
                cooldown_ok = (now_wall - last_ev) >= self.EVASIVE_COOLDOWN_SEC
                in_evasive_window = (self._evasive_until.get(rid, 0.0) > now_sim)

                rid_near, d_near = nearest_red_of_missile.get(best_mid, (None, None))

                if self.DEBUG_EVADE:
                    dist_str = f"{best_dist:.0f}" if best_dist is not None else "-"
                    mdir_str = f"{mdir:.1f}" if mdir is not None else "None"
                    los_str = f"{los_m2red:.1f}" if los_m2red is not None else "None"
                    ang_str = f"{ang_diff:.1f}" if ang_diff is not None else "None"
                    rid_near_str = f"{rid_near}" if rid_near is not None else "None"
                    d_near_str = f"{d_near:.0f}" if d_near is not None else "None"
                    ms_calc = best_meta.get("mspeed")
                    mspeed_calc_str = f"{ms_calc:.1f}" if ms_calc is not None else "None"
                    ms_radar = best_meta.get("mspeed_radar")
                    mspeed_radar_str = f"{ms_radar:.1f}" if ms_radar is not None else "None"
                    lp = self._last_debug_print.get(rid, 0.0)
                    flip = (self._last_approach_flag.get(rid) != approach_ok)
                    if (now_wall - lp) >= self.MISSILE_DEBUG_PRINT_INTERVAL or flip:
                        self._last_debug_print[rid] = now_wall
                        self._last_approach_flag[rid] = approach_ok
                        print(
                            f"[MISSILE] rid={rid} mid={best_mid} "
                            f"dist={dist_str}m mspeed_calc={mspeed_calc_str}m/s mspeed_radar={mspeed_radar_str}m/s "
                            f"nearest_red_of_missile={rid_near_str}@{d_near_str}m "
                            f"mdir_calc={mdir_str}° los_m2red={los_str}° ang_diff={ang_str} "
                            f"approach={approach_ok} threat={in_threat} "
                            f"cooldown_ok={cooldown_ok} evading_now={in_evasive_window} "
                            f"reason={reason}",
                            flush=True
                        )
                # === 统计规避几何“好/坏”一秒（累加到 _evade_*_last，下一秒结算）
                if in_threat:
                    if (ang_diff is not None) and (ang_diff <= self.MISSILE_BEARING_THRESH_DEG):
                        # 坏几何（迎向）
                        self._evade_bad_last  = float(self._evade_bad_last  + 1.0 / max(1, len(self.red_ids)))
                    else:
                        # 好几何（不是迎向）
                        self._evade_good_last = float(self._evade_good_last + 1.0 / max(1, len(self.red_ids)))

                if in_threat and approach_ok and cooldown_ok:
                    use_mdir = mdir if mdir is not None else (los_m2red or 0.0)
                    cand1 = _ang_norm(use_mdir + self.EVASIVE_TURN_DEG)
                    cand2 = _ang_norm(use_mdir - self.EVASIVE_TURN_DEG)
                    los_red2m = _bearing_deg_from_A_to_B(my_lon, my_lat, best_meta["lon"], best_meta["lat"])
                    away_dir = _ang_norm(los_red2m + 180.0) if los_red2m is not None else None
                    if away_dir is not None:
                        score1 = _ang_diff_abs(cand1, away_dir)
                        score2 = _ang_diff_abs(cand2, away_dir)
                        evade_heading = cand1 if score1 <= score2 else cand2
                    else:
                        evade_heading = cand1
                    v_me = vel_all2.get(rid) if vel_all2 else None
                    v_direct_now, _ = _direct_rate_from_vx_vy(
                        getattr(v_me, "vx", 0.0) if v_me else 0.0,
                        getattr(v_me, "vy", 0.0) if v_me else 0.0
                    )
                    v_direct_cmd = min(self.SPEED_CAP, max(float(v_direct_now or 0.0), self.EVASIVE_SPEED_MIN))

                    vz_keep = getattr(v_me, "vz", 0.0) if v_me else 0.0
                    try:
                        vcmd = rflysim.Vel()
                        vcmd.rate = max(0.0, min(float(a1), self.SPEED_CAP))  # 速度 ∈ [0, 200]

                        vcmd.direct = float(evade_heading)
                        vcmd.vz = float(vz_keep)
                        uid = self.client.set_vehicle_vel(rid, vcmd)
                        self.recorder.mark_vel_cmd(rid, rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=now_sim)
                        print(
                            f"[EVADE] rid={rid} mid={best_mid} "
                            f"CMD: rate={vcmd.rate:.1f} direct={vcmd.direct:.1f} vz={vcmd.vz:.1f} "
                            f"(dist={best_dist:.0f}m, mdir={mdir if mdir is not None else -1:.1f}°, reason={reason})",
                            flush=True
                        )
                    except Exception as e:
                        print(f"[EVADE] set_vehicle_vel({rid}) failed: {e}", flush=True)
                    self._evasive_until[rid] = now_sim + self.EVASIVE_DURATION_SEC
                    self._last_evasive_time[rid] = now_wall

        # === 收集可攻击蓝机 ===
        visible_blue_targets = set()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if tid is None or tid < 10000: continue
                if tid in self.destroyed_targets or tid in self.target_locks: continue
                visible_blue_targets.add(tid)

        if visible_blue_targets != self._last_visible:
            print("未锁定的目标:", visible_blue_targets, flush=True)
            self._last_visible = set(visible_blue_targets)
        if self.target_locks != self._last_locked:
            print("已经锁定的目标:", self.target_locks, flush=True)
            self._last_locked = dict(self.target_locks)
        if self.destroyed_targets != self._last_destroyed:
            print("已经被摧毁的目标:", self.destroyed_targets, flush=True)
            self._last_destroyed = set(self.destroyed_targets)
        if not visible_blue_targets: return

        used_reds_this_round = set()
        assignments = []
        for tid in visible_blue_targets:
            t_lon, t_lat = self._get_target_pos(tid, vis2, all_pos2)
            if t_lon is None or t_lat is None: continue
            best_red, best_d = None, 1e18
            for rid in self.red_ids:
                if rid in used_reds_this_round: continue
                if rid in self.destroyed_targets: continue
                if self._evasive_until.get(rid, 0.0) > now: continue
                if self.ammo.get(rid, 0) <= 0: continue
                if (now - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC: continue
                r_pos = all_pos2.get(rid)
                if r_pos is None: continue
                d = self._distance_m(r_pos.x, r_pos.y, t_lon, t_lat)
                if d < best_d:
                    best_d, best_red = d, rid
            if best_red is not None:
                assignments.append((best_red, tid))
                used_reds_this_round.add(best_red)

        if not assignments: return

        # ===== 策略优先（意图位=1 的红机先打一次）=====
        try:
            now_sim2 = self.client.get_sim_time()
            red_list = sorted(self.red_ids)
            candidate_tids = list(visible_blue_targets)
            for rid in red_list:
                if not self._policy_attack_intent.get(rid, 0):
                    continue
                if self._evasive_until.get(rid, 0.0) > now_sim2: continue
                if (now_sim2 - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC: continue
                if self.ammo.get(rid, 0) <= 0: continue
                mypos = all_pos2.get(rid)
                if mypos is None: continue
                best_tid, best_d = None, 1e18
                for tid in candidate_tids:
                    if tid in self.destroyed_targets or tid in self.target_locks: continue
                    t_lon, t_lat = self._get_target_pos(tid, vis2, all_pos2)
                    if t_lon is None or t_lat is None: continue
                    d = self._distance_m(mypos.x, mypos.y, t_lon, t_lat)
                    if d < best_d: best_d, best_tid = d, tid
                if best_tid is None: continue
                try:
                    uid = self._fire_with_log(rid, best_tid)
                    print(f"[PPO-FIRE] {rid} -> {best_tid}, uid={uid}", flush=True)
                    ok = self._is_fire_success(uid, max_tries=5, wait_s=0.1)
                    if not ok:
                        print(f"[Red] fire NOT confirmed for {rid}->{best_tid}, keep target available.", flush=True)
                        time.sleep(0.1); continue
                    self.recorder.mark_attack_success(rid, self.client.get_sim_time())
                    self.ammo[rid] -= 1
                    self.last_fire_time[rid] = self.client.get_sim_time()
                    self.assigned_target[rid] = best_tid
                    self.target_locks[best_tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}
                    if best_tid in candidate_tids: candidate_tids.remove(best_tid)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[PPO-FIRE] set_target({rid},{best_tid}) failed: {e}", flush=True)
        except Exception as e:
            print("[PPO-FIRE] block failed:", e, flush=True)

        # ===== 基线回落 =====
        for rid, tid in assignments:
            if tid in self.destroyed_targets: continue
            now_sim = self.client.get_sim_time()
            if self._evasive_until.get(rid, 0.0) > now_sim: continue
            if (now_sim - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC: continue
            if self.ammo.get(rid, 0) <= 0: continue
            try:
                uid = self._fire_with_log(rid, tid)
                print(f"[Red] {rid} -> fire at {tid}, uid={uid}", flush=True)
                ok = self._is_fire_success(uid, max_tries=5, wait_s=0.1)
                if not ok:
                    print(f"[Red] fire NOT confirmed for {rid}->{tid}, keep target available.", flush=True)
                    time.sleep(0.1); continue
                self.recorder.mark_attack_success(rid, self.client.get_sim_time())
                self.ammo[rid] -= 1
                self.last_fire_time[rid] = self.client.get_sim_time()
                self.assigned_target[rid] = tid
                self.target_locks[tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}
                time.sleep(0.1)
            except Exception as e:
                print(f"[Red] set_target({rid},{tid}) failed: {e}", flush=True); continue


        # === 全蓝已被锁定 → 触发“最东红机后撤机动” ===
        try:
            self._maybe_maneuver_eastmost(visible_blue_targets, all_pos2, vel_all2, now)
        except Exception as e:
            print("[EAST-MANEUVER] failed:", e, flush=True)


    def run_loop(self, stop_when_paused=False, max_wall_time_sec=None):
        start_t = time.time()
        try:
            while True:
                if self._ended: break
                if stop_when_paused:
                    try:
                        if self.client.is_pause(): break
                    except Exception: pass
                if max_wall_time_sec is not None and (time.time() - start_t) > max_wall_time_sec:
                    print("[Red] Episode timeout reached, stopping...", flush=True)
                    self._end_simulation_once("Timeout"); break
                self.step()
                time.sleep(SCAN_INTERVAL_SEC)
        finally:
            try: self.recorder.dump_csv(self.out_csv_path)
            except Exception as e: print("[RL] dump_csv on finally failed:", e, flush=True)

# ==================== 多轮批量 Runner ====================
def safe_reset(client):
    for fn_name in ["restart", "reset", "reset_scene", "reset_scenario"]:
        fn = getattr(client, fn_name, None)
        if callable(fn):
            try:
                fn(); print(f"[Runner] client.{fn_name}() called.", flush=True); return True
            except Exception as e:
                msg = str(e)
                if "容器正在运行" in msg or "already running" in msg.lower():
                    print(f"[Runner] client.{fn_name}(): container already running, continue.", flush=True)
                    return True
                print(f"[Runner] client.{fn_name}() failed: {e}", flush=True); continue
    try: client.stop()
    except Exception: pass
    time.sleep(1.0)
    try:
        client.start(); print("[Runner] fallback stop()->start() used.", flush=True); return True
    except Exception as e:
        msg = str(e)
        if "容器正在运行" in msg or "already running" in msg.lower():
            print("[Runner] fallback start(): container already running, continue.", flush=True); return True
        print("[Runner] fallback start failed:", e, flush=True); return False

def run_one_episode(client, plan_id, out_csv_path, max_wall_time_sec=360, min_wall_time_sec=10,
                    episode_idx: int = 1, loss_logger: Optional[LossLogger] = None):
    try:
        client.stop(); print("[Runner] pre-episode stop()", flush=True)
    except Exception: pass
    time.sleep(0.5)
    ok = safe_reset(client)
    if not ok:
        print("[Runner] safe_reset failed; give up this episode.", flush=True)
        return False, None

    def _blue_wrapper():
        try:
            blue = CombatSystem(client)
            blue.run_combat_loop()
        except Exception as e:
            print("[Runner] Blue loop exception:", e, flush=True)

    blue_thread = threading.Thread(target=_blue_wrapper, daemon=True)
    blue_thread.start()
    print("[Runner] Blue loop started.", flush=True)

    red_ctrl = RedForceController(client, RED_IDS, out_csv_path)
    # 不再 attach BC（动作改为 PPO）；如需沿用列名对齐可留：red_ctrl.attach_bc_policy("./bc_out_seq")

    red_thread = threading.Thread(
        target=lambda: red_ctrl.run_loop(max_wall_time_sec=max_wall_time_sec),
        daemon=True
    )
    red_thread.start()
    print("[Runner] Red loop started.", flush=True)

    t0 = time.time()
    success = False
    score_obj = None

    try:
        while True:
            if not red_thread.is_alive():
                success = True; break
            if (time.time() - t0) > max_wall_time_sec:
                print("[Runner] Episode timeout reached, stopping...", flush=True)
                try: client.stop()
                except Exception: pass
                break
            time.sleep(0.5)
    finally:
        try: client.stop()
        except Exception: pass

        try:
            if hasattr(red_ctrl, "_fetch_score_with_retry"):
                score_obj = red_ctrl._fetch_score_with_retry(tries=10, wait=0.2, where="runner-final")
        except Exception: score_obj = None

        wall = time.time() - t0
        if wall < min_wall_time_sec:
            success = False
            print(f"[Runner] Episode too short ({wall:.1f}s) -> mark as failed.", flush=True)

        # === PPO 在线更新 + 保存 ===
        try:
            if hasattr(red_ctrl, "traj") and red_ctrl.traj:
                blue_k, red_k = len(red_ctrl.destroyed_blue), len(red_ctrl.destroyed_red)
                d_blue = blue_k - red_ctrl._last_counts["blue"]
                d_red  = red_k  - red_ctrl._last_counts["red"]
                final_rew = d_blue * red_ctrl.REW_BLUE_KILL + d_red * red_ctrl.REW_RED_LOSS
                if red_ctrl.traj:
                    red_ctrl.traj[-1]["rew"] += float(final_rew)
                    red_ctrl.traj[-1]["done"] = 1.0
                out = red_ctrl.ppo.update(red_ctrl.traj, epochs=4, minibatch=2)
                if loss_logger is not None:
                    loss_logger.append_update_stats(episode_idx, out)
                # === 统计并记录本回合回报 / 长度 / 击毁数 ===
                try:
                    if loss_logger is not None and hasattr(red_ctrl, "traj") and red_ctrl.traj:
                        ep_return = sum(float(x.get("rew", 0.0)) for x in red_ctrl.traj)
                        ep_len = len(red_ctrl.traj)
                        blue_k = len(red_ctrl.destroyed_blue)
                        red_k = len(red_ctrl.destroyed_red)
                        loss_logger.append_episode_summary(episode_idx, ep_return, ep_len, blue_k, red_k)

                        # 控制台打印（再给个最近10回合均值，如果文件里已有 >=10 条）
                        import csv, os
                        ma10 = None
                        if os.path.exists(loss_logger.ep_csv_path):
                            rs = []
                            with open(loss_logger.ep_csv_path, "r", encoding="utf-8") as f:
                                for i, row in enumerate(csv.DictReader(f), 1):
                                    rs.append(float(row["return"]))
                            if len(rs) >= 10:
                                ma10 = sum(rs[-10:]) / 10.0

                        msg = f"[EP{episode_idx}] return={ep_return:.3f} | len={ep_len} | blue_kills={blue_k} | red_losses={red_k}"
                        if ma10 is not None:
                            msg += f" | return_ma10={ma10:.3f}"
                        print(msg, flush=True)
                except Exception as e:
                    print("[Logger] episode summary failed:", e, flush=True)

                red_ctrl.ppo.save()  # -> ./bc_out_seq/seq_policy.pt.online
                # 逐 epoch 展开打印（update 内部也会打印一次，这里让 manifest 更完整）
                try:
                    stats = out.get("epoch_stats", []) or []
                    for ei, s in enumerate(stats, 1):
                        print(
                            "[PPO][epoch {}/{}] total={:.4f} | pg={:.4f} v={:.4f} ent={:.4f} | "
                            "kl_ref={:.4f} kl≈={:.4f} clipfrac={:.2%} ev={:.3f}".format(
                                ei, len(stats),
                                s["loss_total"], s["pg_loss"], s["v_loss"], s["entropy"],
                                s["kl_ref"], s["approx_kl"], s["clipfrac"], s["ev"]
                            ),
                            flush=True
                        )
                    print(
                        "[PPO] KL_mean={:.4f} bc_kl_coef={:.5f} updates={} lr={:.2e} ent_coef={:.2e}".format(
                            float(out.get("kl_mean", 0.0)), float(out.get("bc_kl_coef", 0.0)),
                            int(out.get("updates", 0)), float(out.get("lr", 0.0)), float(out.get("ent_coef", 0.0))
                        ),
                        flush=True
                    )
                    if out.get("plot_path"):
                        print(f"[PPO] Saved loss curves to: {out['plot_path']}", flush=True)
                except Exception:
                    print("[PPO] update:", out, flush=True)

                red_ctrl.ppo.save()  # -> ./bc_out_seq/seq_policy.pt.online

        except Exception as e:
            print("[PPO] online update/save failed:", e, flush=True)

    return success, score_obj

def main():
    EPISODES = 1000
    MAX_WALL_TIME_PER_EP = 600
    MIN_WALL_TIME_PER_EP = 10
    HOST, PORT_CMD, PORT_DATA = "172.23.53.35", 16001, 18001
    PLAN_ID = 106

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.abspath(os.path.join("runs", run_tag))
    os.makedirs(out_root, exist_ok=True)
    print("[Runner] Output root:", out_root, flush=True)
    # —— 新增：全局损失记录器 —— #
    loss_logger = LossLogger(out_root)

    config = {"id": PLAN_ID, "config": RflysimEnvConfig(HOST, PORT_CMD, PORT_DATA)}
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    manifest = {"plan_id": PLAN_ID, "episodes": []}

    for ep in range(1, EPISODES + 1):
        print(f"\n[Runner] ===== Episode {ep}/{EPISODES} =====", flush=True)
        out_csv = os.path.join(out_root, f"ep_{ep:04d}.csv")
        success, score = run_one_episode(
            client, PLAN_ID, out_csv,
            MAX_WALL_TIME_PER_EP, MIN_WALL_TIME_PER_EP,
            episode_idx=ep,  # 新增：回合编号传下去，便于落盘
            loss_logger=loss_logger  # 新增：把 logger 传下去
        )
        manifest["episodes"].append({
            "episode": ep,
            "csv": os.path.abspath(out_csv),
            "success": bool(success),
            "score": score
        })
        time.sleep(2.0 + random.uniform(0.0, 2.0))

    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[Runner] Manifest written to:", manifest_path, flush=True)
    print("[Runner] Done. Total episodes:", EPISODES, flush=True)

if __name__ == "__main__":
    main()
