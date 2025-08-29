# -*- coding: utf-8 -*-
# ================= RL 采样定义 =================
# obs16:
#  0-3 : 边界四距 (left, right, down, up)
#  4-6 : vx, vy, vz
#  7-10: 最近4个蓝机距离（不足补 BLUE_DIST_CAP）
#  11  : 最近蓝机速度
#  12  : 最近蓝机方向（deg, 0=北, 顺时针）
#  13  : 到干扰区边界的有符号距离（外正内负）
#  14  : 剩余攻击次数（ammo）
#  15  : 水平速度标量（优先 vel.direct；否则 sqrt(vx^2+vy^2)）
#
# act4:
#  0-2 : (rate, direct, vz) —— 最近1秒内下发的 vel 命令；没有则用实测兜底
#  3   : 攻击标志（上一秒窗口内是否确认过一次“执行打击成功”）

# ================= 依赖与环境 =================
import os
import csv
import json
import time
import re
import threading
from collections import defaultdict

import numpy as np

import rflysim
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem


# ================= 常量与初始化 =================
DESTROYED_FLAG = "DAMAGE_STATE_DESTROYED"
RED_IDS = [10091, 10084, 10085, 10086]

AMMO_MAX = 8
ATTACK_COOLDOWN_SEC = 2.0     # 同一红机连续下发攻击命令的最短间隔
SCAN_INTERVAL_SEC = 0.1       # 侦察扫描周期
LOCK_SEC = 40.0               # 每个目标的锁定时间窗口（秒）

BLUE_SIDE_CODE = 2            # get_situ_info() 中 2=BLUE, 1=RED
DESTROY_CODES = {1}           # damage_state=1 表示摧毁

# —— 初始速度命令（便于飞行）——
vel_0 = rflysim.Vel()
vel_0.vz = 150       # 高度
vel_0.rate = 200     # 速率
vel_0.direct = 90    # 航向（正北=0°）

# ====== 边界与干扰区 ======
BOUNDARY_RECT = {
    "min_x": 101.07326442811694,
    "max_x": 103.08242360888715,
    "min_y": 39.558295557025474,
    "max_y": 40.599429229677526,
}
JAM_CENTER = (101.96786384206956, 40.2325)         # (lon, lat)
JAM_EDGE_PT = (101.84516211958464, 40.2325)        # 圆上一点（同纬度）
JAM_RADIUS_M = None                                 # 运行时计算并缓存

BLUE_DIST_CAP = 70000.0  # m


# ================= 工具函数 =================
def _bearing_deg_from_A_to_B(lonA, latA, lonB, latB):
    """从A看向B的方位角（0°=正北，顺时针0-360）。"""
    import math
    if None in (lonA, latA, lonB, latB):
        return None
    φ1, φ2 = math.radians(latA), math.radians(latB)
    λ1, λ2 = math.radians(lonA), math.radians(lonB)
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng


def _as_dict(obj):
    """把对象/字典统一成 dict。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def _dist_to_boundary_m(client, lon, lat):
    """到矩形边界（左/右/下/上）的距离，单位米。"""
    if lon is None or lat is None:
        return (0.0, 0.0, 0.0, 0.0)
    L = Position(x=BOUNDARY_RECT["min_x"], y=lat, z=0)
    R = Position(x=BOUNDARY_RECT["max_x"], y=lat, z=0)
    D = Position(x=lon, y=BOUNDARY_RECT["min_y"], z=0)
    U = Position(x=lon, y=BOUNDARY_RECT["max_y"], z=0)
    P = Position(x=lon, y=lat, z=0)
    try:
        d_left = client.get_distance_by_lon_lat(L, P)
        d_right = client.get_distance_by_lon_lat(P, R)
        d_down = client.get_distance_by_lon_lat(D, P)
        d_up = client.get_distance_by_lon_lat(P, U)
    except Exception:
        from math import cos, radians, sqrt
        def _approx_m(dx_lon, dy_lat, lat_ref):
            dx = dx_lon * 111320.0 * cos(radians(lat_ref))
            dy = dy_lat * 110540.0
            return sqrt(dx*dx + dy*dy)
        d_left = _approx_m(lon - BOUNDARY_RECT["min_x"], 0, lat)
        d_right = _approx_m(BOUNDARY_RECT["max_x"] - lon, 0, lat)
        d_down = _approx_m(0, lat - BOUNDARY_RECT["min_y"], lat)
        d_up = _approx_m(0, BOUNDARY_RECT["max_y"] - lat, lat)
    return (d_left, d_right, d_down, d_up)


def _signed_dist_to_jam_boundary_m(client, lon, lat):
    """到干扰区边界的有符号距离：外正内负。"""
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
    """从字符串兜底解析 track 字段。"""
    if not isinstance(s, str):
        return None
    m_id = re.search(r'\btarget_id\s*:\s*(\d+)', s) or re.search(r'\bid\s*:\s*(\d+)', s)
    if not m_id:
        return None
    tid = int(m_id.group(1))
    m_xy = (re.search(r'target\s*_?pos.*?\{[^}]*?x\s*:\s*([-\d\.eE]+)\s*[,， ]+\s*y\s*:\s*([-\d\.eE]+)', s) or
            re.search(r'\bx\s*:\s*([-\d\.eE]+)\s*[，, ]+\s*y\s*:\s*([-\d\.eE]+)', s))
    lon = float(m_xy.group(1)) if m_xy else None
    lat = float(m_xy.group(2)) if m_xy else None
    m_spd = re.search(r'target[_\s-]?speed\s*:\s*([-\d\.eE]+)', s, re.I)
    spd = float(m_spd.group(1)) if m_spd else None
    m_dir = re.search(r'(target[_\s-]?direction|direction|dir)\s*:\s*([-\d\.eE]+)', s, re.I)
    dire = float(m_dir.group(2)) if m_dir else None
    if dire is not None:
        dire = dire % 360.0
        if dire < 0:
            dire += 360.0
    return {"target_id": tid, "lon": lon, "lat": lat, "speed": spd, "direction": dire}


def _extract_track_fields_any(t_like):
    """从对象/字典中尽可能提取 {target_id, lon, lat, speed, direction}。"""
    td = _as_dict(t_like)
    tid = td.get('target_id') or td.get('id')
    if tid is None:
        return None
    tid = int(tid)
    pos = td.get('target_pos') or td.get('target pos') or {}
    posd = _as_dict(pos)
    lon = posd.get('x')
    lat = posd.get('y')
    spd = td.get('target_speed') or td.get('speed')
    dire = td.get('target_direction') or td.get('target direction') or td.get('direction') or td.get('dir')
    try:
        if spd is not None:
            spd = float(spd)
    except Exception:
        spd = None
    try:
        if dire is not None:
            dire = float(dire) % 360.0
            if dire < 0:
                dire += 360.0
    except Exception:
        dire = None
    return {"target_id": tid, "lon": lon, "lat": lat, "speed": spd, "direction": dire}


def normalize_visible(visible):
    """
    -> { red_id: [ {target_id, lon, lat, speed(可空), direction(可空)}, ... ] }
    """
    out = {}
    if not isinstance(visible, dict):
        return out
    for detector_id, v in visible.items():
        try:
            detector_id = int(detector_id)
        except Exception:
            pass
        tracks = []
        if v is None:
            pass
        elif isinstance(v, list):
            for t in v:
                if isinstance(t, dict) or hasattr(t, '__dict__'):
                    item = _extract_track_fields_any(t)
                else:
                    item = _parse_track_from_string_rich(t)
                if item:
                    tracks.append(item)
        elif isinstance(v, dict):
            if 'target_id' in v or 'id' in v:
                item = _extract_track_fields_any(v)
                if item:
                    tracks.append(item)
            else:
                for _, t in v.items():
                    item = _extract_track_fields_any(t)
                    if item:
                        tracks.append(item)
        else:
            item = _parse_track_from_string_rich(v)
            if item:
                tracks.append(item)
        out[int(detector_id)] = tracks
    return out


# ================= 记录器 =================
class RLRecorder:
    def __init__(self):
        self.buffer = []                   # 原始 JSONL（可选）
        self.last_action = {}              # {rid: {"type":..., "t":...}}
        self.attack_events = defaultdict(list)  # {rid: [t_success, ...]}
        self.last_vel_cmd = {}             # {rid: {"rate":..., "direct":..., "vz":..., "t":...}}
        self.rows_for_csv = []             # 每秒一行：t + obs64 + act16
        self.latest_obs_vec = None
        self.latest_act_vec = None

    # —— 16维状态打包 —— #
    def pack_single_obs16(self, client, rid, obs, boundary_rect, jam_signed_dist, blue4_dists,
                          nearest_blue_speed, nearest_blue_dir, vel_meas=None):
        left, right, down, up = _dist_to_boundary_m(client, obs["pos"]["lon"], obs["pos"]["lat"])
        vx, vy, vz_meas = obs["vel"]["vx"], obs["vel"]["vy"], obs["vel"]["vz"]
        ammo = obs["ammo"]

        obs16 = [
            left, right, down, up,                                        # 0-3
            vx if vx is not None else 0.0,                                # 4
            vy if vy is not None else 0.0,                                # 5
            vz_meas if vz_meas is not None else 0.0,                      # 6
            *blue4_dists,                                                  # 7-10
            nearest_blue_speed if nearest_blue_speed is not None else 0.0,# 11
            nearest_blue_dir if nearest_blue_dir is not None else 0.0,    # 12
            jam_signed_dist if jam_signed_dist is not None else 0.0,      # 13
            ammo if ammo is not None else 0                               # 14
        ]
        # 第16维：水平速度标量
        if len(obs16) == 15:
            speed_scalar = None
            if vel_meas is not None:
                speed_scalar = getattr(vel_meas, "direct", None)
                if speed_scalar is None:
                    try:
                        from math import sqrt
                        speed_scalar = sqrt(float(vx or 0.0) ** 2 + float(vy or 0.0) ** 2)
                    except Exception:
                        speed_scalar = 0.0
            obs16.append(float(speed_scalar) if speed_scalar is not None else 0.0)  # 15
        return obs16

    # —— 4维动作打包 —— #
    def pack_single_act4(self, rid, sim_sec, vel_meas, pos_meas):
        vcmd = self.last_vel_cmd.get(int(rid))
        if vcmd and (sim_sec - int(vcmd["t"])) <= 1:
            rate = vcmd["rate"]; direct = vcmd["direct"]; vz = vcmd["vz"]
        else:
            rate = getattr(vel_meas, "rate", None)
            direct = getattr(vel_meas, "direct", None)
            vz = getattr(pos_meas, "z", None)

        att_times = self.attack_events.get(int(rid), [])
        attack_flag = 1 if any((sim_sec - 1) < t <= sim_sec for t in att_times) else 0

        def nz(x, default):
            try:
                return float(x)
            except Exception:
                return default

        return [nz(rate, 0.0), (nz(direct, 0.0) % 360.0), nz(vz, 0.0), int(attack_flag)]

    def add_vector_row(self, sim_sec, obs_vec, act_vec):
        row = [int(sim_sec)] + list(obs_vec) + list(act_vec)
        self.rows_for_csv.append(row)

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
        if a is not None:
            rec_action = dict(a); rec_action["age_sec"] = int(sim_sec) - int(a.get("t", sim_sec))
        else:
            rec_action = None
        self.buffer.append({"t": int(sim_sec), "red_id": int(red_id), "obs": obs_dict, "action": rec_action, "info": (extra_info or {})})

    def dump_csv(self, path="rl_traj.csv"):
        if not self.rows_for_csv:
            print("[RL] No vector data to dump (rows_for_csv is empty).", flush=True)
            return
        n_act = 16
        n_obs = len(self.rows_for_csv[0]) - 1 - n_act
        header = ["t"] + [f"obs_{i}" for i in range(n_obs)] + [f"act_{j}" for j in range(n_act)]

        abspath = os.path.abspath(path)
        outdir = os.path.dirname(abspath) or "."
        os.makedirs(outdir, exist_ok=True)

        with open(abspath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows_for_csv)
        print(f"[RL] Dumped {len(self.rows_for_csv)} rows to {abspath}", flush=True)


# ================= 红方控制器 =================
class RedForceController:
    def __init__(self, client, red_ids):
        self.client = client
        self.red_ids = set(int(x) for x in red_ids)
        self.ammo = {int(r): AMMO_MAX for r in red_ids}
        self.last_fire_time = {int(r): 0.0 for r in red_ids}

        self.assigned_target = {}      # {red_id: target_id}
        self.target_in_progress = set()
        self.destroyed_targets = set()
        self.destroyed_blue = set()
        self.destroyed_red = set()

        self._ended = False
        self._destroy_codes = {1}
        self.lock = threading.Lock()
        self.target_locks = {}         # {target_id: {"red_id": int, "until": float}}

        self._last_visible = set()
        self._last_locked = {}
        self._last_destroyed = set()

        self.recorder = RLRecorder()
        self._last_logged_sec = -1
        self._score_cache = None
        self._score_counter = 0

        # 开雷达 + 设初始速度
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

    # ---------- 基础工具 ---------- #
    def _safe_dump_now(self, path="rl_traj.csv"):
        try:
            self.recorder.dump_csv(path)
        except Exception as e:
            print("[RL] dump_csv failed in _safe_dump_now:", e, flush=True)

    def _is_blue_side(self, side):
        return side in (2, "SIDE_BLUE", "BLUE")

    def _is_destroyed_flag(self, dmg):
        if isinstance(dmg, (int, float)):
            return int(dmg) == 1
        s = str(dmg)
        return ("DESTROYED" in s.upper()) or (s.strip() == "1")

    def _is_fire_success(self, uid, max_tries=5, wait_s=0.1):
        if not uid:
            return False
        for _ in range(max_tries):
            time.sleep(wait_s)
            try:
                st = self.client.get_command_status(uid)
            except Exception:
                st = None
            if isinstance(st, dict):
                status = st.get('status', '')
                result = st.get('execute_result', '') or ''
                if status == 'EXECUTE_SUCCESS' or ('执行打击成功' in result):
                    return True
                if status == 'EXECUTE_FAILED':
                    return False
        return False

    def _fire_with_log(self, rid, tid):
        uid = self.client.set_target(vehicle_id=rid, target_id=tid)
        sim_t = self.client.get_sim_time()
        self.recorder.mark_action(rid, {"type": "fire", "target_id": int(tid), "uid": uid}, sim_t)
        return uid

    def _distance_m(self, lon1, lat1, lon2, lat2):
        if None in (lon1, lat1, lon2, lat2):
            return 1e18
        try:
            p1 = Position(x=lon1, y=lat1, z=0)
            p2 = Position(x=lon2, y=lat2, z=0)
            return self.client.get_distance_by_lon_lat(p1, p2)
        except Exception:
            from math import cos, radians, sqrt
            dx = (lon2 - lon1) * 111320.0 * cos(radians((lat1 + lat2) / 2.0))
            dy = (lat2 - lat1) * 110540.0
            return sqrt(dx * dx + dy * dy)

    def _get_target_pos(self, target_id, vis, all_pos):
        p = all_pos.get(target_id)
        if p is not None:
            return p.x, p.y
        for tracks in vis.values():
            for t in tracks:
                if t.get("target_id") == target_id:
                    return t.get("lon"), t.get("lat")
        return (None, None)

    def _fetch_score_with_retry(self, tries=20, wait=0.2, where="(unknown)"):
        score_obj = None
        for _ in range(tries):
            try:
                score_obj = self.client.get_score()
            except Exception:
                score_obj = None
            if score_obj:
                print(f"[Final Score {where}]", score_obj, flush=True)
                return score_obj
            time.sleep(wait)
        print(f"[Final Score {where}] still None after retry.", flush=True)
        return None

    def _end_simulation_once(self, reason=""):
        if self._ended:
            return
        self._ended = True
        if reason:
            print(f"[AutoStop] {reason}", flush=True)

        # 停之前拉一次（有些引擎此时已有结算）
        self._fetch_score_with_retry(tries=5, wait=0.2, where="pre-stop")

        try:
            self.client.stop()
        except Exception as e:
            print("[AutoStop] client.stop() failed:", e, flush=True)

        # 停后再尝试一阵
        self._fetch_score_with_retry(tries=20, wait=0.2, where="post-stop")

        # 赛后落盘
        try:
            self.recorder.dump_csv("rl_traj.csv")
        except Exception as e:
            print("[RL] dump_csv failed:", e, flush=True)

    def _update_destroyed_from_situ(self, force=False):
        if self._ended:
            return
        if not force:
            self._situ_counter = getattr(self, "_situ_counter", 0) + 1
            if self._situ_counter % 5 != 0:
                return
        try:
            situ_raw = self.client.get_situ_info() or {}
        except Exception:
            return

        any_new = False
        for vid, info in situ_raw.items():
            if not info:
                continue
            side = getattr(info, "side", None)
            if side is None and isinstance(info, dict):
                side = info.get("side")
            dmg = getattr(info, "damage_state", None)
            if dmg is None and isinstance(info, dict):
                dmg = info.get("damage_state")

            try:
                vvid = getattr(info, "id", None)
                if vvid is None and isinstance(info, dict):
                    vvid = info.get("id")
                if vvid is None:
                    vvid = int(vid)
                else:
                    vvid = int(vvid)
            except Exception:
                continue

            if not self._is_destroyed_flag(dmg):
                continue

            if self._is_blue_side(side):
                if vvid not in self.destroyed_blue:
                    self.destroyed_blue.add(vvid)
                    self.destroyed_targets.add(vvid)
                    any_new = True
            else:
                if vvid not in self.destroyed_red:
                    self.destroyed_red.add(vvid)
                    any_new = True
                    if vvid in self.ammo:
                        self.ammo[vvid] = 0
                    self.target_locks.pop(vvid, None)

        if any_new:
            print(f"[Situ] BLUE destroyed: {sorted(self.destroyed_blue)} | "
                  f"RED destroyed: {sorted(self.destroyed_red)}", flush=True)

        try:
            if len(self.destroyed_blue) >= 4:
                self._end_simulation_once("All 4 BLUE aircraft destroyed.")
                return
            if len(self.destroyed_red) >= 4:
                self._end_simulation_once("All 4 RED aircraft destroyed.")
                return
        except Exception:
            pass

    # ---------- 主循环一步：先采样记录，再执行攻击逻辑 ---------- #
    def step(self):
        if self._ended:
            return

        # ===== 观测（用于记录 obs/act）=====
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True)
            return
        vis = normalize_visible(raw_visible)

        try:
            all_pos = self.client.get_vehicle_pos()
        except Exception:
            all_pos = {}
        try:
            vel_all = self.client.get_vehicle_vel()
        except Exception:
            vel_all = {}

        # 定期刷新损毁集合
        self._update_destroyed_from_situ()

        sim_t = self.client.get_sim_time()
        sim_sec = int(sim_t)

        self._score_counter = getattr(self, "_score_counter", 0) + 1
        if self._score_counter % 5 == 0:
            try:
                self._score_cache = self.client.get_score() or None
            except Exception:
                self._score_cache = None

        # ====== 1Hz 记录 obs64 + act16 ======
        if sim_sec != getattr(self, "_last_logged_sec", -1):
            obs_concat = []
            act_concat = []

            for rid in sorted(self.red_ids):
                my_p = all_pos.get(rid)
                my_lon = getattr(my_p, "x", None) if my_p else None
                my_lat = getattr(my_p, "y", None) if my_p else None
                tracks = vis.get(rid, []) or []

                # 最近4蓝距
                dlist = []
                for t in tracks:
                    lonB, latB = t.get("lon"), t.get("lat")
                    if lonB is None or latB is None or my_lon is None or my_lat is None:
                        continue
                    try:
                        d = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                Position(x=lonB, y=latB, z=0))
                    except Exception:
                        d = None
                    if d is not None:
                        dlist.append(float(d))
                dlist.sort()
                if len(dlist) < 4:
                    dlist = dlist + [BLUE_DIST_CAP] * (4 - len(dlist))
                else:
                    dlist = dlist[:4]

                # 最近蓝机速度/方向（没有方向则用几何方位角）
                nb_speed, nb_dir = None, None
                if tracks and my_lon is not None and my_lat is not None:
                    best = None
                    best_t = None
                    for t in tracks:
                        lonB, latB = t.get("lon"), t.get("lat")
                        if lonB is None or latB is None:
                            continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                     Position(x=lonB, y=latB, z=0))
                        except Exception:
                            dd = None
                        if dd is not None and (best is None or dd < best):
                            best, best_t = dd, t
                    if best_t:
                        nb_speed = best_t.get("speed")
                        nb_dir = best_t.get("direction")
                        if nb_dir is None:
                            nb_dir = _bearing_deg_from_A_to_B(my_lon, my_lat, best_t.get("lon"), best_t.get("lat"))

                # 干扰区距离（有符号）
                jam_signed = _signed_dist_to_jam_boundary_m(self.client, my_lon, my_lat)

                # 打包 obs16
                vel_meas = vel_all.get(rid)
                obs_raw = {
                    "pos": {"lon": my_lon, "lat": my_lat, "alt": getattr(my_p, "z", None) if my_p else None},
                    "vel": {"vx": getattr(vel_meas, "vx", None) if vel_meas else None,
                            "vy": getattr(vel_meas, "vy", None) if vel_meas else None,
                            "vz": getattr(vel_meas, "vz", None) if vel_meas else None},
                    "ammo": int(self.ammo.get(rid, 0)),
                }
                obs16 = self.recorder.pack_single_obs16(
                    self.client, rid, obs_raw, BOUNDARY_RECT, jam_signed, dlist, nb_speed, nb_dir, vel_meas=vel_meas
                )
                obs_concat.extend(obs16)

                # 打包 act4
                act4 = self.recorder.pack_single_act4(rid, sim_sec, vel_meas=vel_meas, pos_meas=my_p)
                act_concat.extend(act4)

                # JSONL 便于排查（可选）
                self.recorder.record_tick(sim_sec, rid, {
                    "boundary_dists": {"left": obs16[0], "right": obs16[1], "down": obs16[2], "up": obs16[3]},
                    "vel": {"vx": obs16[4], "vy": obs16[5], "vz": obs16[6]},
                    "blue_dists": dlist,
                    "nearest_blue_speed": nb_speed,
                    "nearest_blue_dir": nb_dir,
                    "jam_signed_dist": obs16[13],
                    "ammo": obs16[14],
                    "speed_scalar": obs16[15],
                })

            self.recorder.latest_obs_vec = obs_concat
            self.recorder.latest_act_vec = act_concat
            self.recorder.add_vector_row(sim_sec, obs_concat, act_concat)

            # 可选的尺寸断言（调不通就注释掉）
            # assert len(obs_concat) == 64, f"obs_concat len={len(obs_concat)} != 64"
            # assert len(act_concat) == 16, f"act_concat len={len(act_concat)} != 16"

            self._last_logged_sec = sim_sec

        # ====== 执行红方攻击逻辑 ======
        if self._ended:
            return

        # 再拿一次可见与位置（保证用最新）
        try:
            raw_visible2 = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True)
            return
        vis2 = normalize_visible(raw_visible2)
        try:
            all_pos2 = self.client.get_vehicle_pos()
        except Exception:
            all_pos2 = {}

        # 刷新损毁集合
        self._update_destroyed_from_situ()

        now = self.client.get_sim_time()

        # 清理过期锁定
        expired = [tid for tid, meta in self.target_locks.items() if now >= meta["until"]]
        for tid in expired:
            self.target_locks.pop(tid, None)

        # 汇总“当前可见”的蓝机集合（去掉已损毁 + 正在锁定）
        visible_blue_targets = set()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if tid is None:
                    continue
                if tid < 10000:             # 只允许蓝机 ID
                    continue
                if tid in self.destroyed_targets:
                    continue
                if tid in self.target_locks:
                    continue
                visible_blue_targets.add(tid)

        # 只在变化时输出
        if visible_blue_targets != self._last_visible:
            print("未锁定的目标:", visible_blue_targets, flush=True)
            self._last_visible = set(visible_blue_targets)

        if self.target_locks != self._last_locked:
            print("已经锁定的目标:", self.target_locks, flush=True)
            self._last_locked = dict(self.target_locks)

        if self.destroyed_targets != self._last_destroyed:
            print("已经被摧毁的目标:", self.destroyed_targets, flush=True)
            self._last_destroyed = set(self.destroyed_targets)

        if not visible_blue_targets:
            return

        # 为每个蓝机挑最近可用红机
        used_reds_this_round = set()
        assignments = []  # (rid, tid)
        for tid in visible_blue_targets:
            t_lon, t_lat = self._get_target_pos(tid, vis2, all_pos2)
            if t_lon is None or t_lat is None:
                continue

            best_red, best_d = None, 1e18
            for rid in self.red_ids:
                if rid in used_reds_this_round:
                    continue
                if rid in self.destroyed_targets:
                    continue
                if self.ammo.get(rid, 0) <= 0:
                    continue
                if (now - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                    continue
                r_pos = all_pos2.get(rid)
                if r_pos is None:
                    continue
                d = self._distance_m(r_pos.x, r_pos.y, t_lon, t_lat)
                if d < best_d:
                    best_d, best_red = d, rid

            if best_red is not None:
                assignments.append((best_red, tid))
                used_reds_this_round.add(best_red)

        if not assignments:
            return

        # 下发攻击 + 锁定 20s（只有确认成功才扣弹/加锁/记冷却）
        for rid, tid in assignments:
            if tid in self.destroyed_targets:
                continue

            now_sim = self.client.get_sim_time()
            if (now_sim - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                continue
            if self.ammo.get(rid, 0) <= 0:
                continue

            try:
                uid = self._fire_with_log(rid, tid)
                print(f"[Red] {rid} -> fire at {tid}, uid={uid}", flush=True)

                ok = self._is_fire_success(uid, max_tries=5, wait_s=0.1)
                if not ok:
                    print(f"[Red] fire NOT confirmed for {rid}->{tid}, keep target available.", flush=True)
                    time.sleep(0.1)
                    continue

                # 成功：记录一次成功事件（用于 act4 的 attack_flag）
                self.recorder.mark_attack_success(rid, self.client.get_sim_time())

                self.ammo[rid] -= 1
                self.last_fire_time[rid] = self.client.get_sim_time()
                self.assigned_target[rid] = tid
                self.target_locks[tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}

                time.sleep(0.1)

            except Exception as e:
                print(f"[Red] set_target({rid},{tid}) failed: {e}", flush=True)
                continue

    # ---------- 循环 ----------
    def run_loop(self, stop_when_paused=False):
        try:
            while True:
                if self._ended:
                    break
                try:
                    if stop_when_paused and self.client.is_pause():
                        break
                except Exception:
                    pass
                self.step()
                time.sleep(SCAN_INTERVAL_SEC)
        finally:
            # 不管为何退出循环，都强制落一次盘
            self._safe_dump_now("rl_traj.csv")


# ================= 主程序 =================
if __name__ == "__main__":
    config = {
        "id": 106,                        # 方案号
        "config": RflysimEnvConfig("172.23.53.35", 16001, 18001)
    }
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    # 蓝方后台线程
    combat_system = CombatSystem(client)
    blue_thread = threading.Thread(target=combat_system.run_combat_loop, daemon=True)
    blue_thread.start()
    print("[Main] Blue combat loop started.", flush=True)

    # 开始仿真
    client.start()

    # 红方控制器
    red_ctrl = RedForceController(client, RED_IDS)
    print("[Main] Red control loop starting...", flush=True)
    print("[RL] Will write trajectory CSV to", os.path.abspath("rl_traj.csv"), flush=True)

    red_thread = threading.Thread(target=red_ctrl.run_loop, daemon=True)
    red_thread.start()
    print("[Main] Red control loop started.", flush=True)

    try:
        while True:
            time.sleep(2.0)
            if not red_ctrl._ended:
                score = client.get_score()
                if score:
                    print("[Score]", score, flush=True)
            else:
                # 结束后再抢一次最终分数
                final_score = client.get_score()
                if not final_score:
                    for _ in range(10):
                        time.sleep(0.2)
                        final_score = client.get_score()
                        if final_score:
                            break
                print("[Score Final in main]", final_score, flush=True)
                break
    except KeyboardInterrupt:
        print("[Main] Interrupted, exiting...", flush=True)
    finally:
        print("[Main] CWD =", os.path.abspath("."), flush=True)
        try:
            red_ctrl._safe_dump_now("rl_traj.csv")
        except Exception as e:
            print("[Main] safe dump failed:", e, flush=True)
