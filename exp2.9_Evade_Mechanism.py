# -*- coding: utf-8 -*-
"""
批量采集多轮（例如 1000 轮）对抗的 obs64+act16 数据。
- 每轮将数据写入 runs/<timestamp>/ep_XXXX.csv
- 写入一个 runs/<timestamp>/manifest.json 索引
- 依赖：你上一版“完整整合”的脚本内容（类 RedForceController / RLRecorder / normalize_visible 等）
  本文件已内嵌必要代码（可独立运行）。
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
"""
# 给后端资源释放缓冲，加随机抖动更温和
import random
import os
import csv
import json
import time
import re
import threading
from collections import defaultdict
from datetime import datetime

import rflysim
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

# ==================== 你的原始常量（可按需调整） ====================
DESTROYED_FLAG = "DAMAGE_STATE_DESTROYED"
RED_IDS = [10091, 10084, 10085, 10086]

AMMO_MAX = 8
ATTACK_COOLDOWN_SEC = 2.0
SCAN_INTERVAL_SEC = 0.1
LOCK_SEC = 40.0
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
    """
    以经纬度计算两点大圆距离（单位: m），不再依赖 sim 的距离接口。
    """
    import math
    if None in (lon1, lat1, lon2, lat2):
        return None
    R = 6371000.0  # 地球半径 (m)
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def _direct_rate_from_vx_vy(vx, vy):
    """
    从 (vx, vy) 反算:
      - direct: 水平速度大小 (标量)
      - rate  : 水平速度方向 (deg), 0=正北, 顺时针
    文档约定: vx=正北分量, vy=正东方向分量
    因此航向角=atan2(v_east, v_north) -> 度, 再规约到 [0, 360)
    """
    from math import sqrt, atan2, degrees
    try:
        vxn = float(vx) if vx is not None else 0.0
        vye = float(vy) if vy is not None else 0.0
    except Exception:
        vxn, vye = 0.0, 0.0
    direct = sqrt(vxn * vxn + vye * vye)
    rate = (degrees(atan2(vye, vxn)) + 360.0) % 360.0
    return direct, rate

def _ang_norm(deg):
    return (float(deg) + 360.0) % 360.0

def _ang_diff_abs(a, b):
    """返回最小夹角绝对值 [0, 180]"""
    d = abs(_ang_norm(a) - _ang_norm(b))
    return d if d <= 180.0 else 360.0 - d

def _bearing_deg_A_to_B(lonA, latA, lonB, latB):
    # 与 _bearing_deg_from_A_to_B 一样，这里提供一个更短名包装，方便调用
    return _bearing_deg_from_A_to_B(lonA, latA, lonB, latB)

def _is_missile_track(tid):
    """ 导弹 ID 规则：< 10000 """
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
        if k.startswith('_'):
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
    if not isinstance(s, str):
        return None
    m_id = re.search(r'\btarget_id\s*:\s*(\d+)', s) or re.search(r'\bid\s*:\s*(\d+)', s)
    if not m_id:
        return None
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
        if dire < 0:
            dire += 360.0
    return {"target_id": tid, "lon": lon, "lat": lat, "speed": spd, "direction": dire}

def _extract_track_fields_any(t_like):
    td = _as_dict(t_like)
    tid = td.get('target_id') or td.get('id')
    if tid is None:
        return None
    tid = int(tid)
    pos  = td.get('target_pos') or td.get('target pos') or {}
    posd = _as_dict(pos)
    lon  = posd.get('x'); lat = posd.get('y')
    spd  = td.get('target_speed') or td.get('speed')
    dire = td.get('target_direction') or td.get('target direction') or td.get('direction') or td.get('dir')
    try:
        if spd is not None:
            spd  = float(spd)
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
        self.has_dumped = False  # 只写一次

    def pack_single_obs16(self, client, rid, obs, boundary_rect, jam_signed_dist, blue4_dists,
                          nearest_blue_speed, nearest_blue_dir, vel_meas=None):
        left, right, down, up = _dist_to_boundary_m(client, obs["pos"]["lon"], obs["pos"]["lat"])
        vx, vy, vz_meas = obs["vel"]["vx"], obs["vel"]["vy"], obs["vel"]["vz"]
        ammo = obs["ammo"]
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
        if len(obs16) == 15:
            # 一律用 vx, vy 计算水平速度标量（不再信任 vel_meas.direct）
            from math import sqrt
            speed_scalar = sqrt(float(vx or 0.0) ** 2 + float(vy or 0.0) ** 2)
            obs16.append(speed_scalar)

        return obs16

    def pack_single_act4(self, rid, sim_sec, vel_meas, pos_meas):
        """
        act4:
          0: rate   (deg, 0=北, 顺时针)
          1: direct (水平速度大小)
          2: vz     (沿用旧逻辑: 优先命令缓存; 否则 pos_meas.z 作为“高度兜底”)
          3: 攻击标志
        说明:
          - 若最近1秒内我们下发过 vel 命令, 则优先用命令(rate, direct, vz)
          - 否则从实测 vel_meas.vx/vy 反算 (rate, direct), vz 仍按旧逻辑兜底
        """
        vcmd = self.last_vel_cmd.get(int(rid))

        # 攻击标志（保持原样）
        att_times = self.attack_events.get(int(rid), [])
        attack_flag = 1 if any((sim_sec - 1) < t <= sim_sec for t in att_times) else 0

        def nz(x, default):
            try:
                return float(x)
            except Exception:
                return default

        if vcmd and (sim_sec - int(vcmd["t"])) <= 1:
            # 最近 1 秒内有我们下发的命令 -> 直接用命令数值
            rate_cmd = nz(vcmd.get("rate"), 0.0)
            direct_cmd = nz(vcmd.get("direct"), 0.0)
            vz_cmd = nz(vcmd.get("vz"), nz(getattr(pos_meas, "z", None), 0.0))
            return [rate_cmd % 360.0, direct_cmd, vz_cmd, int(attack_flag)]

        # 否则用实测 vx, vy 反算 rate/direct
        vx = getattr(vel_meas, "vx", None) if vel_meas else None
        vy = getattr(vel_meas, "vy", None) if vel_meas else None
        direct_calc, rate_calc = _direct_rate_from_vx_vy(vx, vy)

        # vz 仍按旧逻辑兜底（和你原来的数据对齐：没有命令就拿 pos_meas.z）
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
        if self.has_dumped:
            return
        if not self.rows_for_csv:
            print("[RL] No vector data to dump:", path, flush=True)
            self.has_dumped = True
            return
        n_act = 16
        n_obs = len(self.rows_for_csv[0]) - 1 - n_act
        header = ["t"] + [f"obs_{i}" for i in range(n_obs)] + [f"act_{j}" for j in range(n_act)]
        abspath = os.path.abspath(path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows_for_csv)
        print(f"[RL] Dumped {len(self.rows_for_csv)} rows to {abspath}", flush=True)
        self.has_dumped = True

# ==================== 红方控制器（缩略自你上一版，逻辑一致） ====================
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
        # --- 调试/打印控制导弹的轨迹 ---
        self.DEBUG_EVADE = True
        self.MISSILE_DEBUG_PRINT_INTERVAL = 0.5  # 每架红机最小打印间隔（秒）
        self._last_debug_print = {}              # rid -> 上次打印墙钟时间
        self._last_approach_flag = {}           # rid -> 上次是否判定为“接近”

        # --- 躲避导弹相关常量 ---
        self.MISSILE_THREAT_DIST_M = 50000.0      # 威胁距离阈值
        self.MISSILE_BEARING_THRESH_DEG = 25.0   # 视线角阈值：导弹航向与“导弹->红机连线”夹角小于该值视为“冲向我”
        self.EVASIVE_TURN_DEG = 90.0             # 机动：相对导弹方向 90°
        self.EVASIVE_SPEED_MIN = 220.0           # 躲避时最低水平速度（m/s），会与当前速度取 max
        self.EVASIVE_DURATION_SEC = 10.0          # 每次躲避持续时长（限制攻击）
        self.EVASIVE_COOLDOWN_SEC = 5.0          # 躲避冷却：避免频繁反复机动

        # --- 躲避导弹状态 ---
        self._missile_last_dist = {}             # key=(rid, mid) -> 上次测得距离（米）
        # --- 导弹轨迹缓存：mid -> {"lon": float, "lat": float, "t": float}
        self._missile_last = {}
        self._evasive_until = {}                 # rid -> 时间戳（在此之前禁止开火）
        self._last_evasive_time = {}             # rid -> 最近一次触发躲避的时间戳

        # --- 全灭后延迟结束（20s） ---
        self._end_grace_until = None   # None 或 墙钟时间戳（time.time()）
        self._end_reason = ""



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

    # ---- 内部工具 ----
    def _estimate_msl_heading_speed(self, mid, lon_now, lat_now, t_now):
        """
        基于上一次缓存位置，估计导弹航向(度，0北顺时针) 与 地速(m/s)。
        如果没有历史点或 dt 太小，返回 (None, None)，并只更新缓存。
        """
        prev = self._missile_last.get(int(mid))
        self._missile_last[int(mid)] = {"lon": float(lon_now), "lat": float(lat_now), "t": float(t_now)}
        if not prev:
            return (None, None)

        dt = float(t_now) - float(prev["t"])
        if dt <= 1e-3:
            return (None, None)

        # 航向：上一点 -> 当前点
        mdir = _bearing_deg_from_A_to_B(prev["lon"], prev["lat"], lon_now, lat_now)  # deg
        # 地速：两点大圆距离 / dt
        dist = _geo_dist_haversine_m(prev["lon"], prev["lat"], lon_now, lat_now) or 0.0
        mspeed = dist / dt  # m/s
        return (mdir, mspeed)

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
        d = _geo_dist_haversine_m(lon1, lat1, lon2, lat2)
        return float(d) if d is not None else 1e18

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
        self._fetch_score_with_retry(tries=5, wait=0.2, where="pre-stop")
        try:
            self.client.stop()
        except Exception as e:
            print("[AutoStop] client.stop() failed:", e, flush=True)
        self._fetch_score_with_retry(tries=20, wait=0.2, where="post-stop")
        try:
            self.recorder.dump_csv(self.out_csv_path)
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
            side = getattr(info, "side", None) if not isinstance(info, dict) else info.get("side")
            dmg  = getattr(info, "damage_state", None) if not isinstance(info, dict) else info.get("damage_state")
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
                    self.destroyed_blue.add(vvid); self.destroyed_targets.add(vvid); any_new = True
            else:
                if vvid not in self.destroyed_red:
                    self.destroyed_red.add(vvid); any_new = True
                    if vvid in self.ammo:
                        self.ammo[vvid] = 0
                    self.target_locks.pop(vvid, None)
        if any_new:
            print(f"[Situ] BLUE destroyed: {sorted(self.destroyed_blue)} | RED destroyed: {sorted(self.destroyed_red)}", flush=True)
        # === 全灭后延迟 20s 才真正结束 ===
        import time as _time
        now_wall = _time.time()

        # 首次触发：设置 20s 优雅收尾窗口
        if (len(self.destroyed_blue) >= 4 or len(self.destroyed_red) >= 4) and self._end_grace_until is None:
            if len(self.destroyed_blue) >= 4:
                self._end_reason = "All 4 BLUE aircraft destroyed."
            else:
                self._end_reason = "All 4 RED aircraft destroyed."
            self._end_grace_until = now_wall + 20.0
            print(f"[Grace] {self._end_reason} Ending in 20s...", flush=True)
            return  # 本次先不结束，等下次再检查是否到时

        # 已经在优雅收尾窗口中：到点就结束
        if self._end_grace_until is not None:
            if now_wall >= self._end_grace_until:
                self._end_simulation_once(self._end_reason)
            # 不到点就什么也不做，让主循环继续


    # ---- 主一步：先采样、再执行攻击逻辑 ----
    def step(self):
        if self._ended:
            return
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True); return
        vis = normalize_visible(raw_visible)
        try:
            all_pos = self.client.get_vehicle_pos()
        except Exception:
            all_pos = {}
        try:
            vel_all = self.client.get_vehicle_vel()
        except Exception:
            vel_all = {}

        self._update_destroyed_from_situ()
        sim_t = self.client.get_sim_time()
        sim_sec = int(sim_t)
        self._score_counter = getattr(self, "_score_counter", 0) + 1
        if self._score_counter % 5 == 0:
            try:
                self._score_cache = self.client.get_score() or None
            except Exception:
                self._score_cache = None

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
                    dlist += [BLUE_DIST_CAP] * (4 - len(dlist))
                else:
                    dlist = dlist[:4]

                nb_speed, nb_dir = None, None
                if tracks and my_lon is not None and my_lat is not None:
                    best, best_t = None, None
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

                jam_signed = _signed_dist_to_jam_boundary_m(self.client, my_lon, my_lat)
                vel_meas = vel_all.get(rid)
                obs_raw = {
                    "pos": {"lon": my_lon, "lat": my_lat, "alt": getattr(my_p, "z", None) if my_p else None},
                    "vel": {"vx": getattr(vel_meas, "vx", None) if vel_meas else None,
                            "vy": getattr(vel_meas, "vy", None) if vel_meas else None,
                            "vz": getattr(vel_meas, "vz", None) if vel_meas else None},
                    "ammo": int(self.ammo.get(rid, 0)),
                }
                obs16 = self.recorder.pack_single_obs16(self.client, rid, obs_raw, BOUNDARY_RECT, jam_signed,
                                                        dlist, nb_speed, nb_dir, vel_meas=vel_meas)
                obs_concat.extend(obs16)
                act4 = self.recorder.pack_single_act4(rid, sim_sec, vel_meas=vel_meas, pos_meas=my_p)
                act_concat.extend(act4)
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
            self._last_logged_sec = sim_sec

        if self._ended:
            return


        # —— 执行攻击逻辑（与单轮版一致）——
        try:
            raw_visible2 = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True);
            return
        vis2 = normalize_visible(raw_visible2)
        try:
            all_pos2 = self.client.get_vehicle_pos()
        except Exception:
            all_pos2 = {}
        self._update_destroyed_from_situ()
        now = self.client.get_sim_time()

        # 导弹方向兜底所需：拿到所有载具（含导弹）的实测速度
        try:
            vel_all2 = self.client.get_vehicle_vel()
        except Exception:
            vel_all2 = {}

        # ===== 调试：导弹原始信息表（每秒一次）=====
        if not hasattr(self, "_last_raw_msl_print_sec"):
            self._last_raw_msl_print_sec = -1
        if sim_sec != self._last_raw_msl_print_sec:
            self._last_raw_msl_print_sec = sim_sec
            try:
                # 收集导弹：从 all_pos2 或可见表里提取 <10000 的 id
                missile_ids = set()
                for vid in (all_pos2 or {}):
                    try:
                        if int(vid) < 10000:
                            missile_ids.add(int(vid))
                    except Exception:
                        pass
                for tracks in (vis2 or {}).values():
                    for t in (tracks or []):
                        tid = t.get("target_id")
                        if tid is not None and int(tid) < 10000:
                            missile_ids.add(int(tid))

                if missile_ids:
                    print("[RAW MISSILES @{}] ids={}".format(sim_sec, sorted(missile_ids)), flush=True)
                    for mid in sorted(missile_ids):
                        mpos = all_pos2.get(mid)
                        mm = vel_all2.get(mid) if vel_all2 else None
                        # 原始 track（从各红机视角里挑一条有该 mid 的，打印原样）
                        raw_track = None
                        for det_id, tracks in (raw_visible2 or {}).items():
                            if isinstance(tracks, list):
                                for item in tracks:
                                    td = _as_dict(item)
                                    if int(td.get("target_id", -1)) == int(mid):
                                        raw_track = td
                                        break
                            elif isinstance(tracks, dict):
                                for _, item in tracks.items():
                                    td = _as_dict(item)
                                    if int(td.get("target_id", -1)) == int(mid):
                                        raw_track = td
                                        break
                            if raw_track:
                                break

                        # 打印一行：原始 track、位置、速度（vx/vy/vz & 直读的 direct/rate）
                        print(
                            "[RAW] mid={} track={} pos=({:.6f},{:.6f}) vel={{vx:{}, vy:{}, vz:{}, direct:{}, rate:{}}}".format(
                                mid,
                                raw_track,
                                getattr(mpos, "x", float("nan")) if mpos else float("nan"),
                                getattr(mpos, "y", float("nan")) if mpos else float("nan"),
                                getattr(mm, "vx", None) if mm else None,
                                getattr(mm, "vy", None) if mm else None,
                                getattr(mm, "vz", None) if mm else None,
                                getattr(mm, "direct", None) if mm else None,
                                getattr(mm, "rate", None) if mm else None
                            ),
                            flush=True
                        )
            except Exception as e:
                print("[RAW MISSILES] dump failed:", e, flush=True)

        # 清理过期锁
        expired = [tid for tid, meta in self.target_locks.items() if now >= meta["until"]]
        for tid in expired:
            self.target_locks.pop(tid, None)

        # === Missile Evasion: 识别导弹 & 判定威胁 & 触发躲避（含调试打印）===
        # === Missile Evasion: 识别导弹 & 判定威胁 & 触发躲避（含详细调试打印）===
        # 说明：
        # - mdir_calc：用导弹上一帧与当前帧位置的方位角(北=0°, 顺时针)估算
        # - mspeed_calc：用两帧大圆距离 / Δt 估算（m/s）
        # - 最近红机：对每一枚导弹，找距它最近的红机 rid_near
        # - 然后对每一架红机 rid，再找离它最近的一枚导弹做威胁判定（保持“每红机只处理一枚最近导弹”的结构）
        # - 统一打印：最近红机ID、mdir_calc、los_m2red、ang_diff、approach、threat、cooldown_ok、evading_now、reason
        # - 触发时额外打印：evade_heading（红机将采取的机动方向角）

        if not hasattr(self, "_missile_prev_xy_t"):
            # mid -> (lon, lat, sim_t)  用于推算导弹航向与标量速度
            self._missile_prev_xy_t = {}

        missiles = []  # list of dict(mid, lon, lat, mspeed_calc, mdir_calc, mspeed_radar)

        # 1) 收集 <10000 的“导弹” + 推算 mdir / mspeed
        now_sim = self.client.get_sim_time()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if not _is_missile_track(tid):
                    continue
                mid = int(tid)
                # 位置来源：优先 all_pos2（更实时/统一），否则用 track 中的 target_pos
                mpos = all_pos2.get(mid)
                if mpos and getattr(mpos, "x", None) is not None and getattr(mpos, "y", None) is not None:
                    mlon, mlat = float(mpos.x), float(mpos.y)
                else:
                    mlon = t.get("lon");
                    mlat = t.get("lat")
                if mlon is None or mlat is None:
                    continue

                mdir_calc, mspeed_calc = None, None
                prev = self._missile_prev_xy_t.get(mid)
                if prev and prev[0] is not None and prev[1] is not None:
                    prev_lon, prev_lat, prev_t = prev
                    dt = max(1e-3, float(now_sim - float(prev_t)))
                    # 方位角（导弹上一帧 -> 当前帧）
                    mdir_calc = _bearing_deg_from_A_to_B(prev_lon, prev_lat, mlon, mlat)
                    # 标量速度（m/s）：两帧大圆距离 / Δt
                    d_m = _geo_dist_haversine_m(prev_lon, prev_lat, mlon, mlat)
                    if d_m is not None:
                        mspeed_calc = d_m / dt

                # 更新上一帧缓存
                self._missile_prev_xy_t[mid] = (mlon, mlat, now_sim)

                missiles.append({
                    "mid": mid,
                    "lon": mlon,
                    "lat": mlat,
                    "mdir": mdir_calc,  # 计算得到的“导弹速度方向角”
                    "mspeed": mspeed_calc,  # 计算得到的速度（m/s）
                    "mspeed_radar": t.get("speed", None)  # 雷达报的速度（若有）
                })

        # 2) 为每枚导弹找“最近红机”，便于打印你要的“最近红方无人机ID”
        nearest_red_of_missile = {}  # mid -> (rid_near, dist_m)
        if missiles:
            for m in missiles:
                mid, mlon, mlat = m["mid"], m["lon"], m["lat"]
                rid_near, d_near = None, None
                for rid in self.red_ids:
                    rp = all_pos2.get(rid)
                    if not rp or getattr(rp, "x", None) is None or getattr(rp, "y", None) is None:
                        continue
                    rlon, rlat = float(rp.x), float(rp.y)
                    d = _geo_dist_haversine_m(mlon, mlat, rlon, rlat)
                    if d is None:
                        continue
                    if d_near is None or d < d_near:
                        rid_near, d_near = rid, float(d)
                if rid_near is not None:
                    nearest_red_of_missile[mid] = (rid_near, d_near)

        # 3) 逐红机：挑离它最近的一枚导弹，完成威胁判定 + 打印
        if missiles:
            now_wall = time.time()
            for rid in sorted(self.red_ids):
                my_p = all_pos2.get(rid)
                if not my_p or getattr(my_p, "x", None) is None or getattr(my_p, "y", None) is None:
                    continue
                my_lon, my_lat = float(my_p.x), float(my_p.y)

                # 找“对这架红机来说最近”的导弹
                best_mid, best_dist, best_meta = None, None, None
                for m in missiles:
                    d = _geo_dist_haversine_m(m["lon"], m["lat"], my_lon, my_lat)
                    if d is None:
                        continue
                    if best_dist is None or d < best_dist:
                        best_mid, best_dist, best_meta = m["mid"], float(d), m
                if best_mid is None:
                    continue

                # 计算导弹->红机的视线角、夹角与“逼近”判定
                mdir = best_meta.get("mdir")  # 计算得到的导弹方向角（可能为 None，刚出现时）
                los_m2red = _bearing_deg_from_A_to_B(best_meta["lon"], best_meta["lat"], my_lon, my_lat)
                ang_diff = _ang_diff_abs(mdir, los_m2red) if (mdir is not None and los_m2red is not None) else None

                approach_ok = False
                reason = "unknown"
                if ang_diff is not None:
                    if ang_diff <= self.MISSILE_BEARING_THRESH_DEG:
                        approach_ok = True
                        reason = f"angle_ok({ang_diff:.1f}<= {self.MISSILE_BEARING_THRESH_DEG})"
                    else:
                        reason = f"angle_large({ang_diff:.1f})"
                else:
                    # 距离趋势兜底（记录每红机-导弹对的上次距离）
                    key = (rid, best_mid, "dist")
                    last_d = self._missile_last_dist.get(key, None)
                    if last_d is not None and best_dist < last_d:
                        approach_ok = True
                        reason = "dist_decreasing"
                    else:
                        reason = "no_dir_and_no_decrease"
                    self._missile_last_dist[key] = best_dist

                # 威胁范围、冷却、是否在躲避窗口
                in_threat = (best_dist is not None and best_dist <= self.MISSILE_THREAT_DIST_M)
                last_ev = self._last_evasive_time.get(rid, -1e9)
                cooldown_ok = (now_wall - last_ev) >= self.EVASIVE_COOLDOWN_SEC
                in_evasive_window = (self._evasive_until.get(rid, 0.0) > now_sim)

                # 最近红机（从“导弹视角”）是谁（用于打印）
                rid_near, d_near = nearest_red_of_missile.get(best_mid, (None, None))

                # === 统一详细打印（带你要的全部字段） ===
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

                # === 触发躲避 ===
                if in_threat and approach_ok and cooldown_ok:
                    # 取导弹参考航向（无mdir则用los兜底）
                    use_mdir = mdir if mdir is not None else (los_m2red or 0.0)
                    cand1 = _ang_norm(use_mdir + self.EVASIVE_TURN_DEG)
                    cand2 = _ang_norm(use_mdir - self.EVASIVE_TURN_DEG)
                    los_red2m = _bearing_deg_from_A_to_B(my_lon, my_lat, best_meta["lon"], best_meta["lat"])
                    away_dir = _ang_norm(los_red2m + 180.0) if los_red2m is not None else None

                    if away_dir is not None:
                        # ✅ 选择更接近“背离导弹”的 90° 候选（注意：<=）
                        score1 = _ang_diff_abs(cand1, away_dir)
                        score2 = _ang_diff_abs(cand2, away_dir)
                        evade_heading = cand1 if score1 <= score2 else cand2
                    else:
                        evade_heading = cand1

                    # 水平速度：max(当前, 最小)
                    v_me = vel_all2.get(rid) if vel_all2 else None
                    v_direct_now, _ = _direct_rate_from_vx_vy(
                        getattr(v_me, "vx", 0.0) if v_me else 0.0,
                        getattr(v_me, "vy", 0.0) if v_me else 0.0
                    )
                    v_direct_cmd = max(float(v_direct_now or 0.0), self.EVASIVE_SPEED_MIN)
                    vz_keep = getattr(v_me, "vz", 0.0) if v_me else 0.0

                    try:
                        vcmd = rflysim.Vel()
                        #仿真接口的速度和方向角是反过来的 是环境的问题
                        vcmd.rate = float(v_direct_cmd)
                        vcmd.direct = float(evade_heading)
                        vcmd.vz = float(vz_keep)
                        uid = self.client.set_vehicle_vel(rid, vcmd)
                        self.recorder.mark_vel_cmd(
                            rid, rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=now_sim
                        )
                        # 这里额外打印“红机将采取的机动方向角”
                        print(
                            f"[EVADE] rid={rid} mid={best_mid} "
                            f"CMD: rate(红机将采取的机动)={vcmd.rate:.1f}° direct={vcmd.direct:.1f} vz={vcmd.vz:.1f} "
                            f"(dist={best_dist:.0f}m, 导弹速度方向角：={mdir if mdir is not None else -1:.1f}°, reason={reason})",
                            flush=True
                        )
                    except Exception as e:
                        print(f"[EVADE] set_vehicle_vel({rid}) failed: {e}", flush=True)

                    self._evasive_until[rid] = now_sim + self.EVASIVE_DURATION_SEC
                    self._last_evasive_time[rid] = now_wall

        # === 原攻击目标搜集 ===
        visible_blue_targets = set()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if tid is None or tid < 10000:
                    continue
                if tid in self.destroyed_targets or tid in self.target_locks:
                    continue
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
        if not visible_blue_targets:
            return

        used_reds_this_round = set()
        assignments = []
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
                # 在躲避窗口内不参与开火
                if self._evasive_until.get(rid, 0.0) > now:
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

        for rid, tid in assignments:
            if tid in self.destroyed_targets:
                continue
            now_sim = self.client.get_sim_time()
            # 躲避窗口内不执行打击
            if self._evasive_until.get(rid, 0.0) > now_sim:
                continue
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
                    time.sleep(0.1);
                    continue
                self.recorder.mark_attack_success(rid, self.client.get_sim_time())
                self.ammo[rid] -= 1
                self.last_fire_time[rid] = self.client.get_sim_time()
                self.assigned_target[rid] = tid
                self.target_locks[tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}
                time.sleep(0.1)
            except Exception as e:
                print(f"[Red] set_target({rid},{tid}) failed: {e}", flush=True);
                continue

    def run_loop(self, stop_when_paused=False, max_wall_time_sec=None):
        start_t = time.time()
        try:
            while True:
                if self._ended:
                    break
                if stop_when_paused:
                    try:
                        if self.client.is_pause():
                            break
                    except Exception:
                        pass
                if max_wall_time_sec is not None and (time.time() - start_t) > max_wall_time_sec:
                    print("[Red] Episode timeout reached, stopping...", flush=True)
                    self._end_simulation_once("Timeout")
                    break
                self.step()
                time.sleep(SCAN_INTERVAL_SEC)
        finally:
            try:
                self.recorder.dump_csv(self.out_csv_path)
            except Exception as e:
                print("[RL] dump_csv on finally failed:", e, flush=True)

# ==================== 多轮批量 Runner（改动点 1：safe_reset 幂等、只复位一次） ====================
def safe_reset(client):
    """
    只尝试一次“原生复位/重启”，成功后立刻返回 True；不在此处做额外 start()。
    若均不可用，再 fallback 为 stop()->sleep->start()，其中 start() 报“容器正在运行”也视为成功。
    """
    for fn_name in ["restart", "reset", "reset_scene", "reset_scenario"]:
        fn = getattr(client, fn_name, None)
        if callable(fn):
            try:
                fn()
                print(f"[Runner] client.{fn_name}() called.", flush=True)
                return True
            except Exception as e:
                msg = str(e)
                if "容器正在运行" in msg or "already running" in msg.lower():
                    print(f"[Runner] client.{fn_name}(): container already running, continue.", flush=True)
                    return True
                print(f"[Runner] client.{fn_name}() failed: {e}", flush=True)
                continue

    # 硬重启兜底
    try:
        client.stop()
    except Exception:
        pass
    time.sleep(1.0)
    try:
        client.start()
        print("[Runner] fallback stop()->start() used.", flush=True)
        return True
    except Exception as e:
        msg = str(e)
        if "容器正在运行" in msg or "already running" in msg.lower():
            print("[Runner] fallback start(): container already running, continue.", flush=True)
            return True
        print("[Runner] fallback start failed:", e, flush=True)
        return False

# ==================== 改动点 2：run_one_episode 统一整轮并返回 (success, score) ====================
def run_one_episode(client, plan_id, out_csv_path, max_wall_time_sec=360, min_wall_time_sec=10):
    """
    执行一整轮并返回 (success: bool, score_obj: dict|None)
    统一在这里做：预停 -> 复位 -> 启动蓝军/红军线程 -> 等待结束/超时 -> 停止 -> 拉取分数 -> 落盘
    """
    # 预停（忽略失败）
    try:
        client.stop()
        print("[Runner] pre-episode stop()", flush=True)
    except Exception:
        pass
    time.sleep(0.5)

    # 复位（只做一次）
    ok = safe_reset(client)
    if not ok:
        print("[Runner] safe_reset failed; give up this episode.", flush=True)
        return False, None

    # === 启动蓝军线程（时序关键：场景已在运行态） ===
    def _blue_wrapper():
        try:
            blue = CombatSystem(client)
            blue.run_combat_loop()
        except Exception as e:
            print("[Runner] Blue loop exception:", e, flush=True)

    blue_thread = threading.Thread(target=_blue_wrapper, daemon=True)
    blue_thread.start()
    print("[Runner] Blue loop started.", flush=True)

    # === 启动红军控制线程 ===
    red_ctrl = RedForceController(client, RED_IDS, out_csv_path)
    red_thread = threading.Thread(
        target=lambda: red_ctrl.run_loop(max_wall_time_sec=max_wall_time_sec),
        daemon=True
    )
    red_thread.start()
    print("[Runner] Red loop started.", flush=True)

    # === 等待结束条件 ===
    t0 = time.time()
    success = False
    score_obj = None

    try:
        # 轮询等待：红控在“全灭/超时”内会自结束并 dump
        while True:
            if not red_thread.is_alive():
                success = True
                break

            if (time.time() - t0) > max_wall_time_sec:
                print("[Runner] Episode timeout reached, stopping...", flush=True)
                try:
                    client.stop()
                except Exception:
                    pass
                break

            time.sleep(0.5)
    finally:
        # 统一收尾
        try:
            client.stop()
        except Exception:
            pass
        # 拉分：多试几次（仿真刚停可能分数未刷新）
        try:
            if hasattr(red_ctrl, "_fetch_score_with_retry"):
                score_obj = red_ctrl._fetch_score_with_retry(tries=10, wait=0.2, where="runner-final")
        except Exception:
            score_obj = None

        wall = time.time() - t0
        if wall < min_wall_time_sec:
            success = False
            print(f"[Runner] Episode too short ({wall:.1f}s) -> mark as failed.", flush=True)

    return success, score_obj

# ==================== 改动点 3：main 精简每轮流程（不再二次 reset/start） ====================
def main():
    # === 可调参数 ===
    EPISODES = 1000
    MAX_WALL_TIME_PER_EP = 600      # 每轮最多持续秒数（到时自动结束）
    MIN_WALL_TIME_PER_EP = 10       # 少于该时长认为异常/无效轮
    HOST, PORT_CMD, PORT_DATA = "172.23.53.35", 16001, 18001
    PLAN_ID = 106

    # 输出目录
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.abspath(os.path.join("runs", run_tag))
    os.makedirs(out_root, exist_ok=True)
    print("[Runner] Output root:", out_root, flush=True)

    # 客户端
    config = {"id": PLAN_ID, "config": RflysimEnvConfig(HOST, PORT_CMD, PORT_DATA)}
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    manifest = {"plan_id": PLAN_ID, "episodes": []}

    for ep in range(1, EPISODES + 1):
        print(f"\n[Runner] ===== Episode {ep}/{EPISODES} =====", flush=True)

        out_csv = os.path.join(out_root, f"ep_{ep:04d}.csv")
        success, score = run_one_episode(
            client, PLAN_ID, out_csv,
            MAX_WALL_TIME_PER_EP, MIN_WALL_TIME_PER_EP
        )

        manifest["episodes"].append({
            "episode": ep,
            "csv": os.path.abspath(out_csv),
            "success": bool(success),
            "score": score
        })

        # 给后端 资源释放 缓冲（稍长一点更稳）
        time.sleep(2.0 + random.uniform(0.0, 2.0))

    # 写索引文件
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[Runner] Manifest written to:", manifest_path, flush=True)
    print("[Runner] Done. Total episodes:", EPISODES, flush=True)

if __name__ == "__main__":
    main()
