# -*- coding: utf-8 -*-
"""
批量采集多轮（例如 1000 轮）对抗的 obs64+act16 数据。
- 每轮将数据写入 runs/<timestamp>/ep_XXXX.csv
- 写入一个 runs/<timestamp>/manifest.json 索引
- 依赖：你上一版“完整整合”的脚本内容（类 RedForceController / RLRecorder / normalize_visible 等）
  本文件已内嵌必要代码（可独立运行）。
"""

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

def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    out = {}
    for k in dir(obj):
        if k.startswith('_'): continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
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
    if not isinstance(visible, dict):
        return out
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
            item = _parse_track_from_string_rich(v);
            if item: tracks.append(item)
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
            speed_scalar = None
            if vel_meas is not None:
                speed_scalar = getattr(vel_meas, "direct", None)
                if speed_scalar is None:
                    from math import sqrt
                    speed_scalar = sqrt(float(vx or 0.0)**2 + float(vy or 0.0)**2)
            obs16.append(float(speed_scalar) if speed_scalar is not None else 0.0)
        return obs16

    def pack_single_act4(self, rid, sim_sec, vel_meas, pos_meas):
        vcmd = self.last_vel_cmd.get(int(rid))
        if vcmd and (sim_sec - int(vcmd["t"])) <= 1:
            rate = vcmd["rate"]; direct = vcmd["direct"]; vz = vcmd["vz"]
        else:
            rate = getattr(vel_meas, "rate", None)
            direct = getattr(vel_meas, "direct", None)
            vz = getattr(pos_meas, "z", None)
        att_times = self.attack_events.get(int(rid), [])
        attack_flag = 1 if any((sim_sec-1) < t <= sim_sec for t in att_times) else 0

        def nz(x, default):
            try: return float(x)
            except Exception: return default
        return [nz(rate, 0.0), (nz(direct, 0.0) % 360.0), nz(vz, 0.0), int(attack_flag)]

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
        if not self.rows_for_csv:
            print("[RL] No vector data to dump:", path, flush=True); return
        n_act = 16
        n_obs = len(self.rows_for_csv[0]) - 1 - n_act
        header = ["t"] + [f"obs_{i}" for i in range(n_obs)] + [f"act_{j}" for j in range(n_act)]
        abspath = os.path.abspath(path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(self.rows_for_csv)
        print(f"[RL] Dumped {len(self.rows_for_csv)} rows to {abspath}", flush=True)

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
    def _is_blue_side(self, side):
        return side in (2, "SIDE_BLUE", "BLUE")

    def _is_destroyed_flag(self, dmg):
        if isinstance(dmg, (int, float)):
            return int(dmg) == 1
        s = str(dmg)
        return ("DESTROYED" in s.upper()) or (s.strip() == "1")

    def _is_fire_success(self, uid, max_tries=5, wait_s=0.1):
        if not uid: return False
        for _ in range(max_tries):
            time.sleep(wait_s)
            try: st = self.client.get_command_status(uid)
            except Exception: st = None
            if isinstance(st, dict):
                status = st.get('status', ''); result = st.get('execute_result', '') or ''
                if status == 'EXECUTE_SUCCESS' or ('执行打击成功' in result): return True
                if status == 'EXECUTE_FAILED': return False
        return False

    def _fire_with_log(self, rid, tid):
        uid = self.client.set_target(vehicle_id=rid, target_id=tid)
        sim_t = self.client.get_sim_time()
        self.recorder.mark_action(rid, {"type": "fire", "target_id": int(tid), "uid": uid}, sim_t)
        return uid

    def _distance_m(self, lon1, lat1, lon2, lat2):
        if None in (lon1, lat1, lon2, lat2): return 1e18
        try:
            p1 = Position(x=lon1, y=lat1, z=0); p2 = Position(x=lon2, y=lat2, z=0)
            return self.client.get_distance_by_lon_lat(p1, p2)
        except Exception:
            from math import cos, radians, sqrt
            dx = (lon2-lon1) * 111320.0 * cos(radians((lat1+lat2)/2.0))
            dy = (lat2-lat1) * 110540.0
            return sqrt(dx*dx + dy*dy)

    def _get_target_pos(self, target_id, vis, all_pos):
        p = all_pos.get(target_id)
        if p is not None: return p.x, p.y
        for tracks in vis.values():
            for t in tracks:
                if t.get("target_id") == target_id:
                    return t.get("lon"), t.get("lat")
        return (None, None)

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
        if len(self.destroyed_blue) >= 4:
            self._end_simulation_once("All 4 BLUE aircraft destroyed."); return
        if len(self.destroyed_red) >= 4:
            self._end_simulation_once("All 4 RED aircraft destroyed."); return

    # ---- 主一步：先采样、再执行攻击逻辑 ----
    def step(self):
        if self._ended: return
        try: raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True); return
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
                    except Exception:
                        d = None
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

        if self._ended: return

        # —— 执行攻击逻辑（与单轮版一致）——
        try: raw_visible2 = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True); return
        vis2 = normalize_visible(raw_visible2)
        try: all_pos2 = self.client.get_vehicle_pos()
        except Exception: all_pos2 = {}
        self._update_destroyed_from_situ()
        now = self.client.get_sim_time()
        expired = [tid for tid, meta in self.target_locks.items() if now >= meta["until"]]
        for tid in expired: self.target_locks.pop(tid, None)

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
                if self.ammo.get(rid, 0) <= 0: continue
                if (now - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC: continue
                r_pos = all_pos2.get(rid)
                if r_pos is None: continue
                d = self._distance_m(r_pos.x, r_pos.y, t_lon, t_lat)
                if d < best_d: best_d, best_red = d, rid
            if best_red is not None:
                assignments.append((best_red, tid))
                used_reds_this_round.add(best_red)

        if not assignments: return

        for rid, tid in assignments:
            if tid in self.destroyed_targets: continue
            now_sim = self.client.get_sim_time()
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
                    self._end_simulation_once("Timeout")
                    break
                self.step()
                time.sleep(SCAN_INTERVAL_SEC)
        finally:
            try:
                self.recorder.dump_csv(self.out_csv_path)
            except Exception as e:
                print("[RL] dump_csv on finally failed:", e, flush=True)

# ==================== 多轮批量 Runner ====================
def safe_reset(client):
    """
    优先使用 reset/restart 之一；若都没有，再用 stop()->sleep()->start() 兜底。
    注意：只做一种操作，避免 '容器正在运行' 冲突。
    """
    # 优先原生重置：restart / reset / reset_scene / reset_scenario
    for fn_name in ["restart", "reset", "reset_scene", "reset_scenario"]:
        fn = getattr(client, fn_name, None)
        if callable(fn):
            try:
                fn()
                print(f"[Runner] client.{fn_name}() called.", flush=True)
                return True
            except Exception as e:
                # 容器正在运行 -> 说明不需要重复启动，直接当作 ok 进入下一步
                msg = str(e)
                if "容器正在运行" in msg or "already running" in msg.lower():
                    print(f"[Runner] client.{fn_name}(): container already running, continue.", flush=True)
                    return True
                print(f"[Runner] client.{fn_name}() failed: {e}", flush=True)
                # 尝试下一个候选
                continue

    # 都没有就硬重启
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
        # 若此处提示“容器正在运行”，说明其实启动着，也当成功
        msg = str(e)
        if "容器正在运行" in msg or "already running" in msg.lower():
            print("[Runner] fallback start(): container already running, continue.", flush=True)
            return True
        print("[Runner] fallback start failed:", e, flush=True)
        return False


def run_one_episode(client, plan_id, out_csv_path, max_wall_time_sec=360, min_wall_time_sec=10):
    """返回 (success, final_score)"""

    # 先确保停一下，避免上一轮残留
    try:
        client.stop()
        print("[Runner] pre-episode stop()", flush=True)
    except Exception:
        pass
    time.sleep(0.5)

    # 先启动蓝军（会阻塞，所以放线程/进程；且需在 start/restart 之前）
    blue = CombatSystem(client)
    blue_thread = threading.Thread(target=blue.run_combat_loop, daemon=True)
    blue_thread.start()
    print("[Runner] Blue loop started (before start/restart).", flush=True)

    # 重置/启动环境：优先 restart/reset；否则 stop->start
    ok = safe_reset(client)
    if not ok:
        # 再试一次硬重启
        try:
            client.stop()
        except Exception:
            pass
        time.sleep(1.0)
        try:
            client.start()
            ok = True
            print("[Runner] start() after second try.", flush=True)
        except Exception as e:
            msg = str(e)
            if "容器正在运行" in msg or "already running" in msg.lower():
                ok = True
                print("[Runner] start(): container already running, continue.", flush=True)
            else:
                print("[Runner] start() failed:", e, flush=True)

    # 如果 safe_reset 已经把容器拉起来了，这里不重复 start；否则补一次
    if ok:
        try:
            client.start()
            print("[Runner] client.start() (idempotent).", flush=True)
        except Exception as e:
            # 常见：already running -> 忽略
            msg = str(e)
            if "容器正在运行" in msg or "already running" in msg.lower():
                pass
            else:
                print("[Runner] extra start() failed (ignored):", e, flush=True)

    # 红方控制器
    red_ctrl = RedForceController(client, RED_IDS, out_csv_path)
    red_thread = threading.Thread(target=lambda: red_ctrl.run_loop(max_wall_time_sec=max_wall_time_sec), daemon=True)
    red_thread.start()
    print("[Runner] Red loop started.", flush=True)

    # 轮询等待结束
    t0 = time.time()
    final_score = None
    while True:
        time.sleep(1.0)
        try:
            s = client.get_score()
            if s:
                final_score = s
        except Exception:
            pass

        if red_ctrl._ended:
            elapsed = time.time() - t0
            if elapsed < min_wall_time_sec:
                print(f"[Runner] Episode ended too fast ({elapsed:.1f}s), mark as invalid.", flush=True)
                return False, final_score
            break

        if (time.time() - t0) > (max_wall_time_sec + 5):
            print("[Runner] Hard timeout guard fired.", flush=True)
            try:
                red_ctrl._end_simulation_once("Timeout")
            except Exception:
                pass
            break

    try:
        s2 = client.get_score()
        if s2:
            final_score = s2
    except Exception:
        pass

    return True, final_score


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
        # 场景复位
        ok = safe_reset(client)
        if not ok:
            print("[Runner] reset failed; sleep and try start()", flush=True)
            time.sleep(1.0)
            try: client.start()
            except Exception as e:
                print("[Runner] start after reset failed:", e, flush=True)

        out_csv = os.path.join(out_root, f"ep_{ep:04d}.csv")
        success, score = run_one_episode(client, PLAN_ID, out_csv, MAX_WALL_TIME_PER_EP, MIN_WALL_TIME_PER_EP)

        manifest["episodes"].append({
            "episode": ep,
            "csv": os.path.abspath(out_csv),
            "success": bool(success),
            "score": score
        })

        # 小憩一下再下一轮，避免端上资源未释放完
        time.sleep(1.0)

    # 写索引文件
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[Runner] Manifest written to:", manifest_path, flush=True)
    print("[Runner] Done. Total episodes:", EPISODES, flush=True)

if __name__ == "__main__":
    main()
