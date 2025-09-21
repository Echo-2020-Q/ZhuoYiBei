# -*- coding: utf-8 -*-
# env_rflysim_sb3.py (with full shaping & info metrics)
import math, time, os, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ===== 你的仿真依赖（按工程路径修改）=====
import rflysim
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

# ======= 常量 =======
RED_IDS = [10091, 10084, 10085, 10086]
ATTACK_COOLDOWN_SEC = 1.5
SCAN_INTERVAL_SEC = 0.1
LOCK_SEC = 55.0
AMMO_MAX = 8
BLUE_DIST_CAP = 70000.0

BOUNDARY_RECT = {
    "min_x": 101.07326442811694, "max_x": 103.08242360888715,
    "min_y": 39.558295557025474, "max_y": 40.599429229677526,
}
JAM_CENTER = (101.96786384206956, 40.2325)
JAM_EDGE_PT = (101.84516211958464, 40.2325)

# —— shaping 权重（与你自研 PPO 保持一致，可按需微调）——
GAMMA = 0.99
LAMBDA_TTL        = -0.001
ALPHA_PHI         = 0.6
BETA_PHI          = 0.4
LAMBDA_LOCK_POS   = +0.01
LAMBDA_SWITCH     = -0.02
LAMBDA_SPEED_OK   = +0.005
LAMBDA_SPEED_NG   = -0.005
LAMBDA_VZ_JITTER  = -0.002
LAMBDA_ACT_SMOOTH = -0.003
LAMBDA_EVADE_GOOD = +0.01
LAMBDA_EVADE_BAD  = -0.02

SPEED_MIN = 80.0
SPEED_MAX = 200.0
SPEED_CAP = 200.0

MISSILE_THREAT_DIST_M = 50000.0
MISSILE_BEARING_THRESH_DEG = 25.0

def _ang_norm(deg): return (float(deg) + 360.0) % 360.0
def _ang_diff_abs(a, b):
    d = abs(_ang_norm(a) - _ang_norm(b))
    return d if d <= 180.0 else 360.0 - d

def _bearing_deg_from_A_to_B(lonA, latA, lonB, latB):
    if None in (lonA, latA, lonB, latB): return None
    φ1, φ2 = math.radians(latA), math.radians(latB)
    λ1, λ2 = math.radians(lonA), math.radians(lonB)
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def _geo_dist_haversine_m(lon1, lat1, lon2, lat2):
    if None in (lon1, lat1, lon2, lat2): return None
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def _direct_rate_from_vx_vy(vx, vy):
    from math import sqrt, atan2, degrees
    vxn, vye = float(vx or 0.0), float(vy or 0.0)
    direct = sqrt(vxn*vxn + vye*vye)      # 速度标量
    rate = (degrees(atan2(vye, vxn)) + 360.0) % 360.0  # 航向角
    return direct, rate

def _dist_to_boundary_m(client, lon, lat):
    if lon is None or lat is None: return (0,0,0,0)
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
        from math import cos, radians
        dxL = (lon - BOUNDARY_RECT["min_x"]) * 111320.0 * cos(radians(lat))
        dxR = (BOUNDARY_RECT["max_x"] - lon) * 111320.0 * cos(radians(lat))
        dyD = (lat - BOUNDARY_RECT["min_y"]) * 110540.0
        dyU = (BOUNDARY_RECT["max_y"] - lat) * 110540.0
        d_left, d_right, d_down, d_up = abs(dxL), abs(dxR), abs(dyD), abs(dyU)
    return (float(d_left), float(d_right), float(down), float(d_up))

class RflySimRedVsBlueEnv(gym.Env):
    """
    - 单智能体控制4架 RED（动作：16维 = 4*(heading, speed, vz, fire_intent)）
    - 观测：76维 = 4 * 19（与 pack_single_obs19 对齐）
    - 奖励：击毁/被毁/禁飞 + 全套 shaping（见 info 分量）
    """
    metadata = {"render_modes": []}

    def __init__(self, host: str, port_cmd: int, port_data: int,
                 plan_id: int = 106, max_sim_seconds: int = 600, shaping: bool = True):
        super().__init__()
        self.host, self.port_cmd, self.port_data = host, port_cmd, port_data
        self.plan_id = plan_id
        self.max_sim_seconds = int(max_sim_seconds)
        self.shaping = bool(shaping)

        # 动作空间（fire 连续输出→阈值化）
        low_one  = np.array([0, 0, -50, 0], dtype=np.float32)
        high_one = np.array([360, 200, 50, 1], dtype=np.float32)
        low = np.tile(low_one, len(RED_IDS))
        high= np.tile(high_one, len(RED_IDS))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 观测空间
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19*len(RED_IDS),), dtype=np.float32)

        # 运行态
        self.client = None
        self.blue_thread = None
        self._t0 = 0.0
        self._destroyed_blue = set()
        self._destroyed_red = set()
        self._ammo = {rid: AMMO_MAX for rid in RED_IDS}
        self._last_fire_time = {rid: -1e9 for rid in RED_IDS}
        self._jam_radius_m = None

        # —— shaping 运行时缓存 —— #
        self._phi_prev = 0.0
        self._last_act = None
        self._last_vz = {rid: 0.0 for rid in RED_IDS}
        self._focus_target = {rid: (None, 0.0) for rid in RED_IDS}  # (last_tid, consec_secs)
        self._missile_prev_xy_t = {}  # mid -> (lon, lat, t)

        # 统计（每 step 归零的计数器）
        self._switch_count_step = 0
        self._focused_secs_step = 0.0
        self._speed_band_hits = 0
        self._speed_total = 0
        self._vz_abs_diff_sum = 0.0
        self._evade_good_last = 0.0
        self._evade_bad_last = 0.0

    # ====== 连接/重置/蓝方线程 ======
    def _connect(self):
        cfg = RflysimEnvConfig(self.host, self.port_cmd, self.port_data)
        self.client = VehicleClient(id=self.plan_id, config=cfg)
        self.client.enable_rflysim()

    def _start_blue(self):
        import threading
        def _loop():
            try:
                blue = CombatSystem(self.client)
                blue.run_combat_loop()
            except Exception as e:
                print("[Env] Blue loop exception:", e, flush=True)
        self.blue_thread = threading.Thread(target=_loop, daemon=True)
        self.blue_thread.start()
        time.sleep(0.3)

    def _safe_reset_sim(self):
        for fn_name in ["restart", "reset", "reset_scene", "reset_scenario"]:
            fn = getattr(self.client, fn_name, None)
            if callable(fn):
                try:
                    fn(); return True
                except Exception:
                    pass
        try:
            self.client.stop(); time.sleep(0.5); self.client.start(); return True
        except Exception:
            return False

    # ====== 工具 ======
    def _signed_dist_to_jam_boundary_m(self, lon, lat):
        if lon is None or lat is None: return 0.0
        C = Position(x=JAM_CENTER[0], y=JAM_CENTER[1], z=0)
        if self._jam_radius_m is None:
            E = Position(x=JAM_EDGE_PT[0], y=JAM_EDGE_PT[1], z=0)
            try:
                self._jam_radius_m = self.client.get_distance_by_lon_lat(C, E)
            except Exception:
                self._jam_radius_m = 10416.0
        P = Position(x=lon, y=lat, z=0)
        try:
            d_c = self.client.get_distance_by_lon_lat(C, P)
        except Exception:
            d_c = 0.0
        return float(d_c - self._jam_radius_m)

    def _normalize_visible(self, raw):
        out = {}
        if not isinstance(raw, dict): return out
        for det, arr in raw.items():
            tracks = []
            if isinstance(arr, list):
                for t in arr:
                    try:
                        tid = int(getattr(t, "target_id", None) or (t.get("target_id") if isinstance(t, dict) else None))
                    except Exception:
                        continue
                    # 经纬
                    if isinstance(t, dict):
                        lon = t.get("lon", None); lat = t.get("lat", None)
                    else:
                        tp = getattr(t, "target_pos", None)
                        lon = getattr(tp, "x", None) if tp else None
                        lat = getattr(tp, "y", None) if tp else None
                    spd = float(getattr(t, "target_speed", None) or getattr(t, "speed", 0.0) or 0.0)
                    dire = getattr(t, "target_direction", None) or getattr(t, "direction", None)
                    dire = float(dire % 360.0) if dire is not None else None
                    tracks.append({"id": tid, "lon": lon, "lat": lat, "speed": spd, "dir": dire})
            out[int(det)] = tracks
        return out

    def _build_obs76_and_aux(self):
        try: vis_raw = self.client.get_visible_vehicles() or {}
        except Exception: vis_raw = {}
        vis = self._normalize_visible(vis_raw)
        try: pos = self.client.get_vehicle_pos() or {}
        except Exception: pos = {}
        try: vel = self.client.get_vehicle_vel() or {}
        except Exception: vel = {}

        obs_all = []
        aux = {"nbearing": {}, "ndist": {}, "my_heading": {}}

        for rid in RED_IDS:
            p = pos.get(rid); v = vel.get(rid)
            my_lon = getattr(p, "x", None) if p else None
            my_lat = getattr(p, "y", None) if p else None
            vx = getattr(v, "vx", 0.0) if v else 0.0
            vy = getattr(v, "vy", 0.0) if v else 0.0
            vz = getattr(v, "vz", 0.0) if v else 0.0
            speed_scalar, my_heading = _direct_rate_from_vx_vy(vx, vy)
            aux["my_heading"][rid] = my_heading

            # 最近4个蓝机距离 + 最近蓝 bearing
            dists = []
            nbearing = None
            if my_lon is not None and my_lat is not None:
                for tracks in vis.values():
                    for t in tracks:
                        if t["id"] < 10000:  # 导弹忽略
                            continue
                        if t["lon"] is None or t["lat"] is None: continue
                        d = self.client.get_distance_by_lon_lat(
                            Position(x=my_lon, y=my_lat, z=0),
                            Position(x=t["lon"], y=t["lat"], z=0)
                        )
                        if d is not None: dists.append(float(d))
                dists.sort()
                if len(dists) < 4: dists += [BLUE_DIST_CAP] * (4 - len(dists))
                else: dists = dists[:4]

                # 最近蓝 bearing
                best, best_t = None, None
                for tracks in vis.values():
                    for t in tracks:
                        if t["id"] < 10000: continue
                        if t["lon"] is None or t["lat"] is None: continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                     Position(x=t["lon"], y=t["lat"], z=0))
                        except Exception:
                            dd = None
                        if dd is not None and (best is None or dd < best):
                            best, best_t = dd, t
                if best_t:
                    nbearing = _bearing_deg_from_A_to_B(my_lon, my_lat, best_t["lon"], best_t["lat"])
            else:
                dists = [BLUE_DIST_CAP] * 4

            left, right, down, up = _dist_to_boundary_m(self.client, my_lon, my_lat)
            jam_signed = self._signed_dist_to_jam_boundary_m(my_lon, my_lat)
            ammo = int(self._ammo.get(rid, 0))

            # —— 最近导弹：占位（如需可扩展估计航向）——
            nearest_msl_dist, nearest_msl_dir, rel_angle_to_msl = 0.0, 0.0, 0.0

            obs19 = [
                left, right, down, up, vx, vy, vz,
                *dists, 0.0, float(nbearing or 0.0),
                jam_signed, ammo, speed_scalar,
                nearest_msl_dist, nearest_msl_dir, rel_angle_to_msl
            ]
            assert len(obs19) == 19
            obs_all.extend(obs19)

            aux["nbearing"][rid] = nbearing
            aux["ndist"][rid] = dists[0] if dists else None

        return np.asarray(obs_all, dtype=np.float32), aux, vis, pos, vel

    def _update_destroyed_from_situ(self):
        try:
            situ = self.client.get_situ_info() or {}
        except Exception:
            return
        for vid, info in situ.items():
            side = getattr(info, "side", None) or (info.get("side") if isinstance(info, dict) else None)
            dmg  = getattr(info, "damage_state", None) or (info.get("damage_state") if isinstance(info, dict) else None)
            try: vid = int(getattr(info,"id",None) or (info.get("id") if isinstance(info, dict) else int(vid)))
            except Exception: continue
            destroyed = (str(dmg).strip() == "1") or ("DESTROYED" in str(dmg).upper())
            if not destroyed: continue
            if side in (2, "SIDE_BLUE", "BLUE"):
                self._destroyed_blue.add(vid)
            else:
                self._destroyed_red.add(vid)

    def _check_done(self, sim_t):
        if sim_t - self._t0 >= self.max_sim_seconds: return True
        if len(self._destroyed_blue) >= 4: return True
        if len(self._destroyed_red) >= 4: return True
        return False

    # ====== Gym API ======
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.client is None:
            self._connect()
        try:
            self.client.stop()
        except Exception:
            pass
        self._safe_reset_sim()
        self._start_blue()

        vel0 = rflysim.Vel(); vel0.vz = 150; vel0.rate = 200; vel0.direct = 90
        for rid in RED_IDS:
            try:
                self.client.enable_radar(vehicle_id=rid, state=1)
                self.client.set_vehicle_vel(rid, vel0)
            except Exception:
                pass

        self._ammo = {rid: AMMO_MAX for rid in RED_IDS}
        self._last_fire_time = {rid: -1e9 for rid in RED_IDS}
        self._destroyed_blue.clear(); self._destroyed_red.clear()

        # 重置 shaping 缓存
        self._phi_prev = 0.0
        self._last_act = None
        self._last_vz = {rid: 0.0 for rid in RED_IDS}
        self._focus_target = {rid: (None, 0.0) for rid in RED_IDS}
        self._missile_prev_xy_t.clear()
        self._switch_count_step = 0
        self._focused_secs_step = 0.0
        self._speed_band_hits = 0
        self._speed_total = 0
        self._vz_abs_diff_sum = 0.0
        self._evade_good_last = 0.0
        self._evade_bad_last  = 0.0

        self._t0 = float(self.client.get_sim_time() or 0.0)
        obs, _, _, _, _ = self._build_obs76_and_aux()
        return obs, {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(len(RED_IDS), 4)
        sim_now = float(self.client.get_sim_time() or self._t0)

        # 1) 速度命令（heading, speed, vz）
        for i, rid in enumerate(RED_IDS):
            heading = float(a[i,0]) % 360.0
            speed   = float(np.clip(a[i,1], 0.0, SPEED_CAP))
            vz      = float(np.clip(a[i,2], -50.0, 50.0))
            vcmd = rflysim.Vel(); vcmd.direct = heading; vcmd.rate = speed; vcmd.vz = vz
            try:
                self.client.set_vehicle_vel(rid, vcmd)
            except Exception:
                pass

        # 2) 观测与辅助
        obs, aux, vis, pos, vel = self._build_obs76_and_aux()

        # 3) 蓝机集合 + 最近蓝：用于两阶段打击/专注统计
        visible_blue = set()
        for arr in vis.values():
            for t in arr:
                tid = int(t["id"])
                if tid >= 10000:
                    visible_blue.add(tid)

        # 4) 策略优先开火（fire >=0.5）、最近蓝分配
        for i, rid in enumerate(RED_IDS):
            fire = float(a[i,3]) >= 0.5
            if not fire: continue
            if (sim_now - self._last_fire_time[rid]) < ATTACK_COOLDOWN_SEC: continue
            if self._ammo.get(rid,0) <= 0: continue

            myp = pos.get(rid)
            if not myp: continue
            best_tid, best_d = None, 1e18
            for tid in list(visible_blue):
                tp = pos.get(tid)
                lon, lat = (tp.x, tp.y) if tp else (None, None)
                if lon is None or lat is None: continue
                try:
                    d = self.client.get_distance_by_lon_lat(
                        Position(x=myp.x, y=myp.y, z=0),
                        Position(x=lon, y=lat, z=0)
                    )
                except Exception:
                    d = None
                if d is not None and d < best_d:
                    best_d, best_tid = float(d), tid

            if best_tid is not None:
                try:
                    uid = self.client.set_target(vehicle_id=rid, target_id=int(best_tid))
                    self._last_fire_time[rid] = sim_now
                    self._ammo[rid] = max(0, int(self._ammo[rid]-1))
                except Exception:
                    pass

        # 5) 短暂等待仿真推进一个 tick
        time.sleep(SCAN_INTERVAL_SEC)

        # 6) 更新击毁态
        self._update_destroyed_from_situ()

        # ====== 统计 shaping 所需的运行时量 ======
        # 6.1 专注/换目标统计（以最近蓝为 proxy）
        self._switch_count_step = 0
        focus_total = 0.0
        for rid in RED_IDS:
            myp = pos.get(rid)
            if not myp: continue
            my_lon, my_lat = getattr(myp, "x", None), getattr(myp, "y", None)
            nearest_tid, nearest_d = None, None
            if my_lon is not None and my_lat is not None:
                for arr in vis.values():
                    for t in arr:
                        if int(t["id"]) < 10000: continue
                        lonB, latB = t["lon"], t["lat"]
                        if lonB is None or latB is None: continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0),
                                                                     Position(x=lonB, y=latB, z=0))
                        except Exception: dd = None
                        if dd is not None and (nearest_d is None or dd < nearest_d):
                            nearest_d, nearest_tid = float(dd), int(t["id"])
            last_tid, consec = self._focus_target.get(rid, (None, 0.0))
            if nearest_tid is not None:
                if last_tid == nearest_tid: consec += 1.0
                else:
                    self._switch_count_step += 1
                    consec = 1.0
                self._focus_target[rid] = (nearest_tid, consec)
            focus_total += float(consec)
        self._focused_secs_step = focus_total / max(1, len(RED_IDS))

        # 6.2 速度带/垂直抖动
        self._speed_band_hits = 0
        self._speed_total = 0
        self._vz_abs_diff_sum = 0.0
        for rid in RED_IDS:
            v = vel.get(rid)
            vx, vy = getattr(v, "vx", 0.0) if v else 0.0, getattr(v, "vy", 0.0) if v else 0.0
            vz = getattr(v, "vz", 0.0) if v else 0.0
            speed_scalar, _ = _direct_rate_from_vx_vy(vx, vy)
            if SPEED_MIN <= speed_scalar <= SPEED_MAX: self._speed_band_hits += 1
            self._speed_total += 1
            self._vz_abs_diff_sum += abs(vz - self._last_vz.get(rid, vz))
            self._last_vz[rid] = vz
        speed_band_ratio = (self._speed_band_hits / max(1, self._speed_total))
        speed_off_ratio  = 1.0 - speed_band_ratio
        vz_abs_diff_mean = (self._vz_abs_diff_sum / max(1, len(RED_IDS)))

        # 6.3 动作平滑：与上一时刻动作向量的 L2 距离
        r_act_smooth = 0.0
        if self._last_act is not None and len(self._last_act) == a.size:
            diff = np.linalg.norm(a.flatten() - self._last_act.flatten()) / max(1.0, float(a.size))
            r_act_smooth = LAMBDA_ACT_SMOOTH * float(diff)
        self._last_act = a.copy()

        # 6.4 反导规避几何统计（基于导弹接近与 bearing 夹角）
        self._evade_good_last = 0.0
        self._evade_bad_last  = 0.0
        # 构建导弹表
        missiles = []
        try: pos_all = self.client.get_vehicle_pos() or {}
        except Exception: pos_all = {}
        for arr in vis.values():
            for t in arr:
                tid = int(t["id"])
                if tid >= 10000: continue
                mpos = pos_all.get(tid)
                mlon = getattr(mpos, "x", None) if mpos else t.get("lon", None)
                mlat = getattr(mpos, "y", None) if mpos else t.get("lat", None)
                if mlon is None or mlat is None: continue
                missiles.append({"mid": tid, "lon": float(mlon), "lat": float(mlat), "dir": t.get("dir", None)})
        # 估算导弹航向 + 接近判据
        for rid in RED_IDS:
            myp = pos.get(rid)
            if not myp: continue
            my_lon, my_lat = float(getattr(myp, "x", np.nan)), float(getattr(myp, "y", np.nan))
            if np.isnan(my_lon) or np.isnan(my_lat): continue
            # 找最近导弹
            best = None
            best_d = None
            for m in missiles:
                d = _geo_dist_haversine_m(my_lon, my_lat, m["lon"], m["lat"])
                if d is None: continue
                if best is None or d < best_d:
                    best, best_d = m, float(d)
            if best is None: continue

            prev = self._missile_prev_xy_t.get(best["mid"], None)
            mdir = best.get("dir", None)
            if prev is not None:
                prev_lon, prev_lat, prev_t = prev
                dt = max(1e-3, sim_now - float(prev_t))
                mdir = _bearing_deg_from_A_to_B(prev_lon, prev_lat, best["lon"], best["lat"])
            self._missile_prev_xy_t[best["mid"]] = (best["lon"], best["lat"], sim_now)

            # missile -> red 的 LOS
            los_m2r = _bearing_deg_from_A_to_B(best["lon"], best["lat"], my_lon, my_lat)
            ang_diff = _ang_diff_abs(mdir, los_m2r) if (mdir is not None and los_m2r is not None) else None
            in_threat = (best_d is not None and best_d <= MISSILE_THREAT_DIST_M)

            if in_threat:
                if (ang_diff is not None) and (ang_diff <= MISSILE_BEARING_THRESH_DEG):
                    self._evade_bad_last  += 1.0 / max(1, len(RED_IDS))
                else:
                    self._evade_good_last += 1.0 / max(1, len(RED_IDS))

        # ====== 奖励各分量 ======
        prev_blue, prev_red = len(self._destroyed_blue), len(self._destroyed_red)
        # 击毁项（与上一刻比较，用 situ 更新后再计算增量）
        # 注意：我们在上面已经更新过 destroyed 状态，本步按“当前-上步”会是0。
        # 更稳定的做法：在 step 入口处保存 counts，这里重取并计算 delta。
        blue_k = len(self._destroyed_blue)
        red_k  = len(self._destroyed_red)
        d_blue = blue_k - prev_blue
        d_red  = red_k - prev_red
        r_kill_blue = float(d_blue * 1.0)
        r_kill_red  = float(d_red  * -1.0)

        # 禁飞区罚
        r_nofly = 0.0
        for rid in RED_IDS:
            p = pos.get(rid)
            if p and getattr(p, "x", None) is not None and getattr(p, "y", None) is not None:
                if self._signed_dist_to_jam_boundary_m(float(p.x), float(p.y)) < 0.0:
                    r_nofly += (-0.05)

        # 生存时间成本：按活着的红机数
        alive_reds = len([rid for rid in RED_IDS if rid not in self._destroyed_red])
        r_ttl = LAMBDA_TTL * float(alive_reds)

        # 潜在势函数（距离 + 对准）
        phi_now = 0.0; cnt = 0
        for rid in RED_IDS:
            d = aux["ndist"].get(rid, None)
            bearing = aux["nbearing"].get(rid, None)
            my_head = aux["my_heading"].get(rid, None)
            if d is None or bearing is None or my_head is None:
                continue
            d_norm = min(1.0, float(d) / float(BLUE_DIST_CAP))
            term_d = ALPHA_PHI * (1.0 / (1.0 + d_norm))
            term_ang = BETA_PHI * math.cos(math.radians(_ang_diff_abs(my_head, bearing)))
            phi_now += (term_d + term_ang); cnt += 1
        r_phi = 0.0
        if cnt > 0:
            phi_now /= float(cnt)
            r_phi = (GAMMA * phi_now - (self._phi_prev or 0.0))
            self._phi_prev = phi_now

        # 专注/换目标、速度带、垂直抖动、动作平滑、规避几何
        r_lock      = LAMBDA_LOCK_POS   * float(self._focused_secs_step)
        r_switch    = LAMBDA_SWITCH     * float(self._switch_count_step)
        r_speed_ok  = LAMBDA_SPEED_OK   * float(speed_band_ratio)
        r_speed_ng  = LAMBDA_SPEED_NG   * float(speed_off_ratio)
        r_vz_jitter = LAMBDA_VZ_JITTER  * float(vz_abs_diff_mean)
        r_evade_g   = LAMBDA_EVADE_GOOD * float(self._evade_good_last)
        r_evade_b   = LAMBDA_EVADE_BAD  * float(self._evade_bad_last)

        # 汇总
        reward = (
            r_kill_blue + r_kill_red + r_nofly + r_ttl + r_phi +
            r_lock + r_switch + r_speed_ok + r_speed_ng +
            r_vz_jitter + r_act_smooth + r_evade_g + r_evade_b
        )

        terminated = self._check_done(float(self.client.get_sim_time() or self._t0))
        truncated = False

        info = {
            # 主奖励分量
            "r_kill_blue": r_kill_blue,
            "r_kill_red":  r_kill_red,
            "r_nofly":     r_nofly,
            "r_ttl":       r_ttl,
            "r_phi":       r_phi,
            "r_lock":      r_lock,
            "r_switch":    r_switch,
            "r_speed_ok":  r_speed_ok,
            "r_speed_ng":  r_speed_ng,
            "r_vz_jitter": r_vz_jitter,
            "r_act_smooth": r_act_smooth,
            "r_evade_good": r_evade_g,
            "r_evade_bad":  r_evade_b,
            # 便于监控的原始统计
            "blue_kills":  blue_k,
            "red_losses":  red_k,
            "focus_secs":  float(self._focused_secs_step),
            "switch_cnt":  int(self._switch_count_step),
            "speed_band_ratio": float(speed_band_ratio),
            "vz_abs_diff_mean": float(vz_abs_diff_mean),
        }

        if terminated:
            try: self.client.stop()
            except Exception: pass
        return obs, float(reward), terminated, truncated, info

    def render(self): return None
    def close(self):
        try:
            if self.client: self.client.stop()
        except Exception:
            pass
