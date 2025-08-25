# 网络与方案初始化配置
import rflysim
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
import threading
import time, re, threading

DESTROYED_FLAG = "DAMAGE_STATE_DESTROYED"
RED_IDS = [10091, 10084, 10085, 10086]
AMMO_MAX = 2
ATTACK_COOLDOWN_SEC = 20.0   # 同一红机连续下发攻击命令的最短间隔（避免刷命令）
SCAN_INTERVAL_SEC = 0.5     # 侦察扫描周期
STATUS_POLL_MAX_TRIES = 10         # 轮询最多次数
STATUS_POLL_INTERVAL = 0.25        # 每次轮询间隔(s)
STATUS_POLL_BACKOFF = 1.5          # 退避倍率（可选）

vel_0 = rflysim.Vel()
vel_0.vz =150#飞行目标高度

vel_0.rate = 100 # 速率

vel_0.direct = 90 # 速度在地图内的绝对方向，正北为0，0-360表示完整一周的方向

def _as_dict(obj):
    """把 track/pos 统一成 dict 读取（兼容属性对象 / dict）"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # 对象：尝试把常见字段取出来
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

def _parse_track_from_string(s):
    """
    兜底：当 SDK 把 track 打印成字符串时，从中提取 target_id / lon=x / lat=y
    返回 (target_id, lon, lat) 或 None
    """
    if not isinstance(s, str):
        return None
    # target_id 优先
    m_id = re.search(r'\btarget_id\s*:\s*(\d+)', s)
    if not m_id:
        # 有些日志里只写 id: 12345
        m_id = re.search(r'\bid\s*:\s*(\d+)', s)
    if not m_id:
        return None
    tid = int(m_id.group(1))

    # 位置可能以 "target pos {x: ..., y: ...}" 或 "target_pos x: ..., y: ..." 出现
    # 经度 -> x，纬度 -> y
    m_xy = re.search(r'target\s*_?pos.*?\{[^}]*?x\s*:\s*([-\d\.eE]+)\s*[,， ]+\s*y\s*:\s*([-\d\.eE]+)', s)
    if not m_xy:
        m_xy = re.search(r'\bx\s*:\s*([-\d\.eE]+)\s*[，, ]+\s*y\s*:\s*([-\d\.eE]+)', s)
    lon = float(m_xy.group(1)) if m_xy else None
    lat = float(m_xy.group(2)) if m_xy else None
    return (tid, lon, lat)

def normalize_visible(visible):
    """
    把 get_visible_vehicles() 的结果规格化为：
    { red_id: [ {"target_id": int, "lon": float|None, "lat": float|None}, ... ], ... }
    兼容 value 为 list / dict / 单个对象 / 字符串 的不同返回形式。
    只保留能拿到 target_id 的条目。
    """
    out = {}
    if not isinstance(visible, dict):
        return out

    for detector_id, v in visible.items():
        try:
            detector_id = int(detector_id)
        except Exception:
            # 某些实现 key 本身就是 int；如果不是，跳过
            pass

        tracks = []
        if v is None:
            pass
        elif isinstance(v, list):
            # list 里可能是 对象 / dict / 字符串
            for t in v:
                if isinstance(t, (dict,)):
                    td = _as_dict(t)
                    tid = td.get('target_id') or td.get('id')
                    if tid is None:
                        continue
                    tid = int(tid)
                    pos = td.get('target_pos') or td.get('target pos') or {}
                    posd = _as_dict(pos)
                    lon = posd.get('x')
                    lat = posd.get('y')
                    tracks.append({"target_id": tid, "lon": lon, "lat": lat})
                elif hasattr(t, '__dict__'):
                    td = _as_dict(t)
                    tid = td.get('target_id') or td.get('id')
                    if tid is None:
                        continue
                    tid = int(tid)
                    pos = td.get('target_pos') or td.get('target pos') or {}
                    posd = _as_dict(pos)
                    lon = posd.get('x')
                    lat = posd.get('y')
                    tracks.append({"target_id": tid, "lon": lon, "lat": lat})
                else:
                    parsed = _parse_track_from_string(t)
                    if parsed:
                        tid, lon, lat = parsed
                        tracks.append({"target_id": tid, "lon": lon, "lat": lat})

        elif isinstance(v, dict):
            # 可能是 {target_id: track} 或 单个 track 的 dict
            if 'target_id' in v or 'id' in v:
                td = _as_dict(v)
                tid = td.get('target_id') or td.get('id')
                if tid is not None:
                    tid = int(tid)
                    pos = td.get('target_pos') or td.get('target pos') or {}
                    posd = _as_dict(pos)
                    lon = posd.get('x')
                    lat = posd.get('y')
                    tracks.append({"target_id": tid, "lon": lon, "lat": lat})
            else:
                # 当成 {tid: track}
                for _, t in v.items():
                    td = _as_dict(t)
                    tid = td.get('target_id') or td.get('id')
                    if tid is None:
                        continue
                    tid = int(tid)
                    pos = td.get('target_pos') or td.get('target pos') or {}
                    posd = _as_dict(pos)
                    lon = posd.get('x')
                    lat = posd.get('y')
                    tracks.append({"target_id": tid, "lon": lon, "lat": lat})

        else:
            # 字符串（整段 dump）
            parsed = _parse_track_from_string(v)
            if parsed:
                tid, lon, lat = parsed
                tracks.append({"target_id": tid, "lon": lon, "lat": lat})

        # 只输出我们关心的红方机（detector）键
        out[int(detector_id)] = tracks
    return out

class RedForceController:
    def __init__(self, client, red_ids):
        self.client = client
        self.red_ids = set(int(x) for x in red_ids)
        self.ammo = {int(r): AMMO_MAX for r in red_ids}
        self.last_fire_time = {int(r): 0.0 for r in red_ids}
        # 目标与并发控制
        self.assigned_target = {}  # {red_id: target_id} 当前锁定
        self.target_in_progress = set()  # 正在被某红机处理的目标（防多机同时打同一目标）
        self.destroyed_targets = set()  # 已确认损毁的目标（来自 get_situ_info）
        self.lock = threading.Lock()

        # 开雷达 + 设速度（不做 get_command_status 轮询）
        for rid in red_ids:
            try:
                uid = self.client.enable_radar(vehicle_id=rid, state=1)
                print(f"[Red] Radar ON for {rid}, uid={uid}", flush=True)
            except Exception as e:
                print(f"[Red] enable_radar({rid}) failed: {e}", flush=True)
            try:
                vuid = self.client.set_vehicle_vel(rid, vel_0)  # mode=0：航向+速度+高度
                print(f"[Red] vel set for {rid}, uid={vuid}", flush=True)
            except Exception as e:
                print(f"[Red] set_vehicle_vel({rid}) failed: {e}", flush=True)

        self.target_in_progress = set()  # 正在被某红机处理的目标，防多机浪费
        self.destroyed_targets = set()  # 已确认损毁的目标
        self.lock = threading.Lock()  # 线程锁，保护上面两个集合


        # ---------- 工具 ---------- #

    def _update_destroyed_from_situ(self):
        """调用 get_situ_info()，把损毁目标加入 destroyed_targets，并释放占用。"""
        try:
            situ_raw = self.client.get_situ_info()
        except Exception:
            return
        situ = self._normalize_situ(situ_raw)

        to_mark_destroyed = []
        for vid, info in situ.items():
            if info and info.get("damage_state") == DESTROYED_FLAG:
                to_mark_destroyed.append(vid)

        if not to_mark_destroyed:
            return

        with self.lock:
            for vid in to_mark_destroyed:
                self.destroyed_targets.add(vid)
                self.target_in_progress.discard(vid)
                # 如果是我方红机损毁，也可以直接清零其弹药，避免后续逻辑再使用
                if vid in self.ammo:
                    self.ammo[vid] = 0

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

    def _normalize_situ(self, situ):
        """
        situ -> { id(int): {"side": str|None, "damage_state": str|None} }
        """
        out = {}
        if not isinstance(situ, dict):
            return out
        for k, v in situ.items():
            try:
                key_id_guess = int(k)
            except Exception:
                key_id_guess = None
            d = _as_dict(v)
            vid = d.get("id", key_id_guess)
            try:
                vid = int(vid)
            except Exception:
                continue
            side = d.get("side")
            damage = d.get("damage_state")
            out[vid] = {"side": side, "damage_state": damage}
        return out



    def _choose_nearest_target(self, red_id, tracks, all_pos):
        if not tracks:
            return None
        my_pos = all_pos.get(red_id)
        if my_pos is None:
            return tracks[0]["target_id"]
        my_lon, my_lat = my_pos.x, my_pos.y
        best_tid, best_d = None, 1e18
        for t in tracks:
            tid = t["target_id"]
            if tid in self.destroyed_targets:
                continue
            lon, lat = t["lon"], t["lat"]
            d = self._distance_m(my_lon, my_lat, lon, lat)
            if d < best_d:
                best_d, best_tid = d, tid
        return best_tid
    # ---------- 主循环一步 ---------- #
    def step(self):
        # 1) 感知
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True)
            return

        vis = normalize_visible(raw_visible)
        print("可见的目标：", vis)
        try:
            all_pos = self.client.get_vehicle_pos()  # {id: Position}
        except Exception:
            all_pos = {}

        # 用态势信息刷新损毁集合（唯一来源）
        self._update_destroyed_from_situ()

        now = time.time()

        # 2) 决策 + 攻击（逐机）
        for red_id, tracks in vis.items():
            if red_id not in self.red_ids:
                continue
            if self.ammo.get(red_id, 0) <= 0:
                continue
            if not tracks:
                continue

            # 构造候选（距我升序），跳过已损毁
            candidates = []
            my_pos = all_pos.get(red_id)
            if my_pos is not None:
                my_lon, my_lat = my_pos.x, my_pos.y
                for t in tracks:
                    tid = t.get("target_id")
                    if tid is None or tid in self.destroyed_targets:
                        continue
                    d = self._distance_m(my_lon, my_lat, t.get("lon"), t.get("lat"))
                    candidates.append((d, tid))
            else:
                # 退化：用出现顺序代替距离
                for idx, t in enumerate(tracks):
                    tid = t.get("target_id")
                    if tid is None or tid in self.destroyed_targets:
                        continue
                    candidates.append((idx, tid))

            if not candidates:
                continue

            # 已锁定目标优先（仍可见且未损毁）
            keep_tid = self.assigned_target.get(red_id)
            still_visible_tids = {t.get("target_id") for t in tracks if t.get("target_id") is not None}
            if keep_tid in still_visible_tids and keep_tid not in self.destroyed_targets:
                candidates = [(-1.0, keep_tid)] + [(d, tid) for d, tid in candidates if tid != keep_tid]

            # 冷却限制
            if (now - self.last_fire_time.get(red_id, 0.0)) < ATTACK_COOLDOWN_SEC:
                continue

            # 选第一个“未被其他红机占用”的目标（最近优先 -> 若最近被占则打第二近…）
            target_id = None
            with self.lock:
                for _, tid in sorted(candidates, key=lambda x: x[0]):
                    if tid not in self.target_in_progress and tid not in self.destroyed_targets:
                        target_id = tid
                        self.target_in_progress.add(tid)  # 立即占用（防并发撞车）
                        break
            if target_id is None:
                continue

            # 记录锁定
            self.assigned_target[red_id] = target_id

            # 3) 发射（不做命令状态轮询）
            try:
                uid = self.client.set_target(vehicle_id=red_id, target_id=target_id)
                print(f"[Red] {red_id} -> fire at {target_id}, uid={uid}", flush=True)
                self.ammo[red_id] -= 1
                self.last_fire_time[red_id] = time.time()
            except Exception as e:
                print(f"[Red] set_target({red_id},{target_id}) failed: {e}", flush=True)
                with self.lock:
                    self.target_in_progress.discard(target_id)  # 失败释放占用
                continue
            # 不再开线程轮询；由 _update_destroyed_from_situ() 在后续扫描中判损毁

    # ---------- 循环 ---------- #
    def run_loop(self, stop_when_paused=False):
        while True:
            try:
                if stop_when_paused and self.client.is_pause():
                    break
            except Exception:
                pass
            self.step()
            time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    config = {
        "id": 106,                        # 填入网页中参赛方案对应的方案号
        "config": RflysimEnvConfig("172.23.53.35", 16001, 18001)
    }
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    # 1) 蓝方放到后台线程（不要阻塞主线程）
    combat_system = CombatSystem(client)
    blue_thread = threading.Thread(target=combat_system.run_combat_loop, daemon=True)
    blue_thread.start()
    print("[Main] Blue combat loop started.", flush=True)

    # 2) 仿真控制
    #client.set_multiple(10)
    client.start()

    # 3) 红方控制：构造时会 enable_radar 并打印“Radar ON...”
    red_ctrl = RedForceController(client, RED_IDS)
    red_thread = threading.Thread(target=red_ctrl.run_loop, daemon=True)
    red_thread.start()
    print("[Main] Red control loop started.", flush=True)

    # 4) 主线程做点轻量工作，避免退出
    try:
        while True:
            time.sleep(2.0)
            # 周期性看分数
            score = client.get_score()
            if score is not None:
                print("[Score]", score, flush=True)

            # 可选：态势信息（量大时注释掉避免刷屏）
            # situ = client.get_situ_info()
            # print("[Situ]", situ, flush=True)
    except KeyboardInterrupt:
        print("[Main] Interrupted, exiting...", flush=True)