# 网络与方案初始化配置
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
import threading
import time, re, threading

RED_IDS = [10091, 10084, 10085, 10086]
AMMO_MAX = 8
ATTACK_COOLDOWN_SEC = 3.0   # 同一红机连续下发攻击命令的最短间隔（避免刷命令）
SCAN_INTERVAL_SEC = 0.5     # 侦察扫描周期

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
        # 各红机当前锁定的目标，避免多次切换同一目标
        self.assigned_target = {}

        # 开启红方雷达
        for rid in red_ids:
            try:
                uid = self.client.enable_radar(vehicle_id=rid, state=1)
                print(f"[Red] Radar ON for {rid}, uuid={uid}")
            except Exception as e:
                print(f"[Red] enable_radar({rid}) failed: {e}")

    def _distance_m(self, lon1, lat1, lon2, lat2):
        """用官方静态方法计算两经纬点距离（米）；若无法获取，返回很大值"""
        if None in (lon1, lat1, lon2, lat2):
            return 1e18
        try:
            p1 = Position(x=lon1, y=lat1, z=0)
            p2 = Position(x=lon2, y=lat2, z=0)
            return self.client.get_distance_by_lon_lat(p1, p2)
        except Exception:
            # 兜底：粗略球面近似（不建议长期用）
            from math import cos, radians, sqrt
            dx = (lon2 - lon1) * 111320.0 * cos(radians((lat1 + lat2)/2.0))
            dy = (lat2 - lat1) * 110540.0
            return sqrt(dx*dx + dy*dy)

    def _choose_nearest_target(self, red_id, tracks, all_pos):
        """
        给定某红机的可见 tracks 列表（已经标准化），结合当前所有载具位置，从中选最近的目标
        返回 target_id 或 None
        """
        if not tracks:
            return None

        my_pos = all_pos.get(red_id, None)
        if my_pos is None:
            # 取不到位置时，退化：就选第一个
            return tracks[0]["target_id"]

        my_lon, my_lat = my_pos.x, my_pos.y
        best_tid, best_d = None, 1e18
        for t in tracks:
            tid = t["target_id"]
            lon, lat = t["lon"], t["lat"]
            d = self._distance_m(my_lon, my_lat, lon, lat)
            if d < best_d:
                best_d, best_tid = d, tid
        return best_tid

    def step(self):
        """执行一次扫描与火力控制"""
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e)
            return

        vis = normalize_visible(raw_visible)
        #输出范围内的敌机
        print(vis)
        try:
            all_pos = self.client.get_vehicle_pos()  # {id: Position}
        except Exception:
            all_pos = {}

        now = time.time()

        # 为避免多机过度集中同一目标，也可以做一个“已被锁定”的目标集合
        already_targeted = set(self.assigned_target.values())

        for red_id, tracks in vis.items():
            # 只响应我们控制的红机
            if red_id not in self.red_ids:
                continue

            # 无弹药则跳过
            if self.ammo.get(red_id, 0) <= 0:
                continue

            # 没看到任何敌机
            if not tracks:
                continue

            # 目标选择：尽可能选最近的；若已有锁定目标且仍在可见列表，继续打原目标
            keep_tid = self.assigned_target.get(red_id)
            candidate_tids = [t["target_id"] for t in tracks if "target_id" in t]

            if keep_tid in candidate_tids:
                target_id = keep_tid
            else:
                target_id = self._choose_nearest_target(red_id, tracks, all_pos)

            if target_id is None:
                continue

            # 冷却限制，避免每个周期都下发命令
            if (now - self.last_fire_time.get(red_id, 0.0)) < ATTACK_COOLDOWN_SEC:
                continue

            # 简单去重：如果别的红机已经在打这个目标，可以换第二近（可按需要保留/去掉）
            if target_id in already_targeted and keep_tid != target_id:
                # 挑选一个不同的目标（若有）
                alt = [tid for tid in candidate_tids if tid not in already_targeted]
                if alt:
                    target_id = alt[0]

            # 下发攻击命令
            try:
                uid = self.client.set_target(vehicle_id=red_id, target_id=target_id)
                print(f"[Red] {red_id} -> fire at {target_id}, order={uid}")
                self.ammo[red_id] -= 1
                self.last_fire_time[red_id] = now
                self.assigned_target[red_id] = target_id
                already_targeted.add(target_id)
            except Exception as e:
                print(f"[Red] set_target({red_id},{target_id}) failed:", e)

    def run_loop(self, stop_when_paused=False):
        """循环运行红方控制"""
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
    client.set_multiple(10)
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
            # 可选：周期性看分数
            if print(client.get_score(), flush=True)== None:
                None
            else:
                print(client.get_score(), flush=True)
    except KeyboardInterrupt:
        print("[Main] Interrupted, exiting...", flush=True)