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
ATTACK_COOLDOWN_SEC = 2.0   # 同一红机连续下发攻击命令的最短间隔（避免刷命令）
SCAN_INTERVAL_SEC = 0.1     # 侦察扫描周期
LOCK_SEC = 20.0  # 每个目标的锁定时间窗口
BLUE_SIDE = 2       #蓝方的编号
# 放在类里某处：枚举与判定
BLUE_SIDE_CODE = 2           # 你的环境观测为 2=BLUE, 1=RED
DESTROY_CODES   = {1}        # 你的环境观测为 1=DESTROYED

vel_0 = rflysim.Vel()
vel_0.vz = 150#飞行目标高度

vel_0.rate = 100 # 速率

vel_0.direct = 90 # 速度在地图内的绝对方向，正北为0，0-360表示完整一周的方向
last_step_time=0
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
        self._destroy_codes = {1}  # 你环境下 1 表示摧毁；若以后发现值变了，只改这里

        self.lock = threading.Lock()
        self.target_locks = {}  # {target_id: {"red_id": int, "until": float}}

        # 新保存上次快照
        self._last_visible = set()
        self._last_locked = {}
        self._last_destroyed = set()
        self._last_situ_info=set()

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
                time.sleep(0.2)
                print(client.get_command_status(vuid))
            except Exception as e:
                print(f"[Red] set_vehicle_vel({rid}) failed: {e}", flush=True)

        self.target_in_progress = set()  # 正在被某红机处理的目标，防多机浪费
        self.destroyed_targets = set()  # 已确认损毁的目标
        self.lock = threading.Lock()  # 线程锁，保护上面两个集合


        # ---------- 工具 ---------- #

    def _is_destroyed_flag(self, dmg):
        """兼容字符串/数值的毁伤判定"""
        if dmg is None:
            return False
        # 字符串：兼容 "DAMAGE_STATE_DESTROYED" / "DESTROYED" / "1"
        if isinstance(dmg, str):
            s = dmg.strip().upper()
            if "DESTROY" in s:
                return True
            try:
                return int(s) in DESTROY_CODES
            except Exception:
                return False
        # 数值：0/1/2...
        try:
            return int(dmg) in DESTROY_CODES
        except Exception:
            return False

    def _to_int_or(self, val, fallback=None):
        try:
            return int(val)
        except Exception:
            return fallback

    def _is_fire_success(self, uid, max_tries=5, wait_s=0.1):
        """
        轻量同步确认：尝试几次查询命令状态。
        成功条件：
          - status == 'EXECUTE_SUCCESS'，或
          - execute_result 文案里包含 '执行打击成功'
        否则视为失败/无效。
        """
        if not uid:
            return False
        for _ in range(max_tries):
            time.sleep(wait_s)  # 真实时间小延迟，给后端写入时间
            try:
                st = self.client.get_command_status(uid)
            except Exception:
                st = None
            if isinstance(st, dict):
                status = st.get('status', '')
                result = st.get('execute_result', '') or ''
                if status == 'EXECUTE_SUCCESS' or ('执行打击成功' in result):
                    return True
                # 有明确失败就可以直接返回 False（避免多等）
                if status == 'EXECUTE_FAILED':
                    return False
        return False


    def _get_target_pos(self, target_id, vis, all_pos):
        """
        目标坐标优先用 all_pos（仿真高精度位置）；若取不到，再从雷达 tracks 里找 lon/lat 兜底。
        返回 (lon, lat) 或 (None, None)
        """
        p = all_pos.get(target_id)
        if p is not None:
            return p.x, p.y

        # 兜底：从任意红机的 tracks 里找这个 target_id 的 lon/lat
        for tracks in vis.values():
            for t in tracks:
                if t.get("target_id") == target_id:
                    return t.get("lon"), t.get("lat")
        return (None, None)

    def _update_destroyed_from_situ(self, force=False):
        """以 get_situ_info() 为唯一真值来源，更新毁伤状态并清理锁。"""
        if not force:
            self._situ_counter = getattr(self, "_situ_counter", 0) + 1
            if self._situ_counter % 5 != 0:  # 每5次step才真正调用，减负
                return

        try:
            situ_raw = self.client.get_situ_info()
        except Exception:
            return

        # 空返回一次性告警（不刷屏）
        if (not situ_raw) and (not getattr(self, "_printed_situ_warn", False)):
            print("[WARN] situ parsing returned empty.", flush=True)
            self._printed_situ_warn = True
            return

        to_mark_destroyed = []

        # situ_raw 可能是 {id: 对象/字典}
        for k, info in (situ_raw or {}).items():
            if info is None:
                continue
            # 统一成 dict（关键！）
            d = info if isinstance(info, dict) else _as_dict(info)

            vid = self._to_int_or(d.get("id", k))
            if vid is None:
                continue

            side = d.get("side")
            # side 也可能是字符串，如 "SIDE_BLUE"
            side_code = self._to_int_or(side)
            if side_code is None and isinstance(side, str):
                s = side.upper()
                if "BLUE" in s:
                    side_code = BLUE_SIDE_CODE
                elif "RED" in s:
                    side_code = 1  # 给个常见缺省

            dmg = d.get("damage_state")

            # 只标记蓝机被毁
            if side_code == BLUE_SIDE_CODE and self._is_destroyed_flag(dmg):
                to_mark_destroyed.append(vid)

        if not to_mark_destroyed:
            return

        with self.lock:
            for vid in to_mark_destroyed:
                if vid not in self.destroyed_targets:
                    print(f"[Situ] Mark destroyed: {vid}", flush=True)
                self.destroyed_targets.add(vid)
                # 若该目标在锁定表里，立即释放
                self.target_locks.pop(vid, None)
                # 我方红机毁伤（极端 case）：清零弹药，避免后续逻辑再使用
                if vid in self.ammo:
                    self.ammo[vid] = 0

    def _normalize_situ(self, situ):
        """
        situ -> { id(int): {"side": str|None, "damage_state": str|None} }
        兼容：dict/对象/字符串。
        """
        out = {}
        # 1) 整体是 dict 的常见情况：key 是 id，value 是对象/字典/字符串
        if isinstance(situ, dict):
            for k, v in situ.items():
                # 尝试先拿 id
                vid = None
                # k 可能就是 id
                try:
                    vid = int(k)
                except Exception:
                    vid = None

                if isinstance(v, dict):
                    # 直接从 dict 里以更宽松的键名取
                    if vid is None:
                        _id = self._normalize_key_lookup(v, "id")
                        try:
                            vid = int(_id)
                        except Exception:
                            pass
                    side = self._normalize_key_lookup(v, "side")
                    damage = self._normalize_key_lookup(v, "damage_state", "damage state")

                elif hasattr(v, "__dict__") or isinstance(v, object):
                    # 对象：转 dict 再取
                    vd = _as_dict(v)
                    if vid is None:
                        _id = self._normalize_key_lookup(vd, "id")
                        try:
                            vid = int(_id)
                        except Exception:
                            pass
                    side = self._normalize_key_lookup(vd, "side")
                    damage = self._normalize_key_lookup(vd, "damage_state", "damage state")

                else:
                    # 字符串兜底
                    parsed = self._parse_situ_from_string(str(v))
                    if parsed:
                        out.update(parsed)
                    continue  # 此分支已处理完

                if vid is None:
                    continue
                out[int(vid)] = {"side": side, "damage_state": damage}
            return out

        # 2) 整体是字符串：兜底
        if isinstance(situ, str):
            return self._parse_situ_from_string(situ)

        # 3) 其他未知类型
        return out

    def _normalize_key_lookup(self, d, *candidates):
        """
        在字典 d 里，做“大小写不敏感 + 忽略空格和下划线”的键查找。
        例如：("damage_state","damage state") 都能命中。
        """
        if not isinstance(d, dict):
            return None
        norm = {str(k).lower().replace("_", "").replace(" ", ""): k for k in d.keys()}
        for c in candidates:
            ck = str(c).lower().replace("_", "").replace(" ", "")
            if ck in norm:
                return d.get(norm[ck])
        return None

    def _parse_situ_from_string(self, s):
        """
        兜底：当 get_situ_info() 的某条目是字符串时，从中提取 id / side / damage_state。
        返回 {vid: {"side": str|None, "damage_state": str|None}} 或 {}
        """
        if not isinstance(s, str):
            return {}
        out = {}

        # 可能包含多段，用 'id:' 作为分割尝试
        # 1) 先找所有 'id: <num>'
        ids = [int(m.group(1)) for m in re.finditer(r'\bid\s*:\s*(\d+)', s)]
        if not ids:
            # 整段只有一个对象的情况
            mid = re.search(r'\bid\s*:\s*(\d+)', s)
            vid = int(mid.group(1)) if mid else None
            if vid is None:
                return out
            # side
            ms = re.search(r'\bside\s*:\s*([A-Z_ ]+)', s, re.I)
            side = ms.group(1).strip() if ms else None
            # damage_state
            md = re.search(r'\bdamage\s*[_ ]?state\s*:\s*([A-Z_]+)', s, re.I)
            dmg = md.group(1).strip() if md else None
            out[vid] = {"side": side, "damage_state": dmg}
            return out

        # 2) 多对象粗分割：以每个 id 的位置切片
        spans = [(m.start(), int(m.group(1))) for m in re.finditer(r'\bid\s*:\s*(\d+)', s)]
        spans.append((len(s), None))
        for i in range(len(spans) - 1):
            start, vid = spans[i]
            end, _ = spans[i + 1]
            chunk = s[start:end]
            # side
            ms = re.search(r'\bside\s*:\s*([A-Z_ ]+)', chunk, re.I)
            side = ms.group(1).strip() if ms else None
            # damage_state
            md = re.search(r'\bdamage\s*[_ ]?state\s*:\s*([A-Z_]+)', chunk, re.I)
            dmg = md.group(1).strip() if md else None
            out[vid] = {"side": side, "damage_state": dmg}
        return out

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
        """蓝机优先：遍历可见蓝机→每个挑最近可用红机→下发攻击；目标锁定 20s 或直至损毁。"""
        # 1) 感知
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True)
            return
        vis = normalize_visible(raw_visible)

        try:
            all_pos = self.client.get_vehicle_pos()  # {id: Position}
        except Exception:
            all_pos = {}

        # 用态势信息刷新损毁集合，并清理锁定表中已损毁的目标
        self._update_destroyed_from_situ()

        # 改成仿真的时间最好，后面再改
        now = self.client.get_sim_time()

        # 2) 清理过期锁定（20s 到了还没损毁 → 解除锁定）
        expired = [tid for tid, meta in self.target_locks.items() if now >= meta["until"]]
        for tid in expired:
            # 未损毁则释放回可分配；损毁的会在 _update_destroyed_from_situ() 时被彻底跳过
            self.target_locks.pop(tid, None)

        # 3) 汇总“当前可见”的蓝机集合（来自雷达；去掉已损毁 + 正在锁定中的）
        visible_blue_targets = set()
        for tracks in vis.values():
            for t in tracks:
                tid = t.get("target_id")
                if tid is None:
                    continue
                if tid < 10000:  # 🚩 优化：只允许蓝机 ID
                    continue
                if tid in self.destroyed_targets:
                    continue
                if tid in self.target_locks:
                    continue  # 锁定期内不再分配
                visible_blue_targets.add(tid)

        # 只在变化时输出
        if visible_blue_targets != self._last_visible:
            print("未锁定的目标:", visible_blue_targets, flush=True)
            self._last_visible = set(visible_blue_targets)

        if self.target_locks != self._last_locked:
            print("已经锁定的目标:", self.target_locks, flush=True)
            # 这里存个浅拷贝，避免引用同一个对象
            self._last_locked = dict(self.target_locks)

        if self.destroyed_targets != self._last_destroyed:
            print("已经被摧毁的目标:", self.destroyed_targets, flush=True)
            self._last_destroyed = set(self.destroyed_targets)


        if not visible_blue_targets:
            return

        # 4) 为每个蓝机选“最近的可用红机”（有弹药、冷却已过、红机未损毁）
        used_reds_this_round = set()  # 避免一轮里同一红机打多个
        assignments = []  # (red_id, target_id)

        for tid in visible_blue_targets:
            # 取目标位置
            t_lon, t_lat = self._get_target_pos(tid, vis, all_pos)
            if t_lon is None or t_lat is None:
                continue

            # 在所有可用红机中挑最近
            best_red, best_d = None, 1e18
            for rid in self.red_ids:
                if rid in used_reds_this_round:
                    continue
                # 红机自身状态（未损毁、有弹、冷却到期）
                if rid in self.destroyed_targets:
                    continue
                if self.ammo.get(rid, 0) <= 0:
                    continue
                if (now - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                    continue
                # 红机位置
                r_pos = all_pos.get(rid)
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

        # 5) 下发攻击，并对该目标设置 20s 锁定窗口
        for rid, tid in assignments:
            # 双重确认：刚好此刻被其他事件判损毁了 → 跳过
            if tid in self.destroyed_targets:
                continue

            # 发射前再用仿真时间做一次冷却确认
            now_sim = self.client.get_sim_time()
            if (now_sim - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                continue
            # 实际弹药也再判一次
            if self.ammo.get(rid, 0) <= 0:
                continue

            try:
                uid = self.client.set_target(vehicle_id=rid, target_id=tid)
                print(f"[Red] {rid} -> fire at {tid}, uid={uid}", flush=True)

                # —— 关键：只有“执行打击成功”才视为有效 —— #
                ok = self._is_fire_success(uid, max_tries=5, wait_s=0.1)
                if not ok:
                    # 打击无效：不扣弹、不记冷却、不加锁；让该目标留在可分配列表
                    print(f"[Red] fire NOT confirmed for {rid}->{tid}, keep target available.", flush=True)
                    # 节流：多条命令之间 0.1s
                    time.sleep(0.1)
                    continue

                # 发射有效：扣弹、记冷却（用仿真时间）、加锁 20s
                self.ammo[rid] -= 1
                self.last_fire_time[rid] = self.client.get_sim_time()
                self.assigned_target[rid] = tid
                self.target_locks[tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}

                # 节流：多条命令之间 0.1s（真实时间）
                time.sleep(0.1)

            except Exception as e:
                print(f"[Red] set_target({rid},{tid}) failed: {e}", flush=True)
                # 失败：同样不扣弹、不加锁
                continue

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