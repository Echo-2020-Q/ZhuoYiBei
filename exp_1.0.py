"""
空战赛道红方AI（参考实现）
目标：在给定rflysim接口下，实现可运行的一套搜索-分配-攻击-规避联合策略，尽可能提升评分函数：
Total_Score=300*A/(蓝方初始+1) + 200*B + 100*Sigmoid(C) - 1000*D + 400*E

本实现特点：
1) 任务编组：4机编队，分扇区搜索与防区防御（避免扎堆与重复打击）。
2) 目标探测与态势共享：周期性读取 get_visible_vehicles / get_vehicle_pos，融合成公共敌情视图。
3) 武器发射策略：基于距离的“窗口化发射”，控制火力节奏以提高B并兼顾C；同一目标默认只分配1枚导弹，超时未命中再补射。
4) 规避与机动：近距压制/超距牵制；靠近禁飞区自动绕飞；紧急规避采用 set_tactical_maneuver。
5) 巡逻与搜索：根据地图边界自动生成栅格扫描航路；分配不同高度与扇区，覆盖更快。

注意：
- 赛题的武器射程/命中概率精确模型未知，代码中以参数化阈值控制（可在线或线下调参）。
- 本实现强调“可运行、可调参、可扩展”，具体细节可按现场环境快速微调。
"""
from __future__ import annotations
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from BlueForce import CombatSystem
# === 引入官方接口 ===
from rflysim import Position, Vel, RflysimEnvConfig
from rflysim.client import VehicleClient

# ============ 可调参数区（按需改动） ============
RED_IDS: List[int] = [10091, 10084, 10085, 10086]   # 地图标注从上到下四架红机
DEFAULT_ALT = 120.0                                  # m，巡逻高度，结合地形按需调
CRUISE_SPEED = 90                                    # 巡逻速度（接口单位为速度标量或km/h，按平台定义）
ATTACK_SPEED = 120                                   # 进攻/逼近速度
EVASIVE_SPEED = 150                                  # 规避时速度
CONTROL_DT = 0.5                                     # s，主循环周期
FIRE_COOLDOWN = 10.0                                 # s，同一架红机两次发射最小间隔（稳住B）
TARGET_SUPPRESSION_TIMEOUT = 18.0                    # s，单目标压制超时（未命中则可补射）
MAX_SIM_TIME = 5400.0                                # s，仿真上限

# 距离阈值（米），按实测调优：
FIRE_MIN_DIST = 2500.0   # 过近机动规避，不轻易发射
FIRE_OPT_LOW = 3500.0    # 最优窗口下限
FIRE_OPT_HIGH = 6500.0   # 最优窗口上限
FIRE_MAX_DIST = 9000.0   # 超过则先逼近牵制

# 禁飞区安全缓冲（米）：
NFZ_BUFFER = 400.0

# 高度分层（m）：分给四机，避免同空层拥挤
ALT_LAYERS = [100.0, 120.0, 140.0, 160.0]

# ============ 数据结构 ============
@dataclass
class EnemyTrack:
    vid: int
    pos: Position
    first_seen: float
    last_seen: float
    assigned_red: Optional[int] = None
    last_shot_time: float = -1e9

@dataclass
class RedState:
    last_fire_time: float = -1e9
    assigned_enemy: Optional[int] = None
    patrol_index: int = 0
    layer_alt: float = DEFAULT_ALT

@dataclass
class NFZ:
    id: int
    type: int                   # 0 circle, 1 polygon
    points: List[Tuple[float,float,float]]  # list of (x=lon, y=lat, z)

# ============ 工具函数 ============

def haversine_meters(p1: Position, p2: Position) -> float:
    """安全起见，本地再实现一次；也可用client.get_distance_by_lon_lat。"""
    lon1, lat1, lon2, lat2 = map(math.radians, [p1.x, p1.y, p2.x, p2.y])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    R = 6371000.0
    return R * c


def point_in_polygon(lon: float, lat: float, polygon: List[Tuple[float,float]]) -> bool:
    """射线法判定点是否在多边形内。"""
    x, y = lon, lat
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def dist_point_to_poly_border(p: Tuple[float,float], polygon: List[Tuple[float,float]]) -> float:
    """近似：返回点到多边形边界的最小距离（m）。"""
    # 近似计算：将经纬度近似为平面小片区
    def seg_dist(px, py, x1, y1, x2, y2):
        vx, vy = x2-x1, y2-y1
        wx, wy = px-x1, py-y1
        c1 = vx*wx + vy*wy
        c2 = vx*vx + vy*vy
        t = max(0.0, min(1.0, c1/(c2+1e-9)))
        projx, projy = x1 + t*vx, y1 + t*vy
        # 将经纬差转成米（粗略，足够避让）
        return haversine_meters(Position(px, py, 0), Position(projx, projy, 0))
    mind = 1e9
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % len(polygon)]
        mind = min(mind, seg_dist(p[0], p[1], x1, y1, x2, y2))
    return mind

# ============ 核心控制器 ============
class RedController:
    def __init__(self, client: VehicleClient, red_ids: List[int] = None):
        self.client = client
        self.red_ids = red_ids or RED_IDS
        self.red_state: Dict[int, RedState] = {rid: RedState(layer_alt=ALT_LAYERS[i % len(ALT_LAYERS)]) for i, rid in enumerate(self.red_ids)}
        self.enemy_tracks: Dict[int, EnemyTrack] = {}
        self.nfz_list: List[NFZ] = []
        self.map_bbox = None  # (left_top: Position, right_bottom: Position)
        self.sector_waypoints: Dict[int, List[Position]] = {}
        self.start_time = None
        self.last_assign_ts = 0.0

    # ---- 初始化 ----
    def setup(self):
        self.client.enable_rflysim()
        # 打开雷达
        for rid in self.red_ids:
            try:
                self.client.enable_radar(vehicle_id=rid, state=1)
            except Exception:
                pass
        # 读地图与禁飞区信息
        try:
            m = self.client.get_map_info()
            self.map_bbox = (m.left_top, m.right_bottom)
        except Exception:
            self.map_bbox = None
        self._load_nfz()
        # 生成分扇区巡逻航线
        self._build_sector_scan_paths()

    def _load_nfz(self):
        try:
            areas = self.client.get_areas()
        except Exception:
            areas = []
        for a in areas:
            # 约定：TYPE/name含“禁飞”即视作禁飞区
            name = getattr(a, 'name', '') or getattr(a, 'NAME', '')
            if ('禁飞' in str(name)) or getattr(a, 'type', '') == 'NFZ':
                pts = []
                for p in getattr(a, 'lonlat', []) or getattr(a, 'LONLAT', []) or []:
                    # p 形如 Position
                    pts.append((p.x, p.y, p.z))
                self.nfz_list.append(NFZ(id=getattr(a, 'id', -1), type=getattr(a, 'nZoneType', 1), points=pts))

    def _build_sector_scan_paths(self):
        # 根据地图范围生成4个扇区的蛇形航线
        if not self.map_bbox:
            # 兜底：构造四个相邻矩形扇区（经纬度粗略）
            cx = [116.3, 116.35, 116.4, 116.45]
            cy = [39.9, 39.92, 39.94, 39.96]
            for i, rid in enumerate(self.red_ids):
                self.sector_waypoints[rid] = [Position(cx[i%4], cy[i%4], ALT_LAYERS[i%4])]
            return
        lt, rb = self.map_bbox
        xmin, ymin = lt.x, rb.y
        xmax, ymax = rb.x, lt.y
        # 将横向分成4条扫描带
        bands = []
        width = (xmax - xmin) / 4.0
        for i in range(4):
            x1 = xmin + i*width
            x2 = x1 + width
            # 每条带内构造蛇形航点
            rows = 6
            ys = [ymin + k*(ymax - ymin)/rows for k in range(rows+1)]
            wps: List[Position] = []
            for idx, y in enumerate(ys):
                x_start, x_end = (x1, x2) if idx % 2 == 0 else (x2, x1)
                wps.append(Position(x_start, y, DEFAULT_ALT))
                wps.append(Position(x_end, y, DEFAULT_ALT))
            bands.append(wps)
        for i, rid in enumerate(self.red_ids):
            self.sector_waypoints[rid] = bands[i % len(bands)]

    # ---- 主循环 ----
    def run(self):
        self.start_time = time.time()
        # 开始仿真
        self.client.start()
        while True:
            try:
                sim_t = self.client.get_sim_time()
            except Exception:
                sim_t = time.time() - self.start_time
            if sim_t >= MAX_SIM_TIME:
                break
            # 更新态势与公共敌情库
            red_pos = self.client.get_vehicle_pos()
            vis = self.client.get_visible_vehicles()  # {red_id: [blue_ids...]}
            blue_pos = self._estimate_blue_positions(vis)
            self._update_enemy_tracks(blue_pos, sim_t)
            # 目标分配（每1.5s更新一次）
            if sim_t - self.last_assign_ts > 1.5:
                self._assign_targets(red_pos)
                self.last_assign_ts = sim_t
            # 行动执行
            self._execute_actions(red_pos, sim_t)
            time.sleep(CONTROL_DT)
        # 结束
        try:
            score = self.client.get_score()
            print("Final Score:", score)
        except Exception:
            pass

    # ---- 敌情融合 ----
    def _estimate_blue_positions(self, vis: Dict[int, List[int]]) -> Dict[int, Position]:
        # 平台未提供蓝机直接坐标接口；策略：取所有红机get_vehicle_pos，
        # 若能反查到蓝机坐标则用之；否则在EnemyTrack中保持最近一次估计。
        # 这里尝试从 get_vehicle_pos() 中包含的所有实体提取（若平台提供）。
        all_pos = self.client.get_vehicle_pos()
        blue_ids: Set[int] = set()
        for rid, arr in (vis or {}).items():
            for bid in arr or []:
                blue_ids.add(bid)
        out = {}
        for bid in blue_ids:
            if bid in all_pos:
                out[bid] = all_pos[bid]
        return out

    def _update_enemy_tracks(self, blue_pos: Dict[int, Position], sim_t: float):
        now = sim_t
        for bid, p in blue_pos.items():
            if bid not in self.enemy_tracks:
                self.enemy_tracks[bid] = EnemyTrack(vid=bid, pos=p, first_seen=now, last_seen=now)
            else:
                tr = self.enemy_tracks[bid]
                tr.pos = p
                tr.last_seen = now
        # 清理长期未见目标
        stale: List[int] = []
        for bid, tr in self.enemy_tracks.items():
            if now - tr.last_seen > 30.0:
                stale.append(bid)
        for bid in stale:
            self.enemy_tracks.pop(bid, None)

    # ---- 目标分配 ----
    def _assign_targets(self, red_pos: Dict[int, Position]):
        # 最近者优先 + 单目标仅被一机追击，减少重复开火
        # 先清空无效指派
        alive_blues = set(self.enemy_tracks.keys())
        for rid, rs in self.red_state.items():
            if rs.assigned_enemy not in alive_blues:
                rs.assigned_enemy = None

        # 计算所有可分配对（rid,bid,dist）
        pairs: List[Tuple[float,int,int]] = []
        for rid in self.red_ids:
            rpos = red_pos.get(rid)
            if not rpos:
                continue
            for bid, tr in self.enemy_tracks.items():
                d = haversine_meters(rpos, tr.pos)
                pairs.append((d, rid, bid))
        pairs.sort()

        taken_red: Set[int] = set()
        taken_blue: Set[int] = set()
        for d, rid, bid in pairs:
            if rid in taken_red or bid in taken_blue:
                continue
            # 若该红机已有目标且更近，则不改
            cur = self.red_state[rid].assigned_enemy
            if cur is None or d < 0.8 * haversine_meters(red_pos[rid], self.enemy_tracks[cur].pos):
                self.red_state[rid].assigned_enemy = bid
            taken_red.add(rid)
            taken_blue.add(bid)

    # ---- 行动执行：机动与火力 ----
    def _execute_actions(self, red_pos: Dict[int, Position], sim_t: float):
        for rid in self.red_ids:
            rstate = self.red_state[rid]
            rpos = red_pos.get(rid)
            if not rpos:
                continue
            # 1) 禁飞区避让 & 巡逻
            if rstate.assigned_enemy is None:
                self._nfz_safe_patrol(rid, rpos)
                continue
            # 2) 有目标 -> 逼近/保持距离/发射/规避
            bid = rstate.assigned_enemy
            track = self.enemy_tracks.get(bid)
            if not track:
                self._nfz_safe_patrol(rid, rpos)
                continue
            bpos = track.pos
            dist = haversine_meters(rpos, bpos)

            # 高度保持分层
            self._set_altitude(rid, rstate.layer_alt)

            # 发射条件控制
            can_fire = (sim_t - rstate.last_fire_time) > FIRE_COOLDOWN and (sim_t - track.last_shot_time) > TARGET_SUPPRESSION_TIMEOUT
            if FIRE_OPT_LOW <= dist <= FIRE_OPT_HIGH and can_fire:
                self._fire(rid, bid)
                rstate.last_fire_time = sim_t
                track.last_shot_time = sim_t
                # 发射后微调：侧向偏移，降低被对射风险
                self._lateral_offset_move(rid, rpos, bpos, ATTACK_SPEED)
            elif dist < FIRE_MIN_DIST:
                # 太近 -> 规避后再打
                self._evasive_break(rid, rpos, bpos)
            elif dist > FIRE_MAX_DIST:
                # 过远 -> 逼近
                self._intercept_move(rid, rpos, bpos, ATTACK_SPEED)
            else:
                # 位于次优窗口 -> 保持/小幅逼近
                self._intercept_move(rid, rpos, bpos, CRUISE_SPEED)

    # ---- 行动原语 ----
    def _nfz_safe_patrol(self, rid: int, rpos: Position):
        wps = self.sector_waypoints.get(rid, [])
        if not wps:
            return
        st = self.red_state[rid]
        target = wps[st.patrol_index % len(wps)]
        # 如果目标在禁飞区内部，跳到下一个
        if self._in_nfz(target):
            st.patrol_index = (st.patrol_index + 1) % len(wps)
            target = wps[st.patrol_index]
        # 若当前点附近出现禁飞边界，构造旁路点
        if self._near_nfz(rpos):
            bypass = self._compute_bypass_waypoint(rpos)
            if bypass:
                self._fly_to(rid, bypass, CRUISE_SPEED)
                return
        self._fly_to(rid, Position(target.x, target.y, st.layer_alt), CRUISE_SPEED)
        # 到点切换（经纬度阈值~150m）
        if haversine_meters(rpos, target) < 150.0:
            st.patrol_index = (st.patrol_index + 1) % len(wps)

    def _intercept_move(self, rid: int, rpos: Position, bpos: Position, speed: int):
        # 简单PN拦截近似：飞向目标当前点
        self._fly_to(rid, Position(bpos.x, bpos.y, self.red_state[rid].layer_alt), speed)

    def _lateral_offset_move(self, rid: int, rpos: Position, bpos: Position, speed: int):
        # 侧向偏置一个小量（经度方向偏置 ~300m）
        # 估算经度换算（粗略）：1度经度在该纬度约=111km*cos(lat)
        lat_rad = math.radians(rpos.y)
        lon_delta = 0.3 / (111000.0 * math.cos(lat_rad) + 1e-6)
        self._fly_to(rid, Position(bpos.x + lon_delta, bpos.y, self.red_state[rid].layer_alt), speed)

    def _evasive_break(self, rid: int, rpos: Position, bpos: Position):
        # 远离目标 + 战术机动
        # 选择与目标反向的点（退离800m）
        bearing = self._bearing(bpos, rpos)
        away = self._project(rpos, bearing, 800.0)
        self._fly_to(rid, Position(away.x, away.y, self.red_state[rid].layer_alt+10.0), EVASIVE_SPEED)
        try:
            self.client.set_tactical_maneuver(vehicle_id=rid, position=[Position(away.x, away.y, self.red_state[rid].layer_alt)], tactical_type=3, speed=EVASIVE_SPEED)
        except Exception:
            pass

    def _fire(self, rid: int, bid: int):
        try:
            uid = self.client.set_target(vehicle_id=rid, target_id=bid)
            print(f"[FIRE] red {rid} -> blue {bid}, uid={uid}")
        except Exception as e:
            print(f"[FIRE-ERR] red {rid} -> blue {bid}: {e}")

    def _set_altitude(self, rid: int, alt: float):
        vel = Vel()
        vel.vz = alt
        vel.rate = CRUISE_SPEED
        vel.direct = 0  # 朝北占位；真正朝向由set_vehicle_path更合适，这里仅用于高度保持
        try:
            self.client.set_vehicle_vel(rid, vel)
        except Exception:
            pass

    def _fly_to(self, rid: int, p: Position, speed: int):
        try:
            self.client.set_vehicle_path(rid, [p], speed, append=False)
        except Exception as e:
            # 回退到速度模式
            vel = Vel()
            vel.vz = p.z
            vel.rate = speed
            # 简化：计算绝对方位角
            red_pos = self.client.get_vehicle_pos().get(rid)
            if red_pos:
                vel.direct = self._bearing(red_pos, p)
            try:
                self.client.set_vehicle_vel(rid, vel)
            except Exception:
                pass

    # ---- 禁飞区判定/绕飞 ----
    def _in_nfz(self, p: Position) -> bool:
        for z in self.nfz_list:
            if z.type == 0 and len(z.points) >= 2:
                c = z.points[0]; r = z.points[1]
                center = Position(c[0], c[1], 0)
                edge = Position(r[0], r[1], 0)
                if haversine_meters(Position(p.x, p.y, 0), center) <= haversine_meters(center, edge) - NFZ_BUFFER:
                    return True
            elif z.type == 1 and len(z.points) >= 3:
                poly = [(x, y) for x, y, _ in z.points]
                if point_in_polygon(p.x, p.y, poly):
                    return True
        return False

    def _near_nfz(self, p: Position) -> bool:
        for z in self.nfz_list:
            if z.type == 0 and len(z.points) >= 2:
                c = z.points[0]; r = z.points[1]
                center = Position(c[0], c[1], 0)
                edge = Position(r[0], r[1], 0)
                d = haversine_meters(Position(p.x, p.y, 0), center) - haversine_meters(center, edge)
                if abs(d) < (NFZ_BUFFER + 150.0):
                    return True
            elif z.type == 1 and len(z.points) >= 3:
                poly = [(x, y) for x, y, _ in z.points]
                d = dist_point_to_poly_border((p.x, p.y), poly)
                if d < (NFZ_BUFFER + 150.0):
                    return True
        return False

    def _compute_bypass_waypoint(self, p: Position) -> Optional[Position]:
        # 简易：向北/向南偏移 ~600m 试探绕飞
        lat_rad = math.radians(p.y)
        dlat = 0.6 / 111000.0
        cand = [Position(p.x, p.y + dlat, p.z), Position(p.x, p.y - dlat, p.z)]
        for q in cand:
            if not self._in_nfz(q):
                return q
        return None

    # ---- 地理辅助 ----
    @staticmethod
    def _bearing(a: Position, b: Position) -> float:
        # 返回绝对方向角(0~360, 北=0, 顺时针)，与接口direct一致
        dlon = math.radians(b.x - a.x)
        lat1 = math.radians(a.y)
        lat2 = math.radians(b.y)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
        brng = (math.degrees(math.atan2(y, x)) + 360) % 360
        return brng

    @staticmethod
    def _project(p: Position, bearing_deg: float, dist_m: float) -> Position:
        R = 6371000.0
        br = math.radians(bearing_deg)
        lat1 = math.radians(p.y)
        lon1 = math.radians(p.x)
        lat2 = math.asin(math.sin(lat1)*math.cos(dist_m/R) + math.cos(lat1)*math.sin(dist_m/R)*math.cos(br))
        lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dist_m/R)*math.cos(lat1), math.cos(dist_m/R)-math.sin(lat1)*math.sin(lat2))
        return Position(math.degrees(lon2), math.degrees(lat2), p.z)


# ============ 启动入口 ============
if __name__ == "__main__":
    config = {
        "id": 106,
        "config": RflysimEnvConfig(
            "172.23.53.35",  # 替换为实际IP
            16001,
            18001,
        ),
    }
    client_blue = VehicleClient(id=config["id"], config=config["config"])
    client_blue.enable_rflysim()
    # 蓝军算法
    combat_system = CombatSystem(client_blue)
    # 蓝军算法启动
    combat_system.run_combat_loop()
    client_blue.set_multiple(5)
    client_blue.start()
    client = VehicleClient(id=config["id"], config=config["config"])

    ctrl = RedController(client, RED_IDS)
    ctrl.setup()

    # 可选：提高仿真倍速（以加快调参），平台允许时启用
    try:
        client.set_multiple(5)
    except Exception:
        pass

    # 启动主循环
    ctrl.run()
