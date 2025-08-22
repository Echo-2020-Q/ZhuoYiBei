import threading
import time
import math
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

RED_IDS = [10091, 10084, 10085, 10086]

# 蓝方集结点（187s）
BLUE_GATHER = (102.6553, 40.0581)

# 红方埋伏点：蓝方集结点往西约20km
RED_AMBUSH_POINTS = [
    Position(102.475, 40.20, 160),
    Position(102.475, 40.12, 140),
    Position(102.475, 40.00, 120),
    Position(102.475, 39.90, 100),
]

# 发射距离阈值
FIRE_MIN = 25000.0#规避
FIRE_OPT_LOW = 3500.0#射击窗口
FIRE_OPT_HIGH = 26000.0
FIRE_MAX = 25000.0#拉近

FIRE_COOLDOWN = 1#10.0
SUPPRESS_TIMEOUT = 1.8#18.0

last_fire_time = {rid: -1e9 for rid in RED_IDS}
last_shot_target = {}

def haversine(p1: Position, p2: Position) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [p1.x, p1.y, p2.x, p2.y])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371000.0 * c

def assign_targets(red_pos: dict, blue_pos: dict):
    """最近者优先分配，返回 {rid: bid}"""
    pairs = []
    for rid, rpos in red_pos.items():
        for bid, bpos in blue_pos.items():
            d = haversine(rpos, bpos)
            pairs.append((d, rid, bid))
    pairs.sort()  # 距离升序

    assigned_red, assigned_blue = set(), set()
    result = {}
    for d, rid, bid in pairs:
        if rid not in assigned_red and bid not in assigned_blue:
            result[rid] = bid
            assigned_red.add(rid)
            assigned_blue.add(bid)
    return result

def run_red(client: VehicleClient):
    # 打开雷达
    for rid in RED_IDS:
        client.enable_radar(vehicle_id=rid, state=1)

    # 先飞到埋伏点
    for rid, ambush in zip(RED_IDS, RED_AMBUSH_POINTS):
        #client.set_vehicle_path(rid, [ambush], speed=500, append=False)
        client.set_tactical_maneuver(rid, [ambush], 5, speed=500)

    while True:
        sim_t = client.get_sim_time()
        red_pos = client.get_vehicle_pos()
        vis = client.get_visible_vehicles()#返回值很复杂
        all_pos = client.get_vehicle_pos()

        # 获取蓝机坐标
        blue_ids = set()
        for rlist in vis.values():
            blue_ids.update(rlist)
        blue_pos = {bid: all_pos[bid] for bid in blue_ids if bid in all_pos}

        if not blue_pos:
            time.sleep(0.5)
            continue

        # 分配目标
        assignments = assign_targets(red_pos, blue_pos)

        # 执行决策
        for rid in RED_IDS:
            if rid not in red_pos:
                continue
            rpos = red_pos[rid]
            if rid not in assignments:
                continue
            bid = assignments[rid]
            bpos = blue_pos[bid]
            dist = haversine(rpos, bpos)

            can_fire = (sim_t - last_fire_time[rid] > FIRE_COOLDOWN and
                        (bid not in last_shot_target or sim_t - last_shot_target[bid] > SUPPRESS_TIMEOUT))

            if FIRE_OPT_LOW <= dist <= FIRE_OPT_HIGH and can_fire:
                try:
                    client.set_target(rid, bid)
                    print(f"[FIRE] red {rid} -> blue {bid} at {dist:.0f} m")
                    last_fire_time[rid] = sim_t
                    last_shot_target[bid] = sim_t
                except Exception as e:
                    print("fire error:", e)
            elif dist < FIRE_MIN:
                # 太近规避：往西南偏移 1km
                offset = Position(rpos.x - 0.01, rpos.y - 0.01, rpos.z)
                client.set_vehicle_path(rid, [offset], speed=150, append=False)
            elif dist > FIRE_MAX:
                # 太远逼近
                client.set_vehicle_path(rid, [bpos], speed=130, append=False)
            else:
                # 保持距离
                client.set_vehicle_path(rid, [bpos], speed=100, append=False)

        time.sleep(0.1)

def run_red(client):
    return