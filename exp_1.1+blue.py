import threading
import time
import math
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

# 红方飞机 ID
RED_IDS = [10091, 10084, 10085, 10086]

# 区域边界（矩形）
AREA_BOUNDS = {
    "xmin": 101.07326442811694,
    "xmax": 103.08242360888715,
    "ymin": 39.558295557025474,
    "ymax": 40.599429229677526,
}

# 禁飞区（圆形）
NFZ_CENTER = (101.96786384206956, 40.2325)
NFZ_RADIUS_M = 10416.0  # m

# 地理计算：haversine 距离
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371000.0 * c

# 检查是否在禁飞区
def in_nfz(lon, lat):
    d = haversine(lon, lat, NFZ_CENTER[0], NFZ_CENTER[1])
    return d <= NFZ_RADIUS_M

# 红方逻辑：巡逻并避开禁飞区
def run_red(client: VehicleClient):
    print("[RED] start running...")
    for rid in RED_IDS:
        client.enable_radar(vehicle_id=rid, state=1)

    # 巡逻航点（矩形四角）
    waypoints = [
        Position(AREA_BOUNDS["xmin"], AREA_BOUNDS["ymin"], 120),
        Position(AREA_BOUNDS["xmax"], AREA_BOUNDS["ymin"], 140),
        Position(AREA_BOUNDS["xmax"], AREA_BOUNDS["ymax"], 160),
        Position(AREA_BOUNDS["xmin"], AREA_BOUNDS["ymax"], 140),
    ]

    idx = 0
    while True:
        target = waypoints[idx % len(waypoints)]
        # 如果航点落在禁飞区，就跳过
        if in_nfz(target.x, target.y):
            idx += 1
            continue

        for rid in RED_IDS:
            try:
                client.set_vehicle_path(rid, [target], speed=100, append=False)
            except Exception as e:
                print(f"[RED] set_vehicle_path {rid} error:", e)

        time.sleep(10)  # 每 10 秒换一个航点
        idx += 1


if __name__ == "__main__":
    config = {
        "id": 106,
        "config": RflysimEnvConfig(
            "172.23.53.35",
            16001,
            18001
        )
    }

    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    # 蓝方线程
    def run_blue():
        combat_system = CombatSystem(client)
        combat_system.run_combat_loop()

    blue_thread = threading.Thread(target=run_blue, daemon=True)
    blue_thread.start()

    # 可选：提高仿真倍速（以加快调参），平台允许时启用
    try:
        client.set_multiple(5)
    except Exception:
        pass
    # 启动仿真
    client.start()

    # 红方主循环（阻塞）
    run_red(client)
