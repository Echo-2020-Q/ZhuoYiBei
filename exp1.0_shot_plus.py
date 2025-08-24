# 网络与方案初始化配置
from rflysim import RflysimEnvConfig, Position
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
import time

# ------------------- 红方控制算法 ------------------- #
class RedForce:
    def __init__(self, client, red_ids):
        self.client = client
        self.red_ids = red_ids
        # 每架无人机弹药上限 8 发
        self.ammo = {rid: 8 for rid in red_ids}

        # 启动雷达
        for rid in red_ids:
            res = self.client.enable_radar(vehicle_id=rid, state=1)
            print(f"Radar enabled for {rid}, uuid={res}")

    def attack_visible_targets(self):
        # 获取敌情
        visible_list = self.client.get_visible_vehicles()

        # 遍历所有红方无人机
        for red_id, enemies in visible_list.items():
            if not enemies:
                continue  # 没有敌机在雷达范围内

            # 检查是否有弹药
            if self.ammo.get(red_id, 0) <= 0:
                print(f"Red {red_id} 弹药耗尽，无法攻击")
                continue

            # 选择一个敌方目标（这里默认选第一个）
            target_id = enemies[0]

            # 发起攻击
            order_id = self.client.set_target(vehicle_id=red_id, target_id=target_id)
            print(f"Red {red_id} -> 攻击 Blue {target_id}, order_id={order_id}")

            # 弹药减一
            self.ammo[red_id] -= 1

    def run(self):
        while self.client.is_start():
            self.attack_visible_targets()
            time.sleep(0.1)  # 每秒检测一次敌情
# --------------------------------------------------- #

if __name__ == "__main__":
    config = {
        "id": 106,  # 填入网页中参赛方案对应的方案号
        "config": RflysimEnvConfig("172.23.53.35", 16001, 18001)
    }
    # rflysim变量初始化
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    # 蓝军算法 (已设定不用改)
    combat_system = CombatSystem(client)
    combat_system.run_combat_loop()

    # ————————————————————主算法：—————————————————————————— #
    client.set_multiple(10)   # 仿真加速
    client.start()            # 开始仿真

    # 初始化红方控制
    red_ids = [10091, 10084, 10085, 10086]
    red_force = RedForce(client, red_ids)

    # 启动红方控制循环
    red_force.run()

    # 输出得分
    print(client.get_score())
