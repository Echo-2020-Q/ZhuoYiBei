# 网络与方案初始化配置
from rflysim import RflysimEnvConfig
from rflysim.client import VehicleClient
from BlueForce import CombatSystem

if __name__ == "__main__":
    config = {
        "id": 106,                        # 填入网页中参赛方案对应的方案号
        "config": RflysimEnvConfig(
            "172.23.53.35",              # 填入对应的ip地址
            16001,
            18001
        )
    }
    # rflysim变量初始化
    client = VehicleClient(id=config["id"], config=config["config"])
    # 蓝军算法
    combat_system = CombatSystem(client)  # 蓝军算法启动
    combat_system.run_combat_loop()
    # ————————————————————主算法：—————————————————————————— #

    pass