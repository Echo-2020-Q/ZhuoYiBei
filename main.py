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
    # rflysim变量初始化  红军的四架飞机id从地图上上到下分别为10091 10084 10085 10086
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()

    res = client.enable_radar(vehicle_id=10091, state=1)
    print("10091uuid:",res)
    res = client.enable_radar(vehicle_id=10086, state=1)
    print("10086uuid:",res)
    visible_list = client.get_visible_vehicles()
    print(visible_list)




    # 蓝军算法 (已经设置好的不用管)
    #combat_system = CombatSystem(client)
    # 蓝军算法启动
    #combat_system.run_combat_loop()

    # ————————————————————主算法：—————————————————————————— #
    # 开始仿真
    client.set_multiple(10)
    client.start()
    client.set_multiple(10)
    print(client.get_score())
