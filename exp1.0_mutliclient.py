import multiprocessing as mp
from rflysim import RflysimEnvConfig
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
from your_red_logic import run_red   # 你写好的红方逻辑函数

def run_blue(config):
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()
    combat_system = CombatSystem(client)
    combat_system.run_combat_loop()

if __name__ == "__main__":
    config = {
        "id": 106,
        "config": RflysimEnvConfig(
            "172.23.53.35",
            16001,
            18001
        )
    }

    # 主进程创建红方 client
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()
    client.set_multiple(5)   # 加速因子（可选）

    # 蓝方进程
    blue_proc = mp.Process(target=run_blue, args=(config,))
    blue_proc.daemon = True
    blue_proc.start()

    # 启动仿真
    client.start()

    # 主进程跑红方逻辑
    run_red(client)

    blue_proc.join()
