import time
from rflysim import RflysimEnvConfig
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
import threading
def as_dict(obj):
    out = {}
    for k in dir(obj):
        if not k.startswith("_"):
            try:
                v = getattr(obj, k)
                if not callable(v):
                    out[k] = v
            except Exception:
                continue
    return out

if __name__ == "__main__":
    config = {
        "id": 106,
        "config": RflysimEnvConfig("172.23.53.35", 16001, 18001)
    }
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()
    client.start()

    combat_system = CombatSystem(client)
    blue_thread = threading.Thread(target=combat_system.run_combat_loop, daemon=True)
    blue_thread.start()
    print("[Main] Blue combat loop started.", flush=True)
    print("[Main] 仿真已启动，开始循环打印 situ_info")

    try:
        while True:
            situ = client.get_situ_info()
            print("=" * 60)
            print(f"[SimTime] {client.get_sim_time():.1f}s")

            if not situ:
                print("[Situ] (empty)")
            else:
                for vid, info in situ.items():
                    d = as_dict(info)
                    print(f"  ID={vid}, side={d.get('side')}, damage={d.get('damage_state')}")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("[Main] Interrupted, exiting...")
