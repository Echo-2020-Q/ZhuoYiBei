import multiprocessing as mp
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from rflysim import RflysimEnvConfig
from rflysim.client import VehicleClient
from BlueForce import CombatSystem
from rflysim import RflysimEnvConfig
from rflysim.client import VehicleClient
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# ============ 1. 定义红方环境 ============
class RedEnv(gym.Env):
    def __init__(self, client):
        super(RedEnv, self).__init__()
        self.client = client
        self.red_ids = [10091, 10084, 10085, 10086]
        self.blue_ids = [10087, 10089, 10088, 10090]
        # 假设 observation 是敌方相对位置 + 自己状态 (可以扩展)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(4,), dtype=np.float32)

        # 假设 action 空间 = {前进, 后退, 左转, 右转, 开火}
        self.action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        success = self.client.restart()
        if not success:
            print("Warning: restart failed.")

        obs = self._get_obs()
        return obs

    def step(self, action):
        # 执行动作
        if action == 0:
            self.client.move_forward(10)
        elif action == 1:
            self.client.move_backward(10)
        elif action == 2:
            self.client.turn_left(15)
        elif action == 3:
            self.client.turn_right(15)
        elif action == 4:
            self.client.fire()

        # 获取新状态
        obs = self._get_obs()

        # 奖励函数（例子：击中敌人 +100，靠近敌人 +距离奖励）
        reward = 0
        visible = self.client.get_visible_vehicles
        if visible:  # 如果看见敌人
            reward += 10
        # TODO: 可以在这里加上命中敌人/存活时间/燃料消耗的奖励

        terminated = False   # ✅ episode 是否自然结束（比如被击毁）
        truncated = False    # ✅ 是否因为时间/限制被强行截断
        info = {}

        return obs, reward, terminated, truncated, info   # ✅ 新接口要求 5 个返回值

    def _get_obs(self):
        # 例子：用敌人位置 - 自己位置
        me = self.client.get_position()   # (x,y)
        enemy_list = self.client.get_visible_vehicles()
        if enemy_list:
            enemy = self.client.get_enemy_position(enemy_list[0])
            obs = np.array([enemy[0]-me[0], enemy[1]-me[1], me[0], me[1]], dtype=np.float32)
        else:
            obs = np.zeros(4, dtype=np.float32)
        return obs


# ============ 2. 蓝方逻辑 ============
def run_blue(config):
    client = VehicleClient(id=config["id"], config=config["config"])
    client.enable_rflysim()
    combat_system = CombatSystem(client)
    combat_system.run_combat_loop()


# ============ 3. 主程序 ============
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
    client.set_multiple(5)

    # 蓝方进程
    blue_proc = mp.Process(target=run_blue, args=(config,))
    blue_proc.daemon = True
    blue_proc.start()

    # 启动仿真
    client.start()

    # ============ 4. 强化学习训练 ============
    env = RedEnv(client)
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练 10000 步
    model.learn(total_timesteps=10000)

    # 保存策略
    model.save("ppo_red")

    # ============ 5. 使用训练好的策略 ============
    obs, info = env.reset()   # ✅ gymnasium 需要解包
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:       # ✅ 替代 done
            obs, info = env.reset()

    blue_proc.join()
