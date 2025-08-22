import gym
import numpy as np
import time
from gym import spaces
from rflysim.client import VehicleClient
from rflysim import RflysimEnvConfig, Position, Vel
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
# --------------------------
# 1. 集群无人机环境封装
# --------------------------
class DroneSwarmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}
    def __init__(self, client):
        super().__init__()
        self.client = client  # 传入VehicleClient实例
        self.red_ids = []  # 红方无人机ID列表（4架）
        self.blue_ids = []  # 蓝方无人机ID列表（4架）
        self.ground_target_count = 5  # 假设地面目标总数为5（可根据实际场景调整）
        self.episode_steps = 0
        self.max_steps = 1000  # 单局最大步数
        self.sim_time_limit = 5400  # 仿真时间上限（秒）
        # --------------------------
        # 状态空间（单架无人机）
        # 包含：自身状态、目标信息、集群信息
        # --------------------------
        # 观测空间设计
        # [lon_red, lat_red, vx, vy, vz, lon_b1, lat_b1, d1, lon_b2, lat_b2, d2, lon_b3, lat_b3, d3, lon_b4, lat_b4, d4]
        low = np.array([
            101.07326442811694, 39.558295557025474, -200, -200, -200,  # 红1方位置 + 速度
            101.07326442811694, 39.558295557025474, #红2
            101.07326442811694, 39.558295557025474, #红3
            101.07326442811694, 39.558295557025474, #红4
            101.07326442811694, 39.558295557025474, 0,  # 蓝1
            101.07326442811694, 39.558295557025474, 0,  # 蓝2
            101.07326442811694, 39.558295557025474, 0,  # 蓝3
            101.07326442811694, 39.558295557025474, 0,  # 蓝4
        ], dtype=np.float64)

        high = np.array([
            103.08242360888715, 40.599429229677526, 200, 200, 200,  # 红方位置 + 速度
            103.08242360888715, 40.599429229677526,#红2
            103.08242360888715, 40.599429229677526,  # 红3
            103.08242360888715, 40.599429229677526,  # 红4
            103.08242360888715, 40.599429229677526, 180,  # 蓝1 最大距离设20万米
            103.08242360888715, 40.599429229677526, 180,  # 蓝2
            103.08242360888715, 40.599429229677526, 180,  # 蓝3
            103.08242360888715, 40.599429229677526, 180,  # 蓝4
        ], dtype=np.float64)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # --------------------------
        # 动作空间（单架无人机）
        # 包含：速度控制+攻击指令
        # --------------------------
        self.action_space = spaces.Box(
            low=np.array([-15, -15, -5, 0]),  # vx, vy, vz, 攻击指令（0=不攻击）
            high=np.array([15, 15, 5, 1]),   # 攻击指令（1=攻击最近目标）
            dtype=np.float32
        )
        # 初始化仿真
        self.client.enable_rflysim(log_level=2)  # 仅打印warn和error
        self.client.start()
        time.sleep(2)  # 等待仿真加载
    def reset(self, seed=None, options=None):
        """重置环境，初始化红/蓝方无人机ID"""
        super().reset(seed=seed)
        self.client.restart()
        time.sleep(2)
        self.episode_steps = 0
        # 获取红/蓝方无人机ID（通过阵营区分，side=1为红方，2为蓝方）
        situ_info = self.client.get_situ_info()
        self.red_ids = [id for id, info in situ_info.items() if info.side == 1]
        self.blue_ids = [id for id, info in situ_info.items() if info.side == 2]
        assert len(self.red_ids) == 4, "红方无人机数量不为4"
        assert len(self.blue_ids) == 4, "蓝方无人机数量不为4"
        # 开启所有红方无人机雷达
        for red_id in self.red_ids:
            self.client.enable_radar(red_id, state=1)  # 1=开机
        return self._get_observations(), {}
    def step(self, actions):
        """执行集群动作，返回全局状态与奖励"""
        self.episode_steps += 1
        rewards = [0.0 for _ in self.red_ids]  # 每架无人机的奖励
        # --------------------------
        # 1. 执行动作（速度控制+攻击）
        # --------------------------
        for i, red_id in enumerate(self.red_ids):
            action = actions[i]  # 第i架无人机的动作
            # 控制速度（NED模式）
            vel = Vel()
            vel.vx = action[0]
            vel.vy = action[1]
            vel.vz = action[2]
            self.client.set_vehicle_vel(red_id, vel, mode=1)
            # 攻击指令（1=攻击最近的蓝方目标）
            if action[3] > 0.5:
                visible = self.client.get_visible_vehicles()
                if red_id in visible and len(visible[red_id]) > 0:
                    # 攻击最近的敌方目标
                    target_id = visible[red_id][0]
                    self.client.set_target(red_id, target_id)
                    rewards[i] += 10  # 攻击尝试奖励
        time.sleep(0.2)  # 步长间隔（200ms）
        # --------------------------
        # 2. 获取全局状态与评分信息
        # --------------------------
        observations = self._get_observations()
        score_info = self.client.get_score()  # 包含红蓝方毁伤数等
        collide_info = self.client.get_vehicle_collide()
        sim_time = self.client.get_sim_time()
       禁飞区总时间 = score_info.get("red_in_area_time_sec", 0)
        # --------------------------
        # 3. 计算集群奖励（结合评分细则）
        # --------------------------
        # （1）损失比奖励（A）：鼓励击落蓝方，减少自身损失
        blue_destroyed = score_info.get("blue_destroyed", 0)
        red_destroyed = score_info.get("red_destroyed", 0)
        A = (blue_destroyed + 1) / (red_destroyed + 1)
        team_reward = 300 * (A / (len(self.blue_ids) + 1))  # 团队共享奖励
        # （2）攻击成功率奖励（B）：每架无人机单独计算
        red_hit_num = score_info.get("red_hit_num", 0)
        total_attacks = score_info.get("red_attack_num", 0)  # 假设接口返回攻击次数
        B = red_hit_num / total_attacks if total_attacks > 0 else 0
        for i in range(len(rewards)):
            rewards[i] += 200 * B / 4  # 平均分配团队奖励
        # （3）攻击效率奖励（C）：鼓励高频有效攻击
        if total_attacks > 0:
            C = total_attacks / (len(self.red_ids) * sim_time)
            sigmoid_C = 1 / (1 + np.exp(-C * 10))  # 缩放Sigmoid函数
            team_reward += 100 * sigmoid_C
        # （4）禁飞区惩罚（D）：全体受罚
        D = 禁飞区总时间 / (len(self.red_ids) * self.sim_time_limit)
        team_penalty = 1000 * D
        team_reward -= team_penalty
        # （5）地面目标奖励（E）：团队共享
        ground_destroyed = score_info.get("ground_destroyed", 0)  # 假设接口返回
        E = ground_destroyed / self.ground_target_count
        team_reward += 400 * E
        # 叠加团队奖励到个体
        for i in range(len(rewards)):
            rewards[i] += team_reward / len(self.red_ids)
        # --------------------------
        # 4. 终止条件
        # --------------------------
        terminated = False
        # 红方全毁或蓝方全毁
        if red_destroyed == 4 or blue_destroyed == 4:
            terminated = True
        # 步数超限或仿真时间达上限
        if self.episode_steps >= self.max_steps or sim_time >= self.sim_time_limit:
            terminated = True
        return observations, rewards, terminated, False, {}
    def _get_observations(self):
        """获取所有红方无人机的状态"""
        observations = []
        for red_id in self.red_ids:
            # 自身位置与速度
            pos = self.client.get_vehicle_pos()[red_id]
            vel = self.client.get_vehicle_vel()[red_id]
            # 可见蓝方目标距离（取最近4个）
            visible = self.client.get_visible_vehicles()
            blue_distances = []
            if red_id in visible:
                for blue_id in visible[red_id][:4]:  # 最多4个目标
                    blue_pos = self.client.get_vehicle_pos().get(blue_id, Position())
                    dist = self.client.get_distance_by_lon_lat(pos, blue_pos)
                    blue_distances.append(dist)
            # 不足4个用0填充
            while len(blue_distances) < 4:
                blue_distances.append(0.0)
            # 地面目标距离（简化为3个）
            ground_distances = [0.0, 0.0, 0.0]  # 实际需从接口获取地面目标坐标计算
            # 是否在禁飞区（从get_score接口的red_in_area_time_sec推导）
            in_no_fly = 1.0 if self.client.get_score().get("red_in_area_time_sec", 0) > 0 else 0.0
            # 剩余弹药（简化为1发）
            ammo_left = 1.0
            # 单架无人机的状态向量
            obs = np.array([
                pos.x, pos.y, pos.z,
                vel.vx, vel.vy, vel.vz,
                *blue_distances,
                *ground_distances,
                in_no_fly,
                ammo_left
            ], dtype=np.float32)
            observations.append(obs)
        return observations
    def render(self, mode="human"):
        """打印集群状态摘要"""
        if mode == "human":
            score = self.client.get_score()
            print(f"Step: {self.episode_steps}, 红方毁伤: {score.get('red_destroyed',0)}, "
                  f"蓝方毁伤: {score.get('blue_destroyed',0)}, 分数: {score.get('score',0):.1f}")
    def close(self):
        """关闭仿真环境"""
        self.client.stop()
# --------------------------
# 2. 初始化与训练
# --------------------------
if __name__ == "__main__":
    # 初始化客户端（使用用户提供的配置）
    config = {
        "id": 106,
        "config": RflysimEnvConfig(
            "172.19.95.113",  # 仿真服务器IP
            16001,            # 端口
            18001
        )
    }
    client = VehicleClient(id=config["id"], config=config["config"])
    # 创建集群环境
    env = DroneSwarmEnv(client)
    # 定义PPO算法（多智能体共享策略）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=4096,  # 增大采样步数以适应多智能体
        batch_size=128,
        n_epochs=10,
        gamma=0.98,  # 折扣因子略低，鼓励短期收益
        verbose=1,
        tensorboard_log="./swarm_rl_logs/"
    )
    # 保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path="./swarm_checkpoints/",
        name_prefix="swarm_ppo"
    )
    # 开始训练
    print("开始集群无人机训练...")
    model.learn(
        total_timesteps=200000,  # 总训练步数
        callback=checkpoint_callback,
        progress_bar=True
    )
    # 保存最终模型
    model.save("swarm_ppo_final")
    print("训练完成，模型已保存")
    env.close()