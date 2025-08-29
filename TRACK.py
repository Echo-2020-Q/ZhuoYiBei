# 状态空间（64维：4架×16维） 红方一共四架无人机：RED_IDS = [10091, 10084, 10085, 10086]
        boundary_width = self.boundary_area['max_x'] - self.boundary_area['min_x']  #区域的宽
        boundary_height = self.boundary_area['max_y'] - self.boundary_area['min_y'] #区域的高
        max_boundary_distance = max(boundary_width, boundary_height)
        single_obs_low = np.array([
                0, 0, 0, 0,              # 到四个边界的距离：左、右、下、上  这个需要算根据经纬度进行计算，飞行区域为矩形经 四个顶点为[101.07326442811694, 40.599429229677526], [103.08242360888715, 40.599429229677526], [103.08242360888715, 39.558295557025474], [101.07326442811694, 39.558295557025474]
                -200, -200, -200,        # 这个红方无人机的实际速度(一共4架)（vx/vy/vz） get_vehicle_vel()
#功通过这个函数获得
    # 获取载具速度信息
    #
    # 输入:
    #
    #     无
    #
    # 输出 :
    #
    #     Dict:
    #
    #         key: 载具编号
    #
    #         value: rflysim.Vel类
    #
    #             vel.vx 正北方向速度分量
    #
    #             vel.vy 正东方向速度分量
    #
    #             vel.vz 地方向速度分量
    #
    #             vel.direct 水平速度大小
    #
    #             vel.rate 水平速度方向，0 - 360，0表示正北，顺时针旋转
    #
    # 参考示例 :
    #
    #     vel_all = client.get_vehicle_vel()
    #
    #     for id, vel in vel_all.items():
    #
    #         print(f”{id}: vx: {vel.vx}, vy: {vel.vy}, vz: {vel.vz}”)


                0, 0, 0, 0,              # 最近4个蓝方距离  计算观测到的四架蓝方飞机的距离，按照升序排列，若蓝方飞机没发现，则距离就填最大值
                0,                       # 最近蓝方速度（标量）    可以通过get_visible_vehicles获得，输出事例：
# 10087: id: 10086
# update_time_ms: 615
# target_id: 10087
# target_speed: 118.0
# target direction: -2.0
# reliability: 1.0
# inves_platform: 3
# track_state: 1
# side_identify: 1
# side: SIDE RED
# target_pos {
# 101.44442112287712
# X：
# 40.014447953213086
# y：
# -0.0
# Z：-0.0
#其中target——speed便是
                0,                       # 最近蓝方方向（0-360度）  蓝方的飞机距离最近的红方飞机的方向角(从红方来看) 0表示正北，顺时针方向
                -self.jam_area["threshold"] if self.jam_area else -10000,  # 到干扰区边界距离（负值表示在干扰区内）  干扰区：禁飞区圆形，圆心：101.96786384206956, 40.2325圆上另一点：101.84516211958464,40.2325半径：10.416 km约等于10,416 m
                1                        # 剩余攻击次数  每架红机最大攻击次数是八次，成功射击一次就减少一次
            ])
        single_obs_high = np.array([
                max_boundary_distance, max_boundary_distance,  # 左、右边界距离上限
                max_boundary_distance, max_boundary_distance,  # 下、上边界距离上限
                200, 200, 200,           # 实际速度上限 m/s
                70000, 70000, 70000, 70000,  # 蓝方距离上限 m 大约是雷达探测距离
                300,                     # 蓝方速度上限
                360,                     # 蓝方方向上限（0-360度）
                max_boundary_distance,   # 到干扰区边界距离上限（正值表示在干扰区外）
                self.num_missile         # 攻击次数上限
            ])
        self.observation_space = spaces.Box(
            low=np.tile(single_obs_low, self.num_red_drones),
            high=np.tile(single_obs_high, self.num_red_drones),
            dtype=np.float64
        )

        # 动作空间：适配mode=0（航向+速度+高度）
        single_action_low = np.array([
            50,    # rate下限：固定翼最小速度  红方无人机的速度 可以通过get_vehicle_vel()获得
    # 获取载具速度信息get_vehicle_vel()
    #
    # 输入:
    #
    #     无
    #
    # 输出 :
    #
    #     Dict:
    #
    #         key: 载具编号
    #
    #         value: rflysim.Vel类
    #
    #             vel.vx 正北方向速度分量
    #
    #             vel.vy 正东方向速度分量
    #
    #             vel.vz 地方向速度分量
    #
    #             vel.direct 水平速度大小
    #
    #             vel.rate 水平速度方向，0 - 360，0表示正北，顺时针旋转
    #
    # 参考示例 :
    #
    #     vel_all = client.get_vehicle_vel()
    #
    #     for id, vel in vel_all.items():
    #
    #         print(f”{id}: vx: {vel.vx}, vy: {vel.vy}, vz: {vel.vz}”)


            0,     # direct下限：航向0度 航向角可以通过get_vehicle_vel()获得
    # 获取载具速度信息get_vehicle_vel()
    #
    # 输入:
    #
    #     无
    #
    # 输出 :
    #
    #     Dict:
    #
    #         key: 载具编号
    #
    #         value: rflysim.Vel类
    #
    #             vel.vx 正北方向速度分量
    #
    #             vel.vy 正东方向速度分量
    #
    #             vel.vz 地方向速度分量
    #
    #             vel.direct 水平速度大小
    #
    #             vel.rate 水平速度方向，0 - 360，0表示正北，顺时针旋转
    #
    # 参考示例 :
    #
    #     vel_all = client.get_vehicle_vel()
    #
    #     for id, vel in vel_all.items():
    #
    #         print(f”{id}: vx: {vel.vx}, vy: {vel.vy}, vz: {vel.vz}”)
            10,    # vz下限：最低高度 高度可以通过get_vehicle_vel()获得
    # 获取载具速度信息get_vehicle_vel()
    #
    # 输入:
    #
    #     无
    #
    # 输出 :
    #
    #     Dict:
    #
    #         key: 载具编号
    #
    #         value: rflysim.Vel类
    #
    #             vel.vx 正北方向速度分量
    #
    #             vel.vy 正东方向速度分量
    #
    #             vel.vz 地方向速度分量
    #
    #             vel.direct 水平速度大小
    #
    #             vel.rate 水平速度方向，0 - 360，0表示正北，顺时针旋转
    #
    # 参考示例 :
    #
    #     vel_all = client.get_vehicle_vel()
    #
    #     for id, vel in vel_all.items():
    #
    #         print(f”{id}: vx: {vel.vx}, vy: {vel.vy}, vz: {vel.vz}”)
            0      # 攻击指令：0=不攻击 在仿真时间的t-1与t之前这段时间是否成功执行了设置攻击目标的指令 成功则为1
        ])
        single_action_high = np.array([
            200,   # rate上限：固定翼最大速度
            360,   # direct上限：航向360度
            100,   # vz上限：最高高度
            1      # 攻击指令：1=攻击
        ])
        self.action_space = spaces.Box(
            low=np.tile(single_action_low, self.num_red_drones),  # 4架×4维=16维
            high=np.tile(single_action_high, self.num_red_drones),
            dtype=np.float64
        )