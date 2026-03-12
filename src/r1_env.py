import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState
from cv_bridge import CvBridge
import cv2
import threading
import time
import math
from rclpy.qos import qos_profile_sensor_data

class ROS2BridgeNode(Node):
    """专门负责与 ROS 2 世界打交道的后台节点 (充当 AI 的眼耳手脚)"""
    def __init__(self):
        super().__init__('rl_bridge_node')
        self.bridge = CvBridge()
        
        # 1. 眼睛：订阅 RGB 摄像头 (接入真正的灰度视神经)
        self.sub_cam = self.create_subscription(Image, '/camera/image_raw', self.cam_callback, qos_profile_sensor_data)
        
        # 2. 耳朵：订阅里程计 (获取小车当前的 X, Y 坐标和速度)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # 3. 手脚：发布速度指令控制底盘轮子
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 4. 上帝之手：调用 Gazebo 服务重置小车位置 (瞬间传送)
        self.client_reset = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # 状态缓存字典
        self.current_image = np.zeros((1, 64, 64), dtype=np.uint8)
        self.car_x = 3.0
        self.car_y = 0.0

    def cam_callback(self, msg):
        """视觉回调：把高清彩色图极度压缩为 64x64 黑白马赛克"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
            self.current_image = np.expand_dims(resized, axis=0) 
        except Exception as e:
            pass

    def odom_callback(self, msg):
        """里程计回调：实时更新坐标"""
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

class AutonomousCarEnv(gym.Env):
    """符合 Gymnasium 国际标准的强化学习环境桥梁"""
    def __init__(self):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
            
        self.ros_node = ROS2BridgeNode()
        self.executor_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.executor_thread.start()

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 根据 30米跑道和 0.5m/s 速度，放宽步数限制
        self.max_steps = 1000 
        self.current_step = 0
        
        # 新增：记录上一步的 X 坐标，用于计算真实位移
        self.previous_x = 3.0 

    def reset(self, seed=None, options=None):
        """每次小车撞毁、撞人或到达终点时，触发此函数重置世界"""
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_x = 3.0 # 重置起始X坐标
        
        twist = Twist()
        self.ros_node.pub_cmd.publish(twist)
        
        req = SetEntityState.Request()
        req.state.name = 'waffle_pi' # 确保名字对齐
        req.state.pose.position.x = 3.0
        req.state.pose.position.y = 0.0
        req.state.pose.position.z = 0.05
        req.state.pose.orientation.w = 1.0 
        
        if self.ros_node.client_reset.wait_for_service(timeout_sec=1.0):
            future = self.ros_node.client_reset.call_async(req)
            def reset_callback(f):
                pass # 隐藏成功的打印，保持终端清爽，失败再打印
            future.add_done_callback(reset_callback)
            
        time.sleep(0.5) 
        return self.ros_node.current_image, {}

    def step(self, action):
        """这是强化学习跳动的心脏！AI 每秒钟会调用这个函数约 10 次"""
        self.current_step += 1
        
        # 1. 动作提速！油门最大映射到 0.5 m/s
        throttle_real = float(np.clip(action[0], -1.0, 1.0) + 1.0) / 2.0 * 0.5  
        steering_real = float(np.clip(action[1], -1.0, 1.0)) * 1.0

        twist = Twist()
        twist.linear.x = throttle_real
        twist.angular.z = steering_real
        self.ros_node.pub_cmd.publish(twist)

        # 物理演进 0.1 秒
        time.sleep(0.1)

        # 2. 获取最新状态
        next_state = self.ros_node.current_image
        car_x = self.ros_node.car_x
        car_y = self.ros_node.car_y

        # 3. 核心价值观大换血 (Delta X Reward)
        reward = 0.0
        terminated = False
        truncated = False

        # 【核心奖励】：位移即正义！
        delta_x = car_x - self.previous_x
        # 乘以 10 是为了让单步得分看起来正常一点 (比如前进 0.05米，得分 0.5)
        reward += delta_x * 10.0 
        self.previous_x = car_x # 更新历史坐标

        # 【终极目标】：到达 28 米处，胜利结束！
        if car_x >= 28.0:
            reward += 200.0 # 给一个巨大的胜利奖金
            terminated = True
            print("🏆 【系统大捷】小车成功抵达终点线，回合胜利！")

        # 悬崖机制 (Y的绝对值大于3米即掉出公路)
        if abs(car_y) > 3.0:
            reward -= 100.0
            terminated = True
            print(f"【悲报】冲出边界 (Y={car_y:.2f})，坠入虚拟悬崖！")

        # 碰撞铁桶判定 (铁桶在 x=10, y=0)
        dist_to_barrel = math.hypot(car_x - 10.0, car_y - 0.0)
        if dist_to_barrel < 0.65:
            reward -= 100.0
            terminated = True
            print("💥 【悲报】撞击铁桶，回合结束！")

        # 超时判断
        if self.current_step >= self.max_steps:
            truncated = True
            print("⏳ 【提示】时间耗尽 (1000步)，未能到达终点，回合重置。")

        return next_state, reward, terminated, truncated, {}

    def close(self):
        """安全退出机制"""
        twist = Twist()
        self.ros_node.pub_cmd.publish(twist) 
        time.sleep(0.5) 
        rclpy.shutdown() 
        self.executor_thread.join(timeout=1.0)

if __name__ == '__main__':
    env = AutonomousCarEnv()
    print("环境已更新！现在测试【速度提升】和【直线撞桶】...")
    state, _ = env.reset()
    
    # 这次我们不让它瞎转圈了，强制让它踩死油门直走
    for i in range(200):
        # 动作: [油门全开(1.0), 方向盘打正(0.0)]
        action = [1.0, 0.0] 
        next_state, reward, done, truncated, info = env.step(action)
        
        # 每 10 步打印一次，避免刷屏太快看不清
        if i % 10 == 0:
            print(f"步数: {i+1}, X坐标: {env.ros_node.car_x:.2f}, 单步得分: {reward:.3f}")
        
        if done or truncated:
            print("--> 成功捕捉到回合结束信号，正在执行重置...")
            env.reset()
            time.sleep(2.0)
            break
            
    env.close()
    print("测试完美退出。")
