import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
import math
import time

class PedestrianMover(Node):
    def __init__(self):
        super().__init__('pedestrian_mover')
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 Gazebo 通信服务启动...')
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.start_time = time.time()

    def timer_callback(self):
        current_time = time.time() - self.start_time
        # 使用 cos 函数，使行人恰好从 y=3 的边缘起步，平滑横穿到 y=-3
        y_position = 3.0 * math.cos(0.5 * current_time) 

        request = SetEntityState.Request()
        request.state.name = 'dynamic_pedestrian'
        request.state.pose.position.x = 20.0  # 定位在公路 2/3 处
        request.state.pose.position.y = y_position
        request.state.pose.position.z = 0.5
        
        self.future = self.client.call_async(request)

def main():
    rclpy.init()
    node = PedestrianMover()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()