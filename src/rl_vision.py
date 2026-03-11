import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        
        # 核心改变：订阅极其健康的普通彩色相机频道
        self.subscription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.listener_callback, 
            qos_profile_sensor_data)
            
        self.get_logger().info('RGB 视觉感知节点已启动，正在接收并压缩画面...')

    def listener_callback(self, msg):
        try:
            # 1. 提取原始彩色图像 (bgr8 格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 2. 灰度化处理：自动驾驶纯视觉基建
            # 将 3 通道的彩色图转化为 1 通道的黑白灰度图，大幅降低神经网络参数量
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 3. 极度降维压缩：将高分辨率画面压缩为 64x64 的状态矩阵
            state_matrix = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
            
            # 4. 可视化放大 (仅供人类观察)
            # 把 64x64 的马赛克图放大到 400x400
            display_img = cv2.resize(state_matrix, (400, 400), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("RL Agent Vision (64x64 Grayscale)", display_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"处理图像时发生错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()