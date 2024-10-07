import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, CompressedImage
from cv_bridge import CvBridge
import cv2


class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('color_depth_analyzer')
        self.bridge = CvBridge()
        self.color_msg = None
        self.depth_msg = None

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # joint_sub = self.create_subscription(JointState, '/spot/joint_states', self.joint_callback, qos_profile)
        color_sub = self.create_subscription(CompressedImage, '/realsense/color_image/compressed', self.img_callback, qos_profile)


    def joint_callback(self, msg):
        # List of joints you're interested in
        joint_names_of_interest = [
            'arm0.sh0', 'arm0.sh1', 'arm0.hr0',
            'arm0.el0', 'arm0.el1', 'arm0.wr0',
            'arm0.wr1', 'arm0.f1x'
        ]
        
        # Create a dictionary to map joint names to positions
        joint_positions = dict(zip(msg.name, msg.position))
        
        # Print the positions of the specific joints
        for joint in joint_names_of_interest:  
            if joint in joint_positions:
                self.get_logger().info(f'{joint} position: {joint_positions[joint]}')
            else:
                self.get_logger().warn(f'{joint} not found in the current message')

    def img_callback(self, msg):
        try:
            # Convert the color and depth images to OpenCV format
            color_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            print(color_image.shape)
            cv2.imshow('color', color_image)
            cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error('Failed in processing frame: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)

    RealSense = ImgSubscriber()
    RealSense.get_logger().info('hello!')
    try:
        rclpy.spin(RealSense)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()  # Close OpenCV windows
        RealSense.destroy_node()

if __name__ == '__main__':
    main()
