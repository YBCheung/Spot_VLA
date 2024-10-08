
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
import cv2
import message_filters
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import open3d as o3d
from .pointnet_bending_clas import get_model
import torch
from .pointcloud_helper_functions import *

def load_model(model_path):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = get_model().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.device = device
    return model

def predict_bending(model, points):
    # convert points to float32
    points = np.array(points).astype(np.float32)

    #rotated_point = create_rotated_point_clouds(points, 5)

    points = torch.tensor(points, device=model.device)

    model.eval()
    with torch.no_grad():
        outputs, _ = model(points)  # Assuming the model expects batch dimension
        outputs = torch.sigmoid(outputs)
        #print(outputs)
        #print(outputs.mean())
    return outputs.mean()

def create_pointcloud2_from_open3d(pcd, header):
    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points, dtype=np.float32)

    # Create PointCloud2 message
    msg = PointCloud2()
    msg.header = header
    msg.header.frame_id = "hand"
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = points.tobytes()

    return msg


def overlay_images(color_image, depth_image):
    """
    Overlays a depth image onto a color image using OpenCV.

    Parameters:
    - color_image (numpy.ndarray): The RGB color image.
    - depth_image (numpy.ndarray): The depth image.

    Returns:
    - overlay_image (numpy.ndarray): The resulting image with depth overlaid on color.
    """

    # Ensure depth image is in 8-bit format for visualization
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_colored = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Create an overlay image: blend color image and depth image
    overlay_image = cv2.addWeighted(color_image, 0.3, depth_image_colored, 0.7, 0)

    return overlay_image

def display_images(color_image, depth_image):
    """
    Displays color and depth images overlaid using OpenCV.

    Parameters:
    - color_image (numpy.ndarray): The RGB color image.
    - depth_image (numpy.ndarray): The depth image.
    """

    # Overlay the images
    overlay_image = overlay_images(color_image, depth_image)

    # Display the overlay image
    cv2.imshow('Color and Depth Overlay', overlay_image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()


class ColorDepthAnalyzer(Node):
    def __init__(self):
        super().__init__('color_depth_analyzer')
        self.bridge = CvBridge()
        self.color_msg = None
        self.depth_msg = None

        color_sub = message_filters.Subscriber(self, CompressedImage, '/realsense/color_image/compressed', qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, '/realsense/depth_image', qos_profile=qos_profile_sensor_data)

        #color_sub = message_filters.Subscriber(self, CompressedImage, '/realsense/color_image/compressed', self.color_callback, qos_profile=qos_profile_sensor_data)
        #depth_sub = message_filters.Subscriber(self, '/realsense/depth_image', self.depth_callback, qos_profile=qos_profile_sensor_data)

        self.point_pub = self.create_publisher(PointCloud2, '/miikka/point_cloud', qos_profile=qos_profile_sensor_data)
        self.fake_point_pub = self.create_publisher(PointCloud2, '/miikka/fake_point_cloud', qos_profile=qos_profile_sensor_data)

        # Synchronize the messages with approximate time policy
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.sync_callback)

        self.is_noodle_bending = False
        self.model_path = "miikka_package/miikka_package/best_model.pth"
        self.model = load_model(self.model_path)

        #self.timer = self.create_timer(0.1, self.timer_callback)  # Process frames every 2 seconds
        self.process_frame = False
        self.mask = None

    def sync_callback(self, color_msg, depth_msg):
        self.color_msg = color_msg
        self.depth_msg = depth_msg
        self.process_frame = True
        self.timer_callback()

    def timer_callback(self):
        if self.process_frame:
            self.process_frame = False
            try:
                # Convert the color and depth images to OpenCV format
                color_image = self.bridge.compressed_imgmsg_to_cv2(self.color_msg, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='passthrough')

                self.mask = create_noodle_segmentation_mask(color_image)
                self.fake_mask = np.ones_like(self.mask)

                cv2.imshow("mask", self.mask)
                cv2.waitKey(1)

                #display_images(color_image, depth_image)


                if self.mask is not None:

                    #pcd2 = create_point_cloud_from_depth(depth_image, color_image, self.fake_mask)
                    pcd = create_point_cloud_from_depth(depth_image, color_image, self.mask)

                    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
                    #pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points) / 1000.0)

                    # calculate average distance from origin
                    #avg_dist = np.mean(np.linalg.norm(np.asarray(pcd.points), axis=1))
                    #print("Average distance from origin: ", avg_dist)

                    # Publish the point cloud
                    #self.point_pub.publish(create_pointcloud2_from_open3d(pcd2, self.color_msg.header))
                    #self.fake_point_pub.publish(create_pointcloud2_from_open3d(pcd, self.color_msg.header))

                    # random sample the point cloud to n_points
                    n_points = 64
                    if len(pcd.points) >= n_points:

                        multisampled = multisample_pcd(pcd, n_points, add_noise=False, n_rotations=64)
                        multisampled_pcd = o3d.geometry.PointCloud()
                        multisampled_pcd.points = o3d.utility.Vector3dVector(multisampled[0])
                        self.fake_point_pub.publish(create_pointcloud2_from_open3d(multisampled_pcd, self.color_msg.header))

                        #multisampled = [np.asarray(pcd.points)]

                        if len(multisampled) < 1:
                            #print("not enough samples: ", len(multisampled))
                            return

                        # predict the displacement
                        is_bending = predict_bending(self.model, multisampled)

                        print(is_bending)

                        #print("IS BENDING:", is_bending)

                    else:
                        print(f"Skipping point cloud with less than n_points. Found {len(pcd.points)} points")
                else:
                    print("mask not made")
            except Exception as e:
                self.get_logger().error('Failed in processing frame: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    node = ColorDepthAnalyzer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()