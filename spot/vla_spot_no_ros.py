from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (HAND_FRAME_NAME, get_a_tform_b)

import time
import threading

import pyrealsense2 as rs
import numpy as np
import cv2

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.api.gripper_command_pb2

from scipy.spatial.transform import Rotation
import numpy as np

class SpotLoop():

    def __init__(self):

        # Initialize spot:
        bosdyn.client.util.setup_logging()
        self.sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')

        # The network address of the robot needs to be
        # specified to reach it. This can be done with a DNS name
        # (e.g. spot.intranet.example.com) or an IP literal (e.g. 10.0.63.1)
        address = "10.0.0.30" #<-- CHANGE THIS TO CORRECT ONE
        self.robot = self.sdk.create_robot(address)

        # Clients need to authenticate to a robot before being able to use it.
        bosdyn.client.util.authenticate(self.robot)

        # Establish time sync with the robot. This kicks off a background thread to establish time sync.
        # Time sync is required to issue commands to the robot. After starting time sync thread, block
        # until sync is established.
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                        'such as the estop SDK example, to configure E-Stop.'

        # The robot state client will allow us to get the robot's state information, and construct
        # a command using frame information published by the robot.
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

        # Only one client at a time can operate a robot. Clients acquire a lease to
        # indicate that they want to control a robot. Acquiring may fail if another
        # client is currently controlling the robot. When the client is done
        # controlling the robot, it should return the lease so other clients can
        # control it. The LeaseKeepAlive object takes care of acquiring and returning
        # the lease for us.
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        # acquire lease
        try:
            self.lease = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
            self.lease.__enter__()
        except bosdyn.client.lease.ResourceAlreadyClaimedError:
            self.lease = self.lease_client.take()
            self.robot.logger.info("Lease forcely taken successfully.")

        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        self.robot.logger.info('Powering on robot... This may take a several seconds.')
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        self.robot.logger.info('Robot powered on.')
    
        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        self.robot.logger.info('Commanding robot to stand...')
        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info('Robot standing.')

        self.img_init()
        self.control_init()
        self.thread_init()

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        unstow_command_id = self.command_client.robot_command(unstow)
        self.robot.logger.info('Unstow command issued.')
        block_until_arm_arrives(self.command_client, unstow_command_id, 3.0)

    def control_init(self):
        # state: img
        # action[t] = arm_pose[t+T] - arm_pose[t], 6d pose difference. 
        # manually regulate range. But no angular boundary for current task. Attention to ry range!
        self.arm_pose = [0.,0.,0.,0.,0.,0.]
        self.safe_pose_command = [] # safe SE3Pose action for execute on spot.
        self.safe = True
        self.safe_info = 'safe'
    
    def img_init(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()

        # Configure the pipeline to stream the color data
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        print('img_start')

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if we got frames, just in case
        if not color_frame:
            print('wait_for_frame')
            return None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def show_frame(self):
        # Get the current frame
        frame = self.get_frame()

        # Show the frame if it's not None
        if frame is not None:
            cv2.imshow('RealSense Color Frame', frame)
            cv2.waitKey(1)  # This is necessary for OpenCV to process window events

    def img_stop(self):
        # Stop the pipeline and clean up
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def print_6d_pose(self, pose):
        if isinstance(pose, np.ndarray):
            print(f"arm pose_1: x: {pose[0]:.3f}, y: {pose[1]:.3f}, z: {pose[2]:.3f}, rx: {pose[3]:.3f}, ry: {pose[4]:.3f}, rz: {pose[5]:.3f}")
    
        elif isinstance(pose, bosdyn.client.math_helpers.SE3Pose): 
            pose = self.quat_2_euler(pose)
            print(f"arm pose_2: x: {pose[0]:.3f}, y: {pose[1]:.3f}, z: {pose[2]:.3f}, rx: {pose[3]:.3f}, ry: {pose[4]:.3f}, rz: {pose[5]:.3f}")
        return pose
    
    def get_arm_pose(self, data='hand_6d'):
        # return odom_T_flat_body quat, and update odom_hand self.arm_pose 6d pose. 

        robot_state = self.robot_state_client.get_robot_state()

        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        flat_body_T_hand = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
        # type bosdyn.client.math_helpers.Quat, can use normalize()
        self.arm_pose = self.quat_2_euler(flat_body_T_hand)
        if data == 'hand_6d':
            return self.arm_pose
        if data == 'hand':
            return flat_body_T_hand
        if data == 'body':
            return odom_T_flat_body

    def quat_2_euler(self, pose_7d):
        # Convert SE3pose (x, y, z, rx, ry, rz, w) to np.array Euler angles (in degrees)

        pos = np.array([
            pose_7d.position.x,
            pose_7d.position.y,
            pose_7d.position.z
        ])
        quat = np.array([
            pose_7d.rotation.x,
            pose_7d.rotation.y,
            pose_7d.rotation.z,
            pose_7d.rotation.w,
        ])
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        pose_6d = np.concatenate((pos, euler))
        return pose_6d
    
    def euler_2_quat(self, pose_6d):
        # Convert Euler angles (in degrees) to (x,y,z,w), return SE3Pose. 
        quat = Rotation.from_euler('xyz', pose_6d[3:], degrees=True).as_quat()
        return math_helpers.SE3Pose(x=pose_6d[0], y=pose_6d[1], z=pose_6d[2],
                                    rot=math_helpers.Quat(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))

    def safe_boundary(self, pose):
        top_l = (1.000, 0.300)
        bottom_l = (0.750, 0.140)
        top_r = (1.000, -0.300)
        bottom_r = (0.750, -0.140)
        z_range = (0.035, 0.230)
        x = pose.x
        y = pose.y
        z = pose.z
        safe = True
        safe_info = 'safe'

        if z < z_range[0]:
            safe_info = 'z to low'
            z = z_range[0]
            safe = False
        elif z > z_range[1]:
            safe_info = 'z to high'
            z = z_range[1]
            safe = False

        if x < bottom_l[0]:
            safe_info = 'x too_close'
            x = bottom_l[0]
            safe = False
        elif x > top_l[0]:
            safe_info = 'x too_far'
            x = top_l[0]
            safe = False
                
        y_l = ((bottom_l[1] - top_l[1]) / (bottom_l[0] - top_l[0])) * (x - top_l[0]) + top_l[1]
        y_r = ((bottom_r[1] - top_r[1]) / (bottom_r[0] - top_r[0])) * (x - top_r[0]) + top_r[1]
        if y >= y_l:
            safe_info = 'y too_left'
            y = y_l
            safe = False
        if y <= y_r:
            safe_info = 'y too right!'
            y = y_r
            safe = False

        if safe == False:
            pose.x = x
            pose.y = y
            pose.z = z
        return pose, safe, safe_info

    def move_spot_arm(self, pose_command = [1, 0, 0.15, 0., 0., 0.], seconds = 2, offset=False): 
        '''
        pos_command: delta [x,y,z] in m
        euler_command: delta [x,y,z] in degree
        seconds: duration in seconds>
        '''
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        # All are relevant difference. 
        safe = True

        pose_command_quat = self.euler_2_quat(pose_command)

        if offset==True:
            # q.pos = q1.pos + q2.pos
            # q.rot = q1.rot * q2.rot.
            # unknown reason, q != q1 * q2, leading to incorrect z value when rx continuously changes. 
            flat_body_T_hand_rot = self.get_arm_pose('hand').rotation * pose_command_quat.rotation
            flat_body_T_hand_pos = self.arm_pose[:3] + pose_command[:3]
            flat_body_T_hand = math_helpers.SE3Pose(x=flat_body_T_hand_pos[0], y=flat_body_T_hand_pos[1], z=flat_body_T_hand_pos[2], 
                                                     rot=flat_body_T_hand_rot)
            # flat_body_T_hand = self.get_arm_pose('hand') * pose_command_quat # have precision problems
            # print('rot', flat_body_T_hand_pos, flat_body_T_hand_rot, flat_body_T_hand, flat_body_T_hand_)
            
        else:
            flat_body_T_hand = pose_command_quat

        self.safe_pose_command, self.safe, self.safe_info = self.safe_boundary(flat_body_T_hand)

        odom_T_hand = self.get_arm_pose('body') * self.safe_pose_command
                         
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)
        print('From', end=' ')
        self.print_6d_pose(self.arm_pose)
        print('  To', end=' ')
        self.print_6d_pose(flat_body_T_hand)
        print(self.get_arm_pose(data='hand'))
        print(flat_body_T_hand)
        print(pose_command_quat)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # # Combine the arm and gripper commands into one RobotCommand
        # command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        cmd_id = self.command_client.robot_command(arm_command)

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(self.command_client, cmd_id)
        return safe
    

    def thread_init(self):
        self.stop_event = threading.Event()
        self.control_period_timer = 7
        self.state_period_timer = 0.5  # Timer period in seconds
        # Start moving the arm in a separate thread
        self.control_thread = threading.Thread(target=self.contril_loop, daemon=True)
        self.control_thread.start()
        # Start the info print loop in another thread
        self.state_thread = threading.Thread(target=self.state_loop, daemon=True)
        self.state_thread.start()

        # Keep threads running until stop_event is set
        # control_thread.join()
        # state_thread.join()

    def contril_loop(self):
        print("Control loop start")
        while True:
            start_time = time.time()
            print(0, self.move_spot_arm([0.95, 0, 0.230, 0,90,0]), self.safe_info) # start pose
            # output_pose = agent.policy(img, prompt)
            print(4, self.move_spot_arm([0.1, 0.4, -0.16, 0, 0, 0], offset=True), self.safe_info)
            # print(5, self.move_spot_arm([-0.6, 0.1, 0, 0, 0, 0], offset=True), self.safe_info)

            time_to_sleep = self.control_period_timer - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def state_loop(self):
        print("State loop start")
        while True:
            start_time = time.time()
            
            _, self.safe, self.safe_info = self.safe_boundary(self.get_arm_pose('hand'))
            print(self.safe_info, f"arm pose: x: {self.arm_pose[0]:.3f}, y: {self.arm_pose[1]:.3f}, z: {self.arm_pose[2]:.3f}, rx: {self.arm_pose[3]:.3f}, ry: {self.arm_pose[4]:.3f}, rz: {self.arm_pose[5]:.3f}")
            self.show_frame()
            time_to_sleep = self.state_period_timer - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def __exit__(self):
        print('stop')
        self.stop_event.set()
        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
        self.robot.logger.info('Robot safely powered off.')
        self.img_stop()

        # Release the lease when forcely take lease        
        if self.lease is not None:
            try:
                self.lease.return_lease()
            except Exception as e:
                print('release lease fail', e)


def main(args=None):
    spot = SpotLoop()
    try: 
        while spot.control_thread.is_alive() and spot.state_thread.is_alive():
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        spot.__exit__()

if __name__ == '__main__':
    main()

