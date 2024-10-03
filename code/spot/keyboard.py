'''
Date: 2024.05.18
Update:try to merge with ros image
'''

import bosdyn.client.util
from bosdyn.client.async_tasks import AsyncPeriodicGRPCTask
from bosdyn.geometry import EulerZXY
import pickle
import curses
import io
import logging
import signal
import threading
from PIL import Image
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.estop import EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, HAND_FRAME_NAME,
                                         get_a_tform_b, get_vision_tform_body)

from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, secs_to_hms

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, image_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, )
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import message_filters
from cv_bridge import CvBridge

import cv2
import numpy as np

import traceback
from datetime import datetime
import os
import time
from scipy import ndimage

LOGGER = logging.getLogger(__name__)

# Specify the task name
TASK_NAME = "Cover" # Curtain, Cover, Uncover

# Based on the task, to change the end-effector manipulation framen name
EE_FRAME_NAME = None
if TASK_NAME is "Curtain":
    EE_FRAME_NAME = "vision"
else:
    EE_FRAME_NAME = "body"
    
if TASK_NAME is "Cover":
    VELOCITY_BASE_SPEED = 0.45  # m/s
    VELOCITY_BASE_ANGULAR = 0.75  # rad/sec
    VELOCITY_CMD_DURATION = 0.4*2  # seconds
    COMMAND_INPUT_RATE = 0.1
else:
    VELOCITY_BASE_SPEED = 0.45  # m/s
    VELOCITY_BASE_ANGULAR = 0.75  # rad/sec
    VELOCITY_CMD_DURATION = 0.4  # seconds
    COMMAND_INPUT_RATE = 0.1

# Robot Arm Movement
VELOCITY_HAND_NORMALIZED = 0.5  # normalized hand velocity [0,1]
VELOCITY_ANGULAR_HAND = 1.0  # rad/sec


# Logs name
# datetime object containing current date and time
# dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
base_path = 'logs-'+TASK_NAME+'/' + dt_string  # logs-Curtain, logs-Cover, logs-Uncover
ACTION_LOG_PATH = base_path + '/'
STATE_LOG_PATH = base_path + '/state/'
IMG_LOG_PATH = base_path + '/image/'
IMG_RGB_LOG_PATH = base_path + '/image/rgb/'
IMG_DEP_LOG_PATH = base_path + '/image/dep/'

os.makedirs(ACTION_LOG_PATH)
os.makedirs(IMG_RGB_LOG_PATH, exist_ok=True)
os.makedirs(IMG_DEP_LOG_PATH, exist_ok=True)
os.makedirs(STATE_LOG_PATH, exist_ok=True)

UNLOCK = True  # if False to lock the arm movement in world coordinates
AI_CONTROL = False  # if False to human input as control

# Buffer to store some previous states/imgs
PREV_IMG = None
PREV_STATE = None

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)


class ExitCheck(object):
    """A class to help exiting a loop, also capturing SIGTERM to exit the loop."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit (rather than sigterm/sigint)."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return the status of the exit checker indicating if it should exit."""
        return self._kill_now


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self, CompressedImage, '/realsense/color_image/compressed',
                                                  qos_profile=qos_profile)  # /realsense/color_image/compressed CompressedImage
        self.depth_sub = message_filters.Subscriber(self, Image, '/realsense/depth_image', qos_profile=qos_profile)
        # self._fram_num = 0
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 1, 0.5)
        self.ts.registerCallback(self.callback)
        self.rgb = None
        self.dep = None

        # self.subscription

    # prevent unused variable warning

    def callback(self, rgb, depth):
        cv_image_rgb = self.bridge.compressed_imgmsg_to_cv2(rgb, "bgr8")
        # cv2.imwrite(IMG_LOG_PATH+f'image_{self._fram_num}_rgb.jpg', cv_image_rgb)
        self.rgb = cv_image_rgb

        # depth process
        cv_image_dep = self.bridge.imgmsg_to_cv2(depth)  # uint8 (480, 640),max :253,min : 0
        # depth_rgb = cv2.cvtColor(cv_image_dep, cv2.COLOR_GRAY2RGB)
        # cv2.imwrite(IMG_LOG_PATH+f'image_{self._fram_num}_dep.jpg', depth_rgb)
        self.dep = cv_image_dep
        # print(f'reviced {self._fram_num} image')
        # self._fram_num = self._fram_num+1


class CursesHandler(logging.Handler):
    """logging handler which puts messages into the curses interface"""

    def __init__(self, arm_wasd_interface):
        super(CursesHandler, self).__init__()
        self._arm_wasd_interface = arm_wasd_interface

    def emit(self, record):
        msg = record.getMessage()
        msg = msg.replace('\n', ' ').replace('\r', '')
        self._arm_wasd_interface.add_message(f'{record.levelname:s} {msg:s}')


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()


key = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'w': 4, 'a': 5, 's': 6, 'd': 7,
    'R': 8, 'F': 9,
}
key_map = list(key.keys())


class WasdInterface():
    """A curses interface for driving the robot."""

    def __init__(self, robot):
        self._robot = robot
        rclpy.init()
        self._img_sub = ImageSubscriber()

        # Create clients -- do not use the for communication yet.
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            # Not the estop.
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self.frame_num = 0

        self.action = 'nope'
        self.ob_buff = []

        self._lock = threading.Lock()
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            # ord('x'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('P'): self._toggle_power,
            ord('p'): self._toggle_power,
            ord('c'): self._sit,
            ord('f'): self._stand,
            ord('w'): self._body_forward,
            ord('s'): self._body_backward,
            ord('a'): self._body_left,
            ord('d'): self._body_right,
            ord('W'): self._arm_cartesian_move_out,
            ord('S'): self._arm_cartesian_move_in,
            ord('A'): self._rotate_cartesian_ccw,
            ord('D'): self._rotate_cartesian_cw,
            ord('R'): self._arm_move_up,
            ord('F'): self._arm_move_down,
            # ord('I'): self.collect_data,  #self._rotate_plus_ry, # check if we need it ir not
            # ord('K'): self.save_data, # self._rotate_minus_ry, # check if we need it ir not
            # ord('U'): self._rotate_plus_rx, # check if we need it ir not
            # ord('O'): self._rotate_minus_rx, # check if we need it ir not
            # ord('J'): self._rotate_plus_rz,
            # ord('L'): self._rotate_minus_rz,
            #ord('N'): self._toggle_gripper_open,
            #ord('M'): self._toggle_gripper_closed,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            # ord('i'): self._toggle_image_capture,
            ord('y'): self._unstow,
            ord('h'): self._stow,
            ord('.'): self.reset_AI_CONTROL,

        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None

        self.action_handler_logger = self._setup_file_logger(loggerName="action_logger",
                                                             fileName=ACTION_LOG_PATH + "action_log.log")
        self.robot_state_handler_logger = self._setup_file_logger(loggerName="robot_state_logger",
                                                                  fileName=STATE_LOG_PATH + "robot_state_log.log")

    def collect_data(self):
        rgb, dep = self.get_image()
        self.bn.append((rgb, dep))

    def save_data(self):
        with open(IMG_LOG_PATH + 'ref.pkl', 'wb') as f:
            pickle.dump(self.bn, f)

    def start(self):
        """Begin communication with the robot."""
        # Construct our lease keep-alive object, which begins RetainLease calls in a thread.
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                               return_at_exit=True)

        self._robot_id = self._robot.get_id()
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup(
            )  # Set this endpoint as the robot's sole estop.

    def shutdown(self):
        """Release control of robot as gracefully as possible."""
        LOGGER.info('Shutting down WasdInterface.')
        if self._estop_keepalive:
            # This stops the check-in thread but does not stop the robot.
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    def flush_and_estop_buffer(self, stdscr):
        """Manually flush the curses input buffer but trigger any estop requests (space)"""
        key = ''
        while key != -1:
            key = stdscr.getch()
            if key == ord(' '):
                self._toggle_estop()
                self._toggle_power()

    def add_message(self, msg_text):
        """Display the given message string to the user in the curses interface."""
        with self._lock:
            self._locked_messages = [msg_text] + self._locked_messages[:-1]

    def message(self, idx):
        """Grab one of the 3 last messages added."""
        with self._lock:
            return self._locked_messages[idx]

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    def drive(self, stdscr):
        """User interface to control the robot via the passed-in curses screen interface object."""
        global PREV_IMG
        global AI_CONTROL

        with ExitCheck() as self._exit_check:
            curses_handler = CursesHandler(self)
            curses_handler.setLevel(logging.INFO)
            LOGGER.addHandler(curses_handler)

            stdscr.nodelay(True)  # Don't block for user input.
            stdscr.resize(26, 140)
            stdscr.refresh()

            # for debug
            curses.echo()
            rgb = None
            dep = None

            try:
                while not self._exit_check.kill_now:
                    self._robot_state_task.update()
                    self._drive_draw(stdscr, self._lease_keepalive)
                    try:
                        cmd = stdscr.getch()
                        if AI_CONTROL:
                            if cmd == ord('.'):
                                AI_CONTROL = False
                                self.flush_and_estop_buffer(stdscr)
                                self._safe_power_off()
                                time.sleep(2.0)
                                break
                            else:
                                True

                        # Do not queue up commands on client
                        self.flush_and_estop_buffer(stdscr)
                        self._drive_cmd(cmd, rgb, dep)
                        time.sleep(2.0)
                    except Exception:
                        # On robot command fault, sit down safely before killing the program.
                        print(traceback.format_exc())
                        self._safe_power_off()
                        time.sleep(2.0)
                        raise


            finally:
                LOGGER.removeHandler(curses_handler)

    def _drive_draw(self, stdscr, lease_keep_alive):
        """Draw the interface screen at each update."""
        stdscr.clear()  # clear screen
        stdscr.resize(28, 140)
        stdscr.addstr(0, 0, f'{self._robot_id.nickname:20s} {self._robot_id.serial_number}')
        stdscr.addstr(1, 0, self._lease_str(lease_keep_alive))
        stdscr.addstr(2, 0, self._battery_str())
        stdscr.addstr(3, 0, self._estop_str())
        stdscr.addstr(4, 0, self._power_state_str())
        stdscr.addstr(5, 0, self._time_sync_str())
        for i in range(3):
            stdscr.addstr(7 + i, 2, self.message(i))
        stdscr.addstr(10, 0, 'Commands: [TAB]: quit                                  ')
        stdscr.addstr(11, 0, '          [T]: Time-sync, [SPACE]: Estop, [P]: Power   ')
        stdscr.addstr(12, 0, '          [i]: Take image , [O]: Video mode N/A            ')
        stdscr.addstr(13, 0, '          [f]: Stand,                ')
        stdscr.addstr(14, 0, '          [c]: Sit,                ')
        stdscr.addstr(15, 0, '          [y]: Unstow arm, [h]: Stow arm               ')
        stdscr.addstr(16, 0, '          [wasd]: Directional strafing                 ')
        stdscr.addstr(17, 0, '          [WASD]: Arm Radial/Azimuthal control         ')
        stdscr.addstr(18, 0, '          [RF]: Up/Down control         ')
        stdscr.addstr(19, 0, '          [UO]: X-axis rotation control             ')
        #stdscr.addstr(20, 0, '          [IK]: Y-axis rotation control             ')
        #stdscr.addstr(21, 0, '          [JL]: Z-axis rotation control             ')
        #stdscr.addstr(22, 0, '          [NM]: Open/Close gripper                  ')
        stdscr.addstr(20, 0, '          [qe]: Body Turning, [ESC]: Stop                   ')
        stdscr.addstr(21, 0, '          [l]: Return/Acquire lease                  ')
        stdscr.addstr(22, 0, '')

        global AI_CONTROL
        global UNLOCK

        if AI_CONTROL:
            stdscr.addstr(23, 0, f'Now AI control mode, {self.frame_num}th action is : {self.action}')
        else:
            stdscr.addstr(23, 0, 'Now Human control mode')

        if UNLOCK:
            stdscr.addstr(24, 0, f'')
        else:
            stdscr.addstr(24, 0, 'Now hand is locked in space')

            # print as many lines of the image as will fit on the curses screen
        '''if self._image_task.ascii_image != None:
                                    max_y, _max_x = stdscr.getmaxyx()
                                    for y_i, img_line in enumerate(self._image_task.ascii_image):
                                        if y_i + 17 >= max_y:
                                            break

                                        stdscr.addstr(y_i + 17, 0, img_line)'''

        stdscr.refresh()

    def _drive_cmd(self, key, rgb=None, dep=None):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            if (key not in [ord(' '), ord('\t')]):
                self._get_obs(key, rgb, dep)
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _get_obs(self, key=None, rgb=None, dep=None):
        timestamp = "%.20f" % time.time()
        arm_filename = STATE_LOG_PATH + str(timestamp) + '.pkl'
        state = self._robot_state_client.get_robot_state()
        with open(arm_filename, 'wb') as f:
            pickle.dump(state, f)
        act_info = timestamp + '-' + chr(key)
        self.action_handler_logger.info(act_info)
        if rgb is None:
            rgb, dep = self.get_image()

        # image = {'dep':dep,'rgb':rgb}

        img_filename_rgb = IMG_RGB_LOG_PATH + str(timestamp) + '_rgb.jpg'
        img_filename_dep = IMG_DEP_LOG_PATH + str(timestamp) + '_dep.jpg'
        cv2.imwrite(img_filename_rgb, rgb)
        cv2.imwrite(img_filename_dep, dep)
        return rgb, dep

    def get_image(self):
        self._img_sub.dep = None
        while self._img_sub.dep is None:
            rclpy.spin_once(self._img_sub)
        dep = self._img_sub.dep
        rgb = self._img_sub.rgb
        return rgb, dep

    def _try_grpc(self, desc, thunk):
        try:
            cmd_id = thunk()
            self.robot_state_handler_logger.info(self.robot_state)
            return cmd_id
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):

        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self.add_message(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)

    def _quit_program(self):
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()

    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                       return_at_exit=True)
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            return self._robot_command_client.robot_command(command=command_proto,
                                                            end_time_secs=end_time_secs)

        return self._try_grpc(desc, _start_command)

    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())
        # self._start_robot_command('open_gripper', RobotCommandBuilder.claw_gripper_open_command())
        self._move_arm_position()
        
        global TASK_NAME 
        if TASK_NAME is 'Cover':
            self._arm_move_up()

    def _body_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _body_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _body_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _body_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _arm_move_out(self):
        self._arm_cylindrical_velocity_cmd_helper('move_out', v_r=VELOCITY_HAND_NORMALIZED)

    def _arm_move_in(self):
        self._arm_cylindrical_velocity_cmd_helper('move_in', v_r=-VELOCITY_HAND_NORMALIZED)

    def _rotate_ccw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)

    def _rotate_plus_rx(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)

    def _arm_cartesian_move_out(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_out_cartesian', v_x=0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_out', v_r=VELOCITY_HAND_NORMALIZED)

    def _arm_cartesian_move_in(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_in_cartesian', v_x=-0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_in', v_r=-VELOCITY_HAND_NORMALIZED)

    def _rotate_cartesian_ccw(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('rotate_ccw_cartesian', v_y=0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cartesian_cw(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('rotate_cw_cartesian', v_y=-0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)

    def _arm_move_up(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_up', v_z=VELOCITY_HAND_NORMALIZED)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_up', v_z=VELOCITY_HAND_NORMALIZED)

    def _arm_move_down(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_down', v_z=-VELOCITY_HAND_NORMALIZED)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_down', v_z=-VELOCITY_HAND_NORMALIZED)

    def _toggle_gripper_open(self):
        self._start_robot_command('open_gripper', RobotCommandBuilder.claw_gripper_open_command())

    def _toggle_gripper_closed(self):
        self._start_robot_command('close_gripper', RobotCommandBuilder.claw_gripper_close_command())

    def reset_AI_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = True
        print('now AI control')

    def reset_HUMAN_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = False

        print('now Human control')

    def _move_arm_position(self, x=0.3, y=0, z=-0.2, qw=1, qx=0, qy=0, qz=0):  # change -0.25 to 0.2
        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         BODY_FRAME_NAME, HAND_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, BODY_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        # gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(arm_command)
        # robot_command.synchronized_command.arm_command
        # Send the request
        self._start_robot_command('move_arm', command)

    def _arm_cylindrical_velocity_cmd_helper_LOCK(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_angular_velocity_cmd_helper_LOCK(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_angular_velocity_cmd_helper(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(frame_name=EE_FRAME_NAME)

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _arm_cartesian_velocity_cmd_helper(self, desc, v_x=0.0, v_y=0.0, v_z=0.0):

        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(frame_name=EE_FRAME_NAME)
        cartesian_velocity.velocity_in_frame_name.x = v_x
        cartesian_velocity.velocity_in_frame_name.y = v_y
        cartesian_velocity.velocity_in_frame_name.z = v_z
        # LOGGER.info(cartesian_velocity.frame_name)
        
        global TASK_NAME 
        global VELOCITY_CMD_DURATION
        
        #if TASK_NAME is 'Cover':
        #    VELOCITY_CMD_DURATION = VELOCITY_CMD_DURATION / 2.
            
        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity(frame_name=EE_FRAME_NAME)
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _arm_velocity_cmd_helper(self, arm_velocity_command, desc=''):

        # Build synchronized robot command
        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command)

        return self._start_robot_command(desc, robot_command,
                                         end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _stow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())

    def _unstow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_ready_command())

    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            self.add_message('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            self._try_grpc('powering-off', self._safe_power_off)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self, lease_keep_alive):
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except bosdyn.client.lease.Error:
                lease = '...'
            if lease_keep_alive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return f'Lease {lease} THREAD:{alive}'

    def _power_state_str(self):
        power_state = self._power_state()
        if power_state is None:
            return ''
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f'Power: {state_str[6:]}'  # get rid of STATE_ prefix

    def _estop_str(self):
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
                    break
        return f'Estop {estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        if not self._robot.time_sync:
            return 'Time sync: (none)'
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            if skew:
                skew_str = f'offset={duration_str(skew)}'
            else:
                skew_str = '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
        return f'Time sync: {status} {skew_str}'

    def _battery_str(self):
        if not self.robot_state:
            return ''
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        if battery_state.charge_percentage.value:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = f'|{"=" * bar_len}{" " * (10 - bar_len)}|'
        else:
            bat_bar = ''
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
        return f'Battery: {status}{bat_bar} {time_left}'

    def _setup_file_logger(self, loggerName, fileName, level=logging.INFO):
        handler = logging.FileHandler(fileName)
        log_formatter = logging.Formatter('%(message)s')  # ('%(created)f - [[[%(message)s]]]')
        handler.setFormatter(log_formatter)

        fileHandlerLogger = logging.getLogger(loggerName)
        fileHandlerLogger.setLevel(level)
        fileHandlerLogger.addHandler(handler)

        return fileHandlerLogger


def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file wasd.log for later debugging.
    file_handler = logging.FileHandler('customized_remote_control.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # The stream handler is useful before and after the application is in curses-mode.
    if verbose:
        stream_level = logging.DEBUG
    else:
        stream_level = logging.INFO

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(log_formatter)
    LOGGER.addHandler(stream_handler)
    return stream_handler


def main():
    # rclpy.init()
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--time-sync-interval-sec',
                        help='The interval (seconds) that time-sync estimate should be updated.',
                        type=float)
    options = parser.parse_args()

    stream_handler = _setup_logging(options.verbose)

    # Create robot object.
    sdk = create_standard_sdk('WASDClient')
    robot = sdk.create_robot(options.hostname)
    try:
        robot.authenticate('rllab', 'robotlearninglab')
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(options.time_sync_interval_sec)
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    wasd_interface = WasdInterface(robot)

    try:
        wasd_interface.start()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed to initialize robot communication: %s', err)
        return False

    LOGGER.removeHandler(stream_handler)  # Don't use stream handler in curses mode.

    try:
        try:
            # Prevent curses from introducing a 1 second delay for ESC key
            os.environ.setdefault('ESCDELAY', '0')
            # Run wasd interface in curses mode, then restore terminal config.
            curses.wrapper(wasd_interface.drive)
        finally:
            # Restore stream handler to show any exceptions or final messages.
            LOGGER.addHandler(stream_handler)
    except Exception as e:
        LOGGER.error('WASD has thrown an error: [%r] %s', e, e, exc_info=True)
    finally:
        # Do any final cleanup steps.
        wasd_interface.shutdown()

    return True


if __name__ == '__main__':
    main()

