# run the recorded trajectory 
'''
0. change 'folder_path' to the target folder, save

1. run estop in a new terminal:
    conda activate openvla
    cd ~/openvla/spot-sdk/python/examples/estop
    export BOSDYN_CLIENT_USERNAME=rllab
    export BOSDYN_CLIENT_PASSWORD=robotlearninglab
    python estop_nogui.py 10.0.0.30 

2. run spot in a new terminal:
    conda activate openvla
    cd ~/openvla/code
    python arm_run_trajectory.py
'''

from spot._spot_no_ros import SpotLoop
from camera._read_img import RealSenseCapture
from dataset._load_recorded_data import lazy_data_loader, lazy_data_loader_delta
# from openvla._openvla import openvla
# from ssh_tunnel.server import send_data_to_server

import time
import threading
import cv2
import os
from PIL import Image

class Mission():
    def __init__(self):
        # self.camera = RealSenseCapture()
        # self.agent = openvla()
        self.spot = SpotLoop()
        self.thread_init()
        self.prompt = 'knock off the blue can'
        # self.prompt = 'lift the blue cube'
        print(f'prompt: {self.prompt}')

    def thread_init(self):
        self.stop_event = threading.Event()
        self.control_period_timer = 7
        self.state_period_timer = 0.5  # Timer period in seconds
        # Start moving the arm in a separate thread
        self.control_thread = threading.Thread(target=self.contril_loop, daemon=True)
        self.control_thread.start()
        # Start the info print loop in another thread
        # self.state_thread = threading.Thread(target=self.state_loop, daemon=True)
        # self.state_thread.start()
    def thread_stop(self):
        self.stop_event.set()

    def contril_loop(self):
        print("Control loop start")

        # Replace with your actual file paths
        folder_path = '/home/spot/docker/spot_optik_ctrl/ros2_ws/recorded_data/20250107_1736241365171415071'
        actual_pose_name = 'actual_poses.npz'
        rgb_video_name = 'rgb_video.avi'
        hand_video_name = 'rgb_hand_video.avi'

        npz_file_path = os.path.join(folder_path, actual_pose_name)
        video_file_path = os.path.join(folder_path, hand_video_name)

        loader = lazy_data_loader_delta(npz_file_path, video_file_path)
        for i, (time_delta, pose, frame) in enumerate(loader):

            if not self.stop_event.is_set():
                break

            # # to follow the recorded time intervals
            # start_time = time.time()

            # # Example: Process the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break

            # img_np = self.camera.get_frame() # np img
            self.spot.move_spot_arm(pose, offset=False, quat=True, seconds=0.3)

            # # to follow the recorded time intervals
            # time_to_sleep = time_delta - (time.time() - start_time)
            # # time_to_sleep = 0.1 - (time.time() - start_time)
            # print(i, pose)
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)
            # else:
            #     time.sleep(0.1)
            # cv2.destroyAllWindows()

    def state_loop(self):
        print("State loop start")
        while True:
            start_time = time.time()
            # check arm safety via state and img
            _, self.spot.safe, self.spot.safe_info = self.spot.safe_boundary(self.spot.get_arm_pose('hand'))
            if not self.spot:
                print(self.spot.safe_info)
            # self.camera.show_frame()
            time_to_sleep = self.state_period_timer - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            else:
                time.sleep(0.1)
    def __exit__(self):
        self.thread_stop()
        self.spot.__exit__()
        # self.camera.stop()
    
def main():
    m = Mission()
    try: 
        # while m.control_thread.is_alive() and m.state_thread.is_alive():
        while m.control_thread.is_alive():
            pass
        print('Error during running!')
        m.__exit__()
            
    except KeyboardInterrupt or Exception:
        m.__exit__()

if __name__ == '__main__':
    main()
