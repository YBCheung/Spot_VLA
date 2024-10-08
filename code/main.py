from spot.spot_no_ros import SpotLoop
from camera.read_img import RealSenseCapture
from vla.openvla import openvla

import time
import threading
from PIL import Image

class Mission():
    def __init__(self):
        self.spot = SpotLoop()
        self.camera = RealSenseCapture()
        self.agent = openvla()
        self.thread_init()
        self.prompt = 'move the arm to the tape'

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

    def contril_loop(self):
        print("Control loop start")
        while True:
            start_time = time.time()
            # print(0, self.spot.move_spot_arm([0.95, 0, 0.230, 0,90,0])) # start pose
            img_np = self.camera.get_frame() # np img
            img_PIL = Image.fromarray(img_np)
            output_pose = self.agent.policy(self.prompt, img_PIL)
            print('out: ', output_pose)
            # print(4, self.spot.move_spot_arm([0.1, 0.4, -0.16, 0, 0, 0], offset=True))
            # print(5, self.move_spot_arm([-0.6, 0.1, 0, 0, 0, 0], offset=True), self.safe_info)

            freq = 1 / (time.time() - start_time)
            print(f'Frequency: {freq:.3f}Hz')
            # time_to_sleep = self.control_period_timer - (time.time() - start_time)
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)

    def state_loop(self):
        print("State loop start")
        while True:
            start_time = time.time()
            # check arm safety via state and img
            _, self.spot.safe, self.spot.safe_info = self.spot.safe_boundary(self.spot.get_arm_pose('hand'))
            print(self.spot.safe_info)
            self.camera.show_frame()
            time_to_sleep = self.state_period_timer - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
    def __exit__(self):
        self.spot.__exit__()
        self.camera.stop()
    
def main():
    m = Mission()
    try: 
        while m.control_thread.is_alive() and m.state_thread.is_alive():
            time.sleep(0.5)
        print('Error during running!')
        m.__exit__()
            
    except KeyboardInterrupt:
        m.__exit__()

if __name__ == '__main__':
    main()
