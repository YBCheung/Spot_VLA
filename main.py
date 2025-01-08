from spot._spot_no_ros import SpotLoop
from camera._read_img import RealSenseCapture
from openvla._openvla import openvla
# from ssh_tunnel.server import send_data_to_server

import time
import threading
from PIL import Image

class Mission():
    def __init__(self):
        self.camera = RealSenseCapture()
        self.agent = openvla()
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
        self.state_thread = threading.Thread(target=self.state_loop, daemon=True)
        self.state_thread.start()
    def thread_stop(self):
        self.stop_event.set()

    def contril_loop(self):
        print("Control loop start")
        while not self.stop_event.is_set():
            start_time = time.time()
            img_np = self.camera.get_frame() # np img
            img_PIL = Image.fromarray(img_np)
            time1 = time.time()
            output_pose = self.agent.policy(self.prompt, img_PIL)
            time2 = time.time() 
            # print('out: ', output_pose)
            print(self.spot.move_spot_arm(output_pose, offset=True))

            print(f'Time: {time1 - start_time:.3f}s, {time2 - start_time:.3f}s, {time.time() - start_time:.3f}')
            # time_to_sleep = self.control_period_timer - (time.time() - start_time)
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)

    def state_loop(self):
        print("State loop start")
        while True:
            start_time = time.time()
            # check arm safety via state and img
            _, self.spot.safe, self.spot.safe_info = self.spot.safe_boundary(self.spot.get_arm_pose('hand'))
            if not self.spot:
                print(self.spot.safe_info)
            self.camera.show_frame()
            time_to_sleep = self.state_period_timer - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
    def __exit__(self):
        self.thread_stop()
        self.spot.__exit__()
        self.camera.stop()
    
def main():
    m = Mission()
    try: 
        while m.control_thread.is_alive() and m.state_thread.is_alive():
            time.sleep(0.5)
        print('Error during running!')
        m.__exit__()
            
    except KeyboardInterrupt or Exception:
        m.__exit__()

if __name__ == '__main__':
    main()
