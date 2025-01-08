import numpy as np
import cv2
import psutil
import tracemalloc
import os
import gc

def monitor_memory():
    """Returns the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def lazy_data_loader(npz_file, video_file):
    """Lazy loader for arm_pose, frame, and timestep."""
    # Load the .npz file with memory mapping
    data = np.load(npz_file, mmap_mode="r")
    arm_poses = data["arm_poses"]
    timesteps = data["timestamps"]

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    # Yield data lazily
    try:
        for i in range(len(arm_poses)):
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Video ended prematurely.")
            
            yield timesteps[i], arm_poses[i], frame

    finally:
        cap.release()


def lazy_data_loader_delta(npz_file, video_file):
    """Lazy loader for arm_pose, frame, and timestep."""
    # Load the .npz file with memory mapping
    data = np.load(npz_file, mmap_mode="r")
    arm_poses = data["arm_poses"]
    timesteps = data["timestamps"]

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")
    
    old_time = None
    old_pose = [0., 0., 0., 0., 0., 0., 0., 0.]

    # Yield data lazily
    try:
        for i in range(len(arm_poses)):
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Video ended prematurely.")
            
            time_delta = 0 if old_time is None else timesteps[i] - old_time
            pose = arm_poses[i]
            # pose_delta = arm_poses[i]
            # pose_delta[:7] -= old_pose[:7]  
            pose[7] = 1 if pose[7] > 80 else 0  # hand open 1, close 0

            yield time_delta, pose, frame

            old_time = timesteps[i]
            old_pose = arm_poses[i]

    finally:
        cap.release()

# Example usage
def main(npz_file, video_file):

    # Replace with your actual file paths
    folder_path = '/home/spot/docker/spot_optik_ctrl/ros2_ws/recorded_data/20250103_1735934820440384621_Sequence'
    actual_pose_name = 'actual_poses.npz'
    rgb_video_name = 'rgb_video.avi'
    hand_video_name = 'rgb_hand_video.avi'

    npz_file_path = os.path.join(folder_path, actual_pose_name)
    video_file_path = os.path.join(folder_path, hand_video_name)

    datas = []
    loader = lazy_data_loader(npz_file_path, video_file_path)
    for i, (timestep, arm_pose, frame) in enumerate(loader):
        
        # Example: Process the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

        # release the memory
        # it's important for memory release if stored in a list. 
        # if use lazy loaded data with yield, no need for release, as generater do not store data to memory.
        # instead, the data produced by yield is immediately passed to the consumer. 
        # it's up to the caller to store, process and release the memory. 
        # it can be done by set to None, then gc.collect(); 
        # or, by overwriting to the same addr. #
        
        datas.append([timestep, arm_pose, frame])
        datas[i] = None  
        if i % 10 == 0:  # Check memory every 10 iterations
            print(f"Iteration {i}, Memory Usage: {monitor_memory():.2f} MB")
        
        if i % 200 == 0:
            gc.collect()  # Force garbage collection
            print(f"Iteration {i}, After Memory Usage: {monitor_memory():.2f} MB")


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()