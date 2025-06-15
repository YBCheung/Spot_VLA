import numpy as np
import tqdm
import os
import re
import cv2  # OpenCV for reading .avi files

# Define paths to the dataset
root_data_path = '/scratch/work/zhangy50/RL/spot_VLA/dataset/raw/grasp_carrot'  # Update this path
RESHAPE = True
TrainVal_div = 5 # 4:1


def center_crop(image):
    """
    Crop the center of the image while keeping the original aspect ratio.
    The crop will be square and the size will be the smaller of height or width.
    Then resize the cropped image to (224, 224).
    """
    h, w, _ = image.shape
    # print(f"Original image shape: {image.shape}")  # Print the original shape of the image
    # Determine the size of the square crop (take the minimum of width and height)
    crop_size = min(h, w)
    
    # Calculate cropping box
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    
    # Crop the image
    cropped_image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize
    cropped_image_resized = cv2.resize(cropped_image, (224, 224)) # 64 for spot_kitchen dataset
    
    return cropped_image_resized

def create_episode_from_files(path, prompt, img_path, hand_path, state_path, action_path):
    episode = []
    
    # Check if files exist before loading
    if not os.path.exists(state_path):
        print(f"Error: The file {state_path} does not exist.")
        return
    if not os.path.exists(action_path):
        print(f"Error: The file {action_path} does not exist.")
        return
    if not os.path.exists(img_path):
        print(f"Error: The video file {img_path} does not exist.")
        return
    if not os.path.exists(hand_path):
        print(f"Error: The video file {hand_path} does not exist.")
        return
    
    # Load videos and data
    state_data = np.load(state_path)  # Load state.npz
    action_data = np.load(action_path)  # Load action.npz
    img_video = cv2.VideoCapture(img_path)  # Read img.avi
    hand_video = cv2.VideoCapture(hand_path)  # Read hand.avi
    
    if (not img_video.isOpened() or not hand_video.isOpened() or
        len(state_data.files) == 0 or len(action_data.files) == 0):
        print("Error: One or more files could not be opened or are empty.")

        # Release video captures if they were opened
        if img_video and img_video.isOpened():
            img_video.release()
        if hand_video and hand_video.isOpened():
            hand_video.release()
        return

    state_arm = state_data['arm_poses']    
    target_gripper = np.zeros((state_arm.shape[0], 1))
    target_gripper[:-1, 0] = state_arm[1:, -1] > 90
    target_gripper[-1, -1] = target_gripper[-1] 
    action_arm = np.hstack((action_data['target_positions'], action_data['target_poses'], target_gripper))
    
    state_arm = np.asarray(state_arm, dtype=np.float32)
    action_arm = np.asarray(action_arm, dtype=np.float32)

    # Get the minimum number of frames among the videos
    img_frame_count = int(img_video.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_frame_count = int(hand_video.get(cv2.CAP_PROP_FRAME_COUNT))
    episode_length = min(img_frame_count, hand_frame_count, state_arm.shape[0], len(action_data))

    for step in range(episode_length):
        # Read frames from the videos (assuming the length is the same in both videos)
        ret_img, image_frame = img_video.read()
        ret_hand, hand_frame = hand_video.read()

        if not ret_img or not ret_hand:
            print(f"Error: Could not read video frames at step {step}")
            break

        # Check shape of images
        # print(f"Original img frame shape: {image_frame.shape}")  # Print the original shape of img frame
        # print(f"Original hand frame shape: {hand_frame.shape}")  # Print the original shape of hand frame
        
        if RESHAPE==True:
            image_frame = center_crop(image_frame)
            hand_frame = center_crop(hand_frame)

        # Ensure the frames are in the right format (uint8 for image frames)
        image_frame = np.asarray(image_frame, dtype=np.uint8)
        hand_frame = np.asarray(hand_frame, dtype=np.uint8)

        # Get state and action data for the current step
        state = state_arm[step]
        action = action_arm[step] 
        # Add to episode
        episode.append({
            'image': image_frame,
            'wrist_image': hand_frame,
            'state': state,
            'action': action,
            'language_instruction': prompt,
        })

    # Save the episode
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, episode)
    print(prompt, episode_length, img_path)
    # Release video captures if they were opened
    if img_video and img_video.isOpened():
        img_video.release()
    if hand_video and hand_video.isOpened():
        hand_video.release()

# Create fake episodes for train and validation
print("Generating train examples...")
# Iterate over each folder to create train episodes
# os.makedirs('data/train', exist_ok=True)
# os.makedirs('data/val', exist_ok=True)

for task in os.listdir(root_data_path):
    task_path = os.path.join(root_data_path, task)
    for i, folder_path in enumerate(tqdm.tqdm(os.listdir(task_path))):
               
        # Define paths to the .avi and .npz files within each folder
        img_file_path = os.path.join(task_path, folder_path, 'rgb_video.avi')
        hand_file_path = os.path.join(task_path, folder_path, 'rgb_hand_video.avi')
        state_file_path = os.path.join(task_path, folder_path, 'actual_poses.npz')
        action_file_path = os.path.join(task_path, folder_path, 'target_poses.npz')

        folder_name = os.path.basename(folder_path)
        prompt = "grasp the carrot"
        if i % TrainVal_div == 0:
            create_episode_from_files(f'data/{task}/val/{task}_{i}.npy', prompt, img_file_path, hand_file_path, state_file_path, action_file_path) 
        else:
            create_episode_from_files(f'data/{task}/train/{task}_{i}.npy', prompt, img_file_path, hand_file_path, state_file_path, action_file_path) 

print('Successfully created example data!')
