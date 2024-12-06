import tensorflow as tf
import numpy as np
import os

def _parse_function(proto):
    # Define the feature description to match the TFRecord structure
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Image stored as bytes
        'pose': tf.io.FixedLenFeature([7], tf.float32),  # Pose as a list of 7 floats
        'prompt': tf.io.FixedLenFeature([], tf.string),  # Prompt as bytes
        'done': tf.io.FixedLenFeature([], tf.int64),     # Done flag as int64
    }

    # Parse the input tf.train.Example proto using the dictionary above
    parsed_example = tf.io.parse_single_example(proto, feature_description)
    
    # Decode the image bytes (if the image is in raw bytes and not encoded as jpeg)
    image = np.frombuffer(parsed_example['image'].numpy(), dtype=np.uint8)  # You can reshape it later based on original image shape
    image = image.reshape((224,224,3)).tobytes()
    parsed_example['image'] = image  # Replace the bytes with the actual image array
    parsed_example['prompt'] = parsed_example['prompt'].numpy()

    return parsed_example

def calculate_pose_difference(pose1, pose2):
    """
    Calculates the difference between two poses.
    :param pose1: The pose from the first step, as a NumPy array
    :param pose2: The pose from the second step, as a NumPy array
    :return: The difference between the two poses as a NumPy array
    """
    return np.array(pose1) - np.array(pose2)


# Set the step variable (how many steps apart to compare poses)
step = 2

# Create a dataset from the .tfrecord file
raw_dir = 'raw/'
action_dir = 'action/'

for filename in os.listdir(raw_dir):
    raw_path = os.path.join(raw_dir, filename)
    if not os.path.isfile(raw_path):
        continue
    if filename.startswith('raw_'):
        filename = filename[len('raw_'):]
    action_path = os.path.join(action_dir, 's_' + str(step) + '_' + filename)
    raw_dataset = tf.data.TFRecordDataset(raw_path)
    action_writer = tf.io.TFRecordWriter(action_path)

    # Initialize variables for tracking poses
    previous_poses = []
    counter = 0


    # Iterate over the dataset and calculate the pose differences
    for raw_record in raw_dataset:
        parsed_record = _parse_function(raw_record)
        
        # Store the current pose
        current_pose = parsed_record['pose']

        # If we have stored enough poses to calculate the difference (based on the step)
        if len(previous_poses) == step:
            # Calculate the difference between the current pose and the one 'step' frames ago
            pose_difference = calculate_pose_difference(current_pose, previous_poses[0])

            # Print or process the pose difference as needed
            print(f"Step: {counter}, Pose Difference: {pose_difference}, {parsed_record['done']}")
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[parsed_record['image']])),
                'pose': tf.train.Feature(float_list=tf.train.FloatList(value=pose_difference)),
                'prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[parsed_record['prompt']])),  # Add prompt as a byte feature
                'done': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_record['done']]))  # 'done' flag
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            action_writer.write(example.SerializeToString())

            # Remove the oldest pose and append the current pose to the list
            previous_poses.pop(0)

        # Add the current pose to the list of previous poses
        previous_poses.append(current_pose)
        
        # Increment counter for tracking steps
        counter += 1
