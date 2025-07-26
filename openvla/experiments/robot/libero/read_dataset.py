import tensorflow as tf

# Path to your TFRecord file
tfrecord_path = "datasets/modified_libero_rlds/libero_object_no_noops/1.0.0/libero_object-train.tfrecord-00000-of-00032"

# Define the feature description dictionary for parsing
# Update the keys and types to match your dataset!
feature_description = {
    "obs/agentview_image": tf.io.FixedLenFeature([], tf.string),  # images are usually stored as bytes
    "actions": tf.io.FixedLenFeature([7], tf.float32),            # example: 7-DoF action
    # Add more features as needed
}

def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    # Decode image if needed
    image = tf.io.decode_raw(parsed["obs/agentview_image"], tf.uint8)
    # You may need to reshape the image, e.g.:
    # image = tf.reshape(image, [height, width, channels])
    return image, parsed["actions"]

# Create a TFRecordDataset
dataset = tf.data.TFRecordDataset([tfrecord_path])

# Parse the dataset
parsed_dataset = dataset.map(_parse_function)

# Iterate and print the first few samples
for i, (image, action) in enumerate(parsed_dataset.take(3)):
    print(f"Sample {i}:")
    print("Image shape:", image.shape)
    print("Action:", action.numpy())