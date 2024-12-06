import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Path to your TFRecord file
tfrecord_path = 'bridge/bridge.tfrecord'

# Load the TFRecord dataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

# Function to parse and print all features dynamically
def parse_and_print_example(serialized_example):
    # Parse the Example
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())
    
    # Iterate over each feature in the example
    for key, feature in example.features.feature.items():
        # Check the type of each feature and print accordingly
        if feature.HasField("int64_list"):
            print(f"{key}: {len(feature.int64_list.value)}")
        elif feature.HasField("float_list"):
            print(f"{key}: {len(feature.float_list.value)}")
        elif feature.HasField("bytes_list"):
            # Print first few bytes for byte features (for brevity)
            # Decode one image to get its dimensions
                           
            if key.split('_')[0] == 'steps/observation/image':            
                try:
                    # Try decoding as JPEG first
                    for first_image_bytes in feature.bytes_list.value[:3]:
                        first_image = tf.io.decode_jpeg(first_image_bytes).numpy()
                        print(f"{key}: {len(feature.bytes_list.value)}, {first_image.shape}, jpeg")
                        pil_image = Image.fromarray(first_image)
                        pil_image.show()
                        plt.imshow(pil_image)
                        plt.title(key)
                        plt.show()
                    

                except tf.errors.InvalidArgumentError:
                    try:
                        # Try decoding as PNG if JPEG fails
                        first_image = tf.io.decode_png(first_image_bytes).numpy()
                        pil_image = Image.fromarray(first_image)
                        pil_image.show()

                        print(f"{key}: {len(feature.bytes_list.value)}, {first_image.shape}, png")
                        
                    except tf.errors.InvalidArgumentError:
                        print(f"{key}: {len(feature.bytes_list.value)}, First image could not be decoded as JPEG or PNG. Unknown format.")
            else:
                print(f"{key}: {feature.bytes_list.value}")
        else:
            print(f"{key}: Unknown feature type")

# Print features from the first few examples
for serialized_example in raw_dataset.take(3):  # Adjust the number as needed
    parse_and_print_example(serialized_example)
    print("-----")  # Separator for each example
