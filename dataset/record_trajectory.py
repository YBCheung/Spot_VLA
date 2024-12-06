
class TrajectoryRecorder:
    def __init__(self, prompt):
        """
        Initialize the TrajectoryRecorder.
        
        Args:
            tfrecord_filename (str): The name of the TFRecord file to save data.
            prompt (str): A prompt string associated with the trajectory.
        """
        self.prompt = prompt
        dir = '../dataset/' # run under code/
        self.tfrecord_filename = dir + 'raw/raw_' + self.prompt
        self.writer = None   # if True, then write.
        self.done = 0  # Initialize 'done' flag

        self.count = 0
        self.json_name = dir + 'prompt_count.json'
        self.load_count()

    def load_count(self):
        # Check if the file exists
        if os.path.exists(self.json_name):
            with open(self.json_name, 'r') as f:
                data = json.load(f)  # Load JSON data
                if self.prompt in data:
                    self.count = data[self.prompt]  # Load count for the specific prompt

    def save_count(self):
        # Load existing data if the file exists
        if os.path.exists(self.json_name):
            with open(self.json_name, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Update the count for the prompt
        data[self.prompt] = self.count

        # Write the updated data back to the file
        with open(self.json_name, 'w') as f:
            json.dump(data, f, indent=4)  # Save data in JSON format

    def increment_count(self):
        self.count += 1  # Increment the count
        self.save_count()  # Save the updated count to the file


    def __del__(self):
        """
        Destructor to ensure the TFRecord writer is closed properly.
        """
        self.writer.close()
    
    def start_recording(self): # press z
        if self.count < 0 or self.count > 999:
            print(f'Error, {self.count}th trajectory')
            return
        filename = f'{self.tfrecord_filename}{self.count:03}.tfrecord'
        
        # overwrite existed file. 
        if self.writer:
            self.writer.close()
        if os.path.exists(filename):  # Check if the file exists
            os.remove(filename)  # Remove the existing file

        self.writer = tf.io.TFRecordWriter(filename)
        self.done = 0
    
    def task_done(self): # press e
        self.done = 1
        print('done')

    def end_recording(self): # press x
        if self.writer:
            self.writer.close()
        if self.done:   # one trajectory finished, ready for the next trajectory. 
            self.increment_count()


    def record_raw_state_step(self, image, pose):
        """
        Create a TFRecord entry with raw state data (image, pose, prompt string).
        
        Args:
            image (np.array): The image data to record.
            pose (list): The pose data to record (7-DOF).  # TODO: check type. 

        Returns:
            tf.train.Example: A serialized TFRecord example.
        """
        if self.record == False:
            return
        
        image_bytes = image.tobytes()  # Convert image to bytes

        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            'pose': tf.train.Feature(float_list=tf.train.FloatList(value=pose)),
            'prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.prompt.encode('utf-8')])),  # Add prompt as a byte feature
            'done': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.done]))  # 'done' flag
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

        if self.done == 1:
            print('trajectory done.')
