import pyrealsense2 as rs  # try to read image via ssh, 
import numpy as np
import cv2

class RealSenseCapture:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()

        # Configure the pipeline to stream the color data
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if we got frames, just in case
        if not color_frame:
            return None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def show_frame(self):
        # Get the current frame
        frame = self.get_frame()

        # Show the frame if it's not None
        if frame is not None:
            cv2.imshow('RealSense Color Frame', frame)
            cv2.waitKey(1)  # This is necessary for OpenCV to process window events

    def stop(self):
        # Stop the pipeline and clean up
        self.pipeline.stop()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    while True:
        realsense_capture = RealSenseCapture()

        # Capture and display a single frame when the function is called
        realsense_capture.show_frame()  # Call this when you want to capture a frame

    # Stop the pipeline when done
    realsense_capture.stop()
