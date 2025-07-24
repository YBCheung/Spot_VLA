from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
import numpy as np
import os
from PIL import Image

class openvla():
    '''
    openvla
    '''
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)
        
        SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))  # `openvla/server_vla`
        # Navigate up to the project root, then into `run/model`
        OPENVLA_ROOT = os.path.dirname(SCRIPT_DIR)  # Goes up to `project/`
        # self.MODEL_PATH = "/home/rllab/spot_vla/Spot_VLA/openvla/runs/spot_kitchen/openvla-7b+spot_kitchen+b16+lr-2e-05+shf100000+lora-r64+dropout-0.0--image_aug--999_chkpt--loss-3.703"
        self.MODEL_PATH = "/home/rllab/spot_vla/Spot_VLA/openvla/runs/test_models/openvla-7b+spot_carrot+b16+lr-0.0005+shf100000+lora-r32+dropout-0.0--image_aug/--1000_chkpt--loss-7.307"
        model_absolute_path = os.path.join(SCRIPT_DIR, self.MODEL_PATH)
        print(model_absolute_path)

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_absolute_path, trust_remote_code=True)

        # Load VLA Model with 4-bit quantization
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_absolute_path, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    def policy(self, prompt, image):
        
        # Process the input and move to the appropriate device
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return action

def main():
    agent = openvla()
    img = Image.open('cutlery.jpg')
    # Assume img_np is a 2D NumPy array, like an image that cv2 can display
    img_np = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Example 2D NumPy array

    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(img_np)
    prompt = "In: Aim the center of Field of view to the blue pepper container {<INSTRUCTION>}?\nOut:"

    action = agent.policy(prompt, img_pil)
    print(action)

if __name__ == '__main__':
    main()