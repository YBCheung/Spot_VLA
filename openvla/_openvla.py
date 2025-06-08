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
        # self.MODEL_PATH = "runs/openvla-7b+spot_kitchen+b16+lr-2e-05+lora-r64+dropout-0.0--image_aug--6000_chkpt"
        self.MODEL_PATH = "/home/rllab/isaacsim/codes/openvla/runs/openvla-7b+spot_kitchen+b1+lr-2e-05+lora-r16+dropout-0.0--image_aug--319_chkpt"
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