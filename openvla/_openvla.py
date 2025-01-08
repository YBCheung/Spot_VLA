from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
import numpy as np
from PIL import Image

class openvla():
    '''
    openvla
    '''
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # Set up 4-bit quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Optional: you can change to other quantization types
            bnb_4bit_use_double_quant=True,  # Double quantization improves memory savings
            bnb_4bit_compute_dtype=torch.bfloat16  # Ensure FlashAttention compatibility
        )
        # Load Processor
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        # Load VLA Model with 4-bit quantization
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            quantization_config=self.bnb_config,  # Use BitsAndBytesConfig for 4-bit loading
            trust_remote_code=True
        )
    def policy(self, prompt, image):
        
        # Process the input and move to the appropriate device
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return action

def main():
    agent = openvla()
    img = Image.open('hold_stick.jpg')
    # Assume img_np is a 2D NumPy array, like an image that cv2 can display
    img_np = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Example 2D NumPy array

    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(img_np)
    prompt = "In: Aim the center of Field of view to the blue pepper container {<INSTRUCTION>}?\nOut:"

    action = agent.policy(prompt, img_pil)
    print(action)

if __name__ == '__main__':
    main()