import os
import torch
from torchvision.transforms import ToTensor

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import comfy.model_management

script_directory = os.path.dirname(os.path.abspath(__file__))

class DiffusersStableCascade:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 512, "min": 128, "max": 8192, "step": 128}),
            "height": ("INT", {"default": 512, "min": 128, "max": 8192, "step": 128}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
            "decoder_steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "",}),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"

    CATEGORY = "DiffusersStableCascade"

    def process(self, width, height, seed, steps, guidance_scale, prompt, negative_prompt, batch_size, decoder_steps):
        
        comfy.model_management.unload_all_models()
        torch.manual_seed(seed)

        device = comfy.model_management.get_torch_device()

        if not hasattr(self, 'prior') or not hasattr(self, 'decoder'):

            self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
            self.prior.enable_model_cpu_offload()
            self.decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to(device)
            self.decoder.enable_model_cpu_offload()

        prior_output = self.prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps
        )
        decoder_output = self.decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=decoder_steps
        ).images
    
        tensors = [ToTensor()(img) for img in decoder_output]
        batch_tensor = torch.stack(tensors).permute(0, 2, 3, 1).cpu()
        
        
        return (batch_tensor,)


NODE_CLASS_MAPPINGS = {
    "DiffusersStableCascade": DiffusersStableCascade,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersStableCascade": "DiffusersStableCascade",
}