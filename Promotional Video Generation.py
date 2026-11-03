#Wan 2.1 TV2 1.3B vs. LTX Video 2B 0.9.8 distilled

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


def wan_ai():
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    prompt = "A photorealistic swimming pool advertisement in a modern Australian backyard. A long rectangular pool with clear turquoise water, pale grey stone coping, and light exposed paving fills the foreground. Behind it stands a single storey white house with a light grey metal roof, a covered patio and timber-framed glass doors. Frameless glass pool fencing runs along the left and right. Green trees surround the property under a clear blue sky with white clouds here and there. One continuous slow camera pan from left to right, filmed from the near end of the pool at eye level. The pool and house remains clearly visible. Gentle ripples shimmer in bright midday sunlight, and leaves of the trees move slightly in a light breeze. Stable architecture, realistic proportions, smooth camera movement, no cuts, no people, no text."
    
    output = pipe(
        prompt=prompt,
        height=480, 
        width=832, 
        num_frames=81,
        guidance_scale=5.0
    ).frames[0]

    export_to_video(output, "C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/outputs/promotional_video_generation/wan_ai_output.mp4", fps=15)

if __name__ == "__main__":
    print("Starting Wan video generation...")
    wan_ai()
    print("Video saved.")