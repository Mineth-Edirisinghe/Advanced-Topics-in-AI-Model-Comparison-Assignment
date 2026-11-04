#Wan 2.1 TV2 1.3B vs. LTX Video 2B 0.9.8 distilled

import torch
from pathlib import Path
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from ltx_video.inference import create_ltx_video_pipeline



def wan_ai():
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    prompt = "A photorealistic swimming pool advertisement in a modern Australian backyard. A long rectangular pool with clear turquoise water, pale grey stone coping, and light exposed paving fills the foreground. Behind it stands a single storey white house with a light grey metal roof, a covered patio and timber-framed glass doors. Frameless glass pool fencing runs along the left and right. Green trees surround the property under a clear blue sky with white clouds here and there. One continuous, smooth foward camera movement from the rear end of the pool towards the house, gliding slowly above the center of the water at a constant height and speed. The camera faces straight ahead, keeping the house centered as it gradually grows larger. Pool edges move naturally past the sides of the frame, creating clear depth and parallax. Physical foward movement, no zoom, no sideways pan. Gentle ripples shimmer in bright midday sinlight, and tree leaves sway slightly in a light breeze. Stable architecture, realistic proportions, no cuts, no people, no text."
    
    output = pipe(
        prompt=prompt,
        height=480, 
        width=832, 
        num_frames=81,
        guidance_scale=5.0
    ).frames[0]

    export_to_video(output, "C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/outputs/promotional_video_generation/wan_ai_output.mp4", fps=15)


def ltx():
    print("Loading LTX Video 2b distilled.")

    checkpoint = hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename="ltxv-2b-0.9.8-distilled.safetensors"
    )

    pipe = create_ltx_video_pipeline(
        ckpt_path=checkpoint,
        precision="bfloat16",
        text_encoder_model_name_or_path=(
            "PixArt-alpha/PixArt-XL-2-1024-MS"
        ),
        sampler="from_checkpoint",
        device="cpu",
        enhance_prompt=False
    )

    pipe.model_cpu_offload_seq = "text_encoder->transformer"
    pipe._exclude_from_cpu_offload = ["vae"]
    pipe.enable_model_cpu_offload()

    prompt = "A photorealistic swimming pool advertisement in a modern Australian backyard. A long rectangular pool with clear turquoise water, pale grey stone coping, and light exposed paving fills the foreground. Behind it stands a single storey white house with a light grey metal roof, a covered patio and timber-framed glass doors. Frameless glass pool fencing runs along the left and right. Green trees surround the property under a clear blue sky with white clouds here and there. One continuous, smooth foward camera movement from the rear end of the pool towards the house, gliding slowly above the center of the water at a constant height and speed. The camera faces straight ahead, keeping the house centered as it gradually grows larger. Pool edges move naturally past the sides of the frame, creating clear depth and parallax. Physical foward movement, no zoom, no sideways pan. Gentle ripples shimmer in bright midday sinlight, and tree leaves sway slightly in a light breeze. Stable architecture, realistic proportions, no cuts, no people, no text."
    
    print("Generating video...")

    result = pipe(
        prompt=prompt,
        height=480,
        width=832,
        num_frames=81,
        frame_rate=15,
        timesteps=[
            1.0, 0.9937, 0.9875, 0.9812,
            0.9750, 0.9094, 0.7250
        ],
        guidance_scale=1.0,
        stg_scale=0.0,
        rescaling_scale=1.0,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        stochastic_sampling=False,
        enhance_prompt=False,
        generator=torch.Generator(device="cuda").manual_seed(42),
        vae_per_channel_normalize=True,
        output_type="pt"
    )

    frames = (
        result.images[0]
        .permute(1, 2, 3, 0)
        .float()
        .cpu()
        .numpy()
    )

    output_path = Path("C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/outputs/promotional_video_generation/ltx_ai_output.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_video(frames, str(output_path), fps=15)

    

if __name__ == "__main__":
    ltx()
    print("Video saved.")