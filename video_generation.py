#Wan 2.1 TV2 1.3B vs. LTX Video 2B 0.9.8 distilled

import torch
import time
import json
from datetime import datetime
from pathlib import Path
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from ltx_video.inference import create_ltx_video_pipeline



def wan_ai():
    project_folder = Path(__file__).resolve().parent
    output_folder = (
        project_folder / "outputs" / "video_generation"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"wan_{timestamp}.mp4"
    details_path = output_folder / f"wan_{timestamp}.json"

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    prompt = "A photorealistic swimming pool advertisement in a modern Australian backyard. A long rectangular pool with clear turquoise water, pale grey stone coping, and light exposed paving fills the foreground. Behind it stands a single storey white house with a light grey metal roof, a covered patio and timber-framed glass doors. Frameless glass pool fencing runs along the left and right. Green trees surround the property under a clear blue sky with white clouds here and there. One continuous, smooth foward camera movement from the rear end of the pool towards the house, gliding slowly above the center of the water at a constant height and speed. The camera faces straight ahead, keeping the house centered as it gradually grows larger. Pool edges move naturally past the sides of the frame, creating clear depth and parallax. Physical foward movement, no zoom, no sideways pan. Gentle ripples shimmer in bright midday sinlight, and tree leaves sway slightly in a light breeze. Stable architecture, realistic proportions, no cuts, no people, no text."

    print("Loading Wan")
    loading_start = time.perf_counter()

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    loading_seconds = time.perf_counter() - loading_start
    generator = torch.Generator(device="cuda").manual_seed(42)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Generating Wan video...")
    start = time.perf_counter()

    with torch.inference_mode():
        frames = pipe(
            prompt=prompt,
            height=480,
            width=832,
            num_frames=33,
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=generator,
        ).frames[0]

    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    export_to_video(frames, str(output_path), fps=15)

    details = {
        "name": "wan",
        "model": model_id,
        "prompt": prompt,
        "width": 832,
        "height": 480,
        "num_frames": 33,
        "actual_num_frames": len(frames),
        "export_fps": 15,
        "duration_seconds": round(len(frames) / 15, 3),
        "seed": 42,
        "random_generator_device": "cuda",
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "dtype": "bfloat16",
        "vae_dtype": "float32",
        "memory_mode": "model_cpu_offload",
        "loading_seconds": round(loading_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_gpu_allocated_gib": round(peak_memory, 3),
        "gpu": torch.cuda.get_device_name(),
        "warmup_performed": False,
        "output_file": str(output_path),
    }

    details_path.write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    print(f"Video saved: {output_path}")
    print(f"Measurements saved: {details_path}")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Peak GPU memory allocated: {peak_memory:.2f} GiB")




def ltx():
    project_folder = Path(__file__).resolve().parent
    output_folder = (
        project_folder / "outputs" / "video_generation"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"ltx_{timestamp}.mp4"
    details_path = output_folder / f"ltx_{timestamp}.json"

    model_id = "Lightricks/LTX-Video"
    checkpoint_filename = "ltxv-2b-0.9.8-distilled.safetensors"

    prompt = "A photorealistic swimming pool advertisement in a modern Australian backyard. A long rectangular pool with clear turquoise water, pale grey stone coping, and light exposed paving fills the foreground. Behind it stands a single storey white house with a light grey metal roof, a covered patio and timber-framed glass doors. Frameless glass pool fencing runs along the left and right. Green trees surround the property under a clear blue sky with white clouds here and there. One continuous, smooth foward camera movement from the rear end of the pool towards the house, gliding slowly above the center of the water at a constant height and speed. The camera faces straight ahead, keeping the house centered as it gradually grows larger. Pool edges move naturally past the sides of the frame, creating clear depth and parallax. Physical foward movement, no zoom, no sideways pan. Gentle ripples shimmer in bright midday sinlight, and tree leaves sway slightly in a light breeze. Stable architecture, realistic proportions, no cuts, no people, no text."
    
    timesteps = [
        1.0, 0.9937, 0.9875, 0.9812,
        0.9750, 0.9094, 0.7250,
    ]

    print("Loading LTX Video 2B distilled")
    loading_start = time.perf_counter()

    checkpoint = hf_hub_download(
        repo_id=model_id,
        filename=checkpoint_filename,
    )

    pipe = create_ltx_video_pipeline(
        ckpt_path=checkpoint,
        precision="bfloat16",
        text_encoder_model_name_or_path=(
            "PixArt-alpha/PixArt-XL-2-1024-MS"
        ),
        sampler="from_checkpoint",
        device="cpu",
        enhance_prompt=False,
    )

    pipe.model_cpu_offload_seq = "text_encoder->transformer"
    pipe._exclude_from_cpu_offload = ["vae"]
    pipe.enable_model_cpu_offload()
    pipe.vae.disable_hw_tiling()
    pipe.vae.disable_z_tiling()

    loading_seconds = time.perf_counter() - loading_start
    generator = torch.Generator(device="cuda").manual_seed(42)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Generating LTX video...")
    start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            is_video=True,
            height=480,
            width=832,
            num_frames=33,
            frame_rate=15,
            timesteps=timesteps,
            guidance_scale=1.0,
            stg_scale=0.0,
            rescaling_scale=1.0,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            stochastic_sampling=False,
            enhance_prompt=False,
            generator=generator,
            vae_per_channel_normalize=True,
            output_type="pt",
            offload_to_cpu=True,
        )

        frames = (
            result.images[0]
            .detach()
            .cpu()
            .float()
            .permute(1, 2, 3, 0)
            .numpy()
        )

    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    export_to_video(frames, str(output_path), fps=15)

    details = {
        "name": "ltx",
        "model": model_id,
        "checkpoint": checkpoint_filename,
        "text_encoder": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "prompt": prompt,
        "width": 832,
        "height": 480,
        "num_frames": 33,
        "actual_num_frames": len(frames),
        "conditioning_frame_rate": 15,
        "export_fps": 15,
        "duration_seconds": round(len(frames) / 15, 3),
        "seed": 42,
        "random_generator_device": "cuda",
        "timesteps": timesteps,
        "num_sampling_timesteps": len(timesteps),
        "guidance_scale": 1.0,
        "stg_scale": 0.0,
        "rescaling_scale": 1.0,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "stochastic_sampling": False,
        "enhance_prompt": False,
        "vae_per_channel_normalize": True,
        "offload_to_cpu": True,
        "dtype": "bfloat16",
        "memory_mode": (
            "text_encoder_and_transformer_cpu_offload_vae_excluded"
        ),
        "vae_hw_tiling": False,
        "vae_z_tiling": False,
        "loading_seconds": round(loading_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_gpu_allocated_gib": round(peak_memory, 3),
        "gpu": torch.cuda.get_device_name(),
        "warmup_performed": False,
        "output_file": str(output_path),
    }

    details_path.write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    print(f"Video saved: {output_path}")
    print(f"Measurements saved: {details_path}")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Peak GPU memory allocated: {peak_memory:.2f} GiB")


def comparison():
    import matplotlib.pyplot as plt

    project_folder = Path(__file__).resolve().parent
    results_folder = (
        project_folder / "outputs" / "video_generation"
    )
    comparison_folder = results_folder / "video_generation_comparison"

    runs = {}

    for name in ("wan", "ltx"):
        files = sorted(results_folder.glob(f"{name}_*.json"))

        if not files:
            raise FileNotFoundError(
                f"No saved measurements for {name} in {results_folder}"
            )

        selected_file = files[-1]
        runs[name] = {
            "source_json": selected_file.name,
            "data": json.loads(selected_file.read_text(encoding="utf-8")),
        }
        print(f"Using {selected_file.name}")

    wan = runs["wan"]["data"]
    ltx = runs["ltx"]["data"]

    for field in (
        "prompt", "width", "height", "seed",
        "num_frames", "actual_num_frames", "export_fps",
    ):
        if wan[field] != ltx[field]:
            raise ValueError(
                f"The selected Wan and LTX runs differ in {field}."
            )

    labels = [
        f"Wan 2.1\n{wan['num_inference_steps']} steps",
        f"LTX Video 2B\n{ltx['num_sampling_timesteps']} sampling timesteps",
    ]

    metrics = [
        ("generation_seconds", "Video generation time", "Seconds"),
        ("loading_seconds", "Model loading time", "Seconds"),
        ("peak_gpu_allocated_gib", "Peak GPU memory allocated", "GiB"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for axis, (field, title, unit) in zip(axes, metrics):
        values = [wan[field], ltx[field]]
        bars = axis.bar(labels, values, color=["#2563eb", "#f97316"])
        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_ylim(0, max(max(values) * 1.2, 0.1))

    fig.suptitle("Pool advertisement generation — Wan 2.1 vs LTX Video 2B")
    fig.text(
        0.5, 0.03,
        f"{wan['width']} × {wan['height']} · "
        f"{wan['actual_num_frames']} frames at {wan['export_fps']} FPS · "
        "One run per model, no warm-up.\n"
        "Generation time excludes MP4 encoding. "
        "Peak memory measures PyTorch allocations during generation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))

    comparison_folder.mkdir(parents=True, exist_ok=True)
    chart_path = comparison_folder / "time_and_memory.png"
    summary_path = comparison_folder / "comparison_summary.json"

    fig.savefig(chart_path, dpi=200)
    summary_path.write_text(
        json.dumps(runs, indent=2),
        encoding="utf-8",
    )

    print(f"Chart saved: {chart_path}")
    print(f"Summary saved: {summary_path}")
    plt.show()

if __name__ == "__main__":
    comparison()
    print("Video saved.")