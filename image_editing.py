#Flux.2 Klein 4B vs. Qwen Image Edit

import torch
import json
import time
from datetime import datetime
from pathlib import Path
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def flux():
        project_folder = Path(__file__).resolve().parent
        input_path = project_folder /   "outputs" / "image_editing" / "backyard_image.jpg"
        
        output_folder = project_folder / "outputs" / "image_editing"
        output_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_folder / f"flux_{timestamp}.png"
        details_path = output_folder / f"flux_{timestamp}.json"

        backyard = Image.open(input_path).convert("RGB")
        model_id = "black-forest-labs/FLUX.2-klein-4B"

        prompt = "Edit this backyard photograph to add a compact rectangular inground swimming pool within the lawn on the left side of the image. Position its long axis roughly parallel to the timber fence on the left, extending from the foreground towards the rear fence. Keep the pool entirely within the grass area, leaving visible space between it, the fences and the house. Give the pool a pale blue interior, clear turquoise water, realistic depth, submerged entry steps at the far end and a narrow underwater bench along one side. Add subtle ripples and reflections of the pink-purple sunset sky. Use pale grey stone coping around the pool with a narrow matching paved border, retaining grass around the outside. Add transparent frameless glass pool fencing with discreet metal spigots around the pool area, including a glass gate facing the existing concrete path. Keep the house entrance and right-hand side passage unobstructed. Preserve the original brick house, roof, solar panels, windows, visible interior, timber boundary fences, neighbouring house, existing concrete paths and flower bed. Preserve the original camera position, perspective, image framing and sunset sky. Match the existing dusk lighting, with subtle underwater pool lights and natural reflections. Produce a photorealistic architectural design mockup. Keep all changes confined to the pool and its immediate surroundings. No people, text, logos or additional buildings."

        print("Loading FLUX")
        loading_start = time.perf_counter()

        pipe = Flux2KleinPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        loading_seconds = time.perf_counter() - loading_start
        generator = torch.Generator(device="cuda").manual_seed(42)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        print("Generating FLUX image")
        start = time.perf_counter()

        with torch.inference_mode():
                image = pipe(
                        image=backyard,
                        prompt=prompt,
                         height=1024,
                         width=1024,
                         guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=generator,
                ).images[0]

        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        image.save(image_path)

        details = {
                "name": "flux",
                "model": model_id,
                "source_image": str(input_path),
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "actual_output_size": list(image.size),
                "seed": 42,
                "random_generator_device": "cuda",
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "dtype": "bfloat16",
                "memory_mode": "model_cpu_offload",
                "loading_seconds": round(loading_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "peak_gpu_allocated_gib": round(peak_memory, 3),
                "gpu": torch.cuda.get_device_name(),
                "warmup_performed": False,
                "output_file": str(image_path),
        }

        details_path.write_text(
                json.dumps(details, indent=2),
                encoding="utf-8",
        )

        print(f"Image saved: {image_path}")
        print(f"Measurements saved: {details_path}")
        print(f"Generation time: {generation_seconds:.2f} seconds")
        print(f"Peak GPU memory allocated: {peak_memory:.2f} GiB")




def qwen_image():
        project_folder = Path(__file__).resolve().parent
        input_path = project_folder /"outputs" / "image_editing" / "backyard_image.jpg"
        

        output_folder = project_folder / "outputs" / "image_editing"
        output_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_folder / f"qwen_{timestamp}.png"
        details_path = output_folder / f"qwen_{timestamp}.json"

        backyard = Image.open(input_path).convert("RGB")
        model_id = "toandev/Qwen-Image-Edit-2511-4bit"

        prompt = "Edit this backyard photograph to add a compact rectangular inground swimming pool within the lawn on the left side of the image. Position its long axis roughly parallel to the timber fence on the left, extending from the foreground towards the rear fence. Keep the pool entirely within the grass area, leaving visible space between it, the fences and the house. Give the pool a pale blue interior, clear turquoise water, realistic depth, submerged entry steps at the far end and a narrow underwater bench along one side. Add subtle ripples and reflections of the pink-purple sunset sky. Use pale grey stone coping around the pool with a narrow matching paved border, retaining grass around the outside. Add transparent frameless glass pool fencing with discreet metal spigots around the pool area, including a glass gate facing the existing concrete path. Keep the house entrance and right-hand side passage unobstructed. Preserve the original brick house, roof, solar panels, windows, visible interior, timber boundary fences, neighbouring house, existing concrete paths and flower bed. Preserve the original camera position, perspective, image framing and sunset sky. Match the existing dusk lighting, with subtle underwater pool lights and natural reflections. Produce a photorealistic architectural design mockup. Keep all changes confined to the pool and its immediate surroundings. No people, text, logos or additional buildings."

        print("Loading Qwen Image Edit")
        loading_start = time.perf_counter()

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
        )
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_tiling()

        loading_seconds = time.perf_counter() - loading_start
        generator = torch.Generator(device="cpu").manual_seed(42)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        print("Generating Qwen image")
        start = time.perf_counter()

        with torch.inference_mode():
                image = pipeline(
                        image=[backyard],
                        prompt=prompt,
                        height=1024,
                        width=1024,
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=40,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                ).images[0]

        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        image.save(image_path)

        details = {
                "name": "qwen",
                "model": model_id,
                "source_image": str(input_path),
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "actual_output_size": list(image.size),
                "seed": 42,
                "random_generator_device": "cpu",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_images_per_prompt": 1,
                "checkpoint_variant": "community_4bit",
                "requested_dtype": "bfloat16",
                "vae_tiling": True,
                "memory_mode": "model_cpu_offload",
                "loading_seconds": round(loading_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "peak_gpu_allocated_gib": round(peak_memory, 3),
                "gpu": torch.cuda.get_device_name(),
                "warmup_performed": False,
                "output_file": str(image_path),
        }

        details_path.write_text(
                json.dumps(details, indent=2),
                encoding="utf-8",
        )

        print(f"Image saved: {image_path}")
        print(f"Measurements saved: {details_path}")
        print(f"Generation time: {generation_seconds:.2f} seconds")
        print(f"Peak GPU memory allocated: {peak_memory:.2f} GiB")


def comparison():
    import matplotlib.pyplot as plt

    project_folder = Path(__file__).resolve().parent
    results_folder = project_folder / "outputs" / "image_editing"

    comparison_folder = (
        results_folder / "image_editing_comparison"
    )
    comparison_folder.mkdir(parents=True, exist_ok=True)

    runs = {}

    for name in ["flux", "qwen"]:
        files = sorted(
            results_folder.glob(f"{name}_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not files:
            raise FileNotFoundError(
                f"No saved measurements found for {name}."
            )

        selected_file = files[0]
        data = json.loads(
            selected_file.read_text(encoding="utf-8")
        )

        if not Path(data["output_file"]).is_file():
            raise FileNotFoundError(data["output_file"])

        runs[name] = {
            "source_json": str(selected_file),
            "data": data,
        }

        print(f"Using {name}: {selected_file.name}")

    flux_data = runs["flux"]["data"]
    qwen_data = runs["qwen"]["data"]

    # Check that the test inputs match.
    for field in ["source_image", "prompt", "width", "height", "seed"]:
        if flux_data[field] != qwen_data[field]:
            raise ValueError(
                f"The saved runs have different {field} values. "
                "Check the selected JSON files."
            )

    labels = [
        f"FLUX.2 Klein\n{flux_data['num_inference_steps']} steps",
        f"Qwen Image Edit\n{qwen_data['num_inference_steps']} steps",
    ]

    colours = ["#2563eb", "#f97316"]

    chart_data = [
        (
            [
                flux_data["generation_seconds"],
                qwen_data["generation_seconds"],
            ],
            "Image editing time",
            "Seconds — lower is better",
        ),
        (
            [
                flux_data["loading_seconds"],
                qwen_data["loading_seconds"],
            ],
            "Model loading time",
            "Seconds — lower is better",
        ),
        (
            [
                flux_data["peak_gpu_allocated_gib"],
                qwen_data["peak_gpu_allocated_gib"],
            ],
            "Peak GPU memory allocated",
            "GiB — lower is better",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for axis, (values, title, ylabel) in zip(axes, chart_data):
        bars = axis.bar(labels, values, color=colours)

        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(max(values) * 1.2, 0.1))
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Backyard pool mockups: selected model configurations"
    )

    fig.text(
        0.5,
        0.03,
        "One run per model; CPU offloading enabled.\n"
        "Qwen uses a community 4-bit checkpoint and VAE tiling. "
        "Memory measures PyTorch allocations.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    chart_path = comparison_folder / "time_and_memory.png"
    fig.savefig(chart_path, dpi=200)

    summary = {
        "evaluation_note": (
            "Comparison of selected configurations, "
            "not equal-step or equal-compute benchmarking."
        ),
        "runs": runs,
    }

    summary_path = comparison_folder / "comparison_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Chart saved: {chart_path}")
    print(f"Summary saved: {summary_path}")

    plt.show()



if __name__ == "__main__":
        comparison()