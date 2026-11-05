

import torch
import json
import time
from datetime import datetime
from pathlib import Path
from diffusers import SanaSprintPipeline
from diffusers import Lumina2Pipeline


def sana():
    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "outputs" / "image_generation"
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_folder /  f"sana_{timestamp}.png"
    details_path = output_folder / f"sana_{timestamp}.json"

    model_id = (
        "Efficient-Large-Model/"
        "Sana_Sprint_1.6B_1024px_diffusers"
    )

    prompt = "A photorealistic architectural concept of a modern Australian backyard, viewed diagonally from a slightly elevated position, showing the complete pool and surrounding garden. A rectangular swimming pool with clear turquoise water occupies the centre. Three wide submerged entry steps are visible at the near end, with a long underwater bench along the right interior wall. Pale sandstone coping surrounds the pool. Exactly two timber sun loungers with white cushions stand side by side on a timber deck to the left of the pool. A small round table sits between them. A white patio umbrella shades the loungers. A frameless glass fence runs behind the pool, with native grasses and flowering shrubs beyond it. A single-storey white house stands in the background. Warm late-afternoon sunlight comes from the left, casting consistent shadows to the right. Natural water reflections, realistic materials and straight architectural lines. No people, text, logos or watermarks."

    print("Loading Sana")
    loading_start = time.perf_counter()

    pipeline = SanaSprintPipeline.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()

    loading_seconds = time.perf_counter() - loading_start
    generator = torch.Generator(device="cuda").manual_seed(42)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Generating Sana image...")
    start = time.perf_counter()

    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=2,
            generator=generator,
        ).images[0]

    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    image.save(image_path)

    details = {
        "name": "sana",
        "model": model_id,
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "generator_device": "cuda",
        "num_inference_steps": 2,
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

def lumina2():
    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "outputs" / "image_generation"
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_folder / f"lumina_{timestamp}.png"
    details_path = output_folder / f"lumina_{timestamp}.json"

    model_id = "Alpha-VLLM/Lumina-Image-2.0"

    prompt = "A photorealistic architectural concept of a modern Australian backyard, viewed diagonally from a slightly elevated position, showing the complete pool and surrounding garden. A rectangular swimming pool with clear turquoise water occupies the centre. Three wide submerged entry steps are visible at the near end, with a long underwater bench along the right interior wall. Pale sandstone coping surrounds the pool. Exactly two timber sun loungers with white cushions stand side by side on a timber deck to the left of the pool. A small round table sits between them. A white patio umbrella shades the loungers. A frameless glass fence runs behind the pool, with native grasses and flowering shrubs beyond it. A single-storey white house stands in the background. Warm late-afternoon sunlight comes from the left, casting consistent shadows to the right. Natural water reflections, realistic materials and straight architectural lines. No people, text, logos or watermarks."

    print("Loading Lumina")
    loading_start = time.perf_counter()

    pipeline = Lumina2Pipeline.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()

    loading_seconds = time.perf_counter() - loading_start
    generator = torch.Generator(device="cpu").manual_seed(42)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Generating Lumina image...")
    start = time.perf_counter()

    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=50,
            cfg_trunc_ratio=0.25,
            cfg_normalization=True,
            generator=generator,
        ).images[0]

    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    image.save(image_path)

    details = {
        "name": "lumina",
        "model": model_id,
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "generator_device": "cpu",
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "cfg_trunc_ratio": 0.25,
        "cfg_normalization": True,
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
    results_folder = project_folder / "outputs" / "image_generation"

    comparison_folder = (
        results_folder / "image_generation_comparison"
    )
    comparison_folder.mkdir(parents=True, exist_ok=True)

    runs = {}

    for name in ["sana", "lumina"]:
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

        runs[name] = {
            "source_json": str(selected_file),
            "data": json.loads(
                selected_file.read_text(encoding="utf-8")
            ),
        }

        print(f"Using {name}: {selected_file.name}")

    sana_data = runs["sana"]["data"]
    lumina_data = runs["lumina"]["data"]

    # Check that the main test inputs match.
    for field in ["prompt", "width", "height", "seed"]:
        if sana_data[field] != lumina_data[field]:
            raise ValueError(
                f"The models have different {field} values. "
                "Check the selected JSON files."
            )

    labels = ["SANA-Sprint", "Lumina 2"]
    colours = ["#2563eb", "#f97316"]

    generation_times = [
        sana_data["generation_seconds"],
        lumina_data["generation_seconds"],
    ]

    loading_times = [
        sana_data["loading_seconds"],
        lumina_data["loading_seconds"],
    ]

    memory_usage = [
        sana_data["peak_gpu_allocated_gib"],
        lumina_data["peak_gpu_allocated_gib"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    chart_data = [
        (
            generation_times,
            "Image generation time",
            "Seconds — lower is better",
        ),
        (
            loading_times,
            "Model loading time",
            "Seconds — lower is better",
        ),
        (
            memory_usage,
            "Peak GPU memory allocated",
            "GiB — lower is better",
        ),
    ]

    for axis, (values, title, ylabel) in zip(axes, chart_data):
        bars = axis.bar(labels, values, color=colours)

        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(max(values) * 1.2, 0.1))
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Pool concept generation: SANA-Sprint vs Lumina 2"
    )

    fig.text(
        0.5,
        0.02,
        "One run per model; no warm-up. CPU offloading enabled. "
        "SANA: 2 steps; Lumina: 50 steps. "
        "Memory measures PyTorch allocations.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    chart_path = comparison_folder / "time_and_memory.png"
    fig.savefig(chart_path, dpi=200)

    summary = {}

    for name, run in runs.items():
        data = run["data"]

        summary[name] = {
            "model": data["model"],
            "source_json": run["source_json"],
            "output_file": data["output_file"],
            "generation_seconds": data["generation_seconds"],
            "loading_seconds": data["loading_seconds"],
            "peak_gpu_allocated_gib": (
                data["peak_gpu_allocated_gib"]
            ),
            "num_inference_steps": data["num_inference_steps"],
            "generator_device": data["generator_device"],
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
