
#Seedance2.5 vs. Kling4.0
import os
import json
from datetime import datetime
import time
from pathlib import Path
from urllib.request import urlopen

import higgsfield_client

def seedance_ai():
    if not os.environ.get("HF_KEY"):
        raise RuntimeError("HF_KEY is missing.")

    project_folder = Path(__file__).resolve().parent
    video_path = project_folder / "outputs" / "video_editing" / "pool_shot.mp4"

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    output_folder  = project_folder / "outputs" / "video_editing"
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"seedance_{timestamp}.mp4"
    details_path = output_folder / f"seedance_{timestamp}.json"

    prompt = "Change this daytime swimming pool scene to a photorealistic nighttime scene. Make the sky datk navy blue and remove daylight. Illuminate the pool with soft blue underwater lights, creating natural reflections and gentle shimmering ripples. Add warm, subtle lighting along the existing poolside paving and warm light from the house windows. Keep the pool and surrounding area clearly visible with realistic nighttime exposure. Preserve the original pool shape, paving materials, house, fencing, landscaping and camera movement. Keep lighting and reflections consistent throughout the video. No people, text, or scene cuts.   "

    print("Uploading source video...")
    video_url = higgsfield_client.upload_file(str(video_path))

    arguments = {
        "prompt":prompt,
        "video_url":video_url,
        "resolution": "720p", 
        "output_format": "mp4",
        "generate_audio": False,
    }

    details = {
        "model": "bytedance/seedance-2.5/video-edit",
        "source_video": str(video_path),
        "arguments": arguments,
        "output_file": str(output_path),
    }

    def save_details():
        details_path.write_text(
            json.dumps(details, indent=2),
            encoding="utf-8",
        )

    def record_request(request_id):
        details["request_id"] = request_id
        save_details()
        print(f"Request submitted: {request_id}")



    print("Running Seedance")
    start = time.perf_counter()

    result = higgsfield_client.subscribe(
        details["model"],
        arguments=arguments,
        on_enqueue=record_request,
    )

    details["api_elapsed_seconds"] = round(
        time.perf_counter() - start, 2
    )
    details["result"] = result
    save_details()

    print("Downloading edited video...")
    with urlopen(result["video"]["url"], timeout=120) as response:
        with output_path.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)

    print(f"Video saved: {output_path}")
    print(f"Run details saved: {details_path}")
    print(f"API time: {details['api_elapsed_seconds']} seconds")   



def kling_ai():
    if not os.environ.get("HF_KEY"):
            raise RuntimeError("HF_KEY is missing.")
    
    project_folder = Path(__file__).resolve().parent
    video_path = project_folder / "outputs" / "video_editing" / "pool_shot.mp4"
    
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    
    output_folder  = project_folder / "outputs" / "video_editing"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"kling_{timestamp}.mp4"
    details_path = output_folder / f"kling_{timestamp}.json"
    
    prompt = "Change this daytime swimming pool scene to a photorealistic nighttime scene. Make the sky datk navy blue and remove daylight. Illuminate the pool with soft blue underwater lights, creating natural reflections and gentle shimmering ripples. Add warm, subtle lighting along the existing poolside paving and warm light from the house windows. Keep the pool and surrounding area clearly visible with realistic nighttime exposure. Preserve the original pool shape, paving materials, house, fencing, landscaping and camera movement. Keep lighting and reflections consistent throughout the video. No people, text, or scene cuts.   "
    
    print("Uploading source video...")
    video_url = higgsfield_client.upload_file(str(video_path))

    arguments = {
        "prompt": prompt,
        "video_url": video_url,
        "duration": 4,
        "resolution": "720p",
        "aspect_ratio": "16:9",
    }

    details = {
        "model": "kling-video/v4.0/video-edit",
        "source_video": str(video_path),
        "arguments": arguments,
        "output_file": str(output_path),
    }

    def save_details():
        details_path.write_text(
            json.dumps(details, indent=2),
            encoding="utf-8",
        )

    def record_request(request_id):
        details["request_id"] = request_id
        save_details()
        print(f"Request submitted: {request_id}")

    print("Running Kling — this submits one paid edit.")
    start = time.perf_counter()

    result = higgsfield_client.subscribe(
        details["model"],
        arguments=arguments,
        on_enqueue=record_request,
    )

    details["api_elapsed_seconds"] = round(
        time.perf_counter() - start, 2
    )
    details["result"] = result
    save_details()

    print("Downloading edited video...")
    with urlopen(result["video"]["url"], timeout=120) as response:
        with output_path.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)

    print(f"Video saved: {output_path}")
    print(f"Run details saved: {details_path}")
    print(f"API time: {details['api_elapsed_seconds']} seconds")


def comparison():
    import matplotlib.pyplot as plt

    project_folder = Path(__file__).resolve().parent
    results_folder = project_folder / "outputs" / "video_editing"
    chart_folder = results_folder / "video_editing_comparison"
    chart_folder.mkdir(parents=True, exist_ok=True)

    models = {
        "Seedance": "bytedance/seedance-2.5/video-edit",
        "Kling": "kling-video/v4.0/video-edit",
    }

    runs = {}

    # Find the latest successful saved run for each model.
    files = sorted(
        results_folder.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for file_path in files:
        data = json.loads(file_path.read_text(encoding="utf-8"))

        for name, model_id in models.items():
            if name in runs or data.get("model") != model_id:
                continue

            if data.get("result", {}).get("status") == "failed":
                continue

            output_path = Path(data.get("output_file", ""))

            if not output_path.is_file():
                continue

            runs[name] = {
                "file": str(file_path),
                "data": data,
            }

    if len(runs) != 2:
        raise RuntimeError(
            "Could not find successful saved runs for both models."
        )

    for name in models:
        print(f"{name}: {runs[name]['file']}")

    print("\nCheck these are your matched 720p runs.")

    # Enter the final charges shown in Higgsfield Billing.
    costs = {
        "Seedance": 2.13,
        "Kling": 0.64,
    }

    names = list(models)

    times = [
        runs[name]["data"]["api_elapsed_seconds"]
        for name in names
    ]

    prices = [costs[name] for name in names]
    colours = ["#2563eb", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    time_bars = axes[0].bar(names, times, color=colours)
    axes[0].bar_label(time_bars, fmt="%.2f", padding=4)
    axes[0].set_title("API response time")
    axes[0].set_ylabel("Seconds — lower is better")

    cost_bars = axes[1].bar(names, prices, color=colours)
    axes[1].bar_label(cost_bars, fmt="$%.2f", padding=4)
    axes[1].set_title("Actual request cost")
    axes[1].set_ylabel("USD — lower is better")

    for axis in axes:
        axis.set_ylim(0, max(axis.get_ylim()[1], 0.01) * 1.15)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Pool video editing: daytime to nighttime")
    fig.text(
        0.5, 0.02,
        "One run per model. Response time includes queuing and processing.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    chart_path = chart_folder / "time_and_cost.png"
    fig.savefig(chart_path, dpi=200)

    summary = {
        name: {
            "source_json": runs[name]["file"],
            "api_elapsed_seconds": times[index],
            "actual_cost_usd": costs[name],
        }
        for index, name in enumerate(names)
    }

    (chart_folder / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    comparison()