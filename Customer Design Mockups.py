#Flux.2 Klein 4B vs. Qwen Image Edit

import torch
from pathlib import Path
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def flux():
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype = torch.bfloat16
        )
        pipe.enable_model_cpu_offload()

        backyard = load_image("C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/customer_design_mockups/backyard_image.jpg")

        prompt = "Edit this backyard photograph to add a compact rectangular inground swimming pool within the lawn on the left side of the image. Position its long axis roughly parallel to the timber fence on the left, extending from the foreground towards the rear fence. Keep the pool entirely within the grass area, leaving visible space between it, the fences and the house. Give the pool a pale blue interior, clear turquoise water, realistic depth, submerged entry steps at the far end and a narrow underwater bench along one side. Add subtle ripples and reflections of the pink-purple sunset sky. Use pale grey stone coping around the pool with a narrow matching paved border, retaining grass around the outside. Add transparent frameless glass pool fencing with discreet metal spigots around the pool area, including a glass gate facing the existing concrete path. Keep the house entrance and right-hand side passage unobstructed. Preserve the original brick house, roof, solar panels, windows, visible interior, timber boundary fences, neighbouring house, existing concrete paths and flower bed. Preserve the original camera position, perspective, image framing and sunset sky. Match the existing dusk lighting, with subtle underwater pool lights and natural reflections. Produce a photorealistic architectural design mockup. Keep all changes confined to the pool and its immediate surroundings. No people, text, logos or additional buildings."

        image = pipe(
            image=backyard,
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

        image.save("C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/customer_design_mockups/flux_mockup.png")


def qwen_image():
        project_folder = Path(__file__).resolve().parent
        image_folder = project_folder / "customer_design_mockups"

        backyard = Image.open(
                image_folder / "backyard_image.jpg"
        ).convert("RGB")

        print("Loading Qwen Image Edit")

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "toandev/Qwen-Image-Edit-2511-4bit",
                torch_dtype = torch.bfloat16
        )

        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_tiling()

        prompt = "Edit this backyard photograph to add a compact rectangular inground swimming pool within the lawn on the left side of the image. Position its long axis roughly parallel to the timber fence on the left, extending from the foreground towards the rear fence. Keep the pool entirely within the grass area, leaving visible space between it, the fences and the house. Give the pool a pale blue interior, clear turquoise water, realistic depth, submerged entry steps at the far end and a narrow underwater bench along one side. Add subtle ripples and reflections of the pink-purple sunset sky. Use pale grey stone coping around the pool with a narrow matching paved border, retaining grass around the outside. Add transparent frameless glass pool fencing with discreet metal spigots around the pool area, including a glass gate facing the existing concrete path. Keep the house entrance and right-hand side passage unobstructed. Preserve the original brick house, roof, solar panels, windows, visible interior, timber boundary fences, neighbouring house, existing concrete paths and flower bed. Preserve the original camera position, perspective, image framing and sunset sky. Match the existing dusk lighting, with subtle underwater pool lights and natural reflections. Produce a photorealistic architectural design mockup. Keep all changes confined to the pool and its immediate surroundings. No people, text, logos or additional buildings."

        with torch.inference_mode():
                output = pipeline(
                        image=[backyard],
                        prompt = prompt,
                        height=1024,
                        width=1024,
                        generator=torch.Generator(device="cpu").manual_seed(42),
                        true_cfg_scale=4.0,
                        negative_prompt=" ",
                        num_inference_steps=40,
                        guidance_scale=1.0,
                        num_images_per_prompt=1
                )
        output_path = image_folder / "qwen_mockup.png"
        output.images[0].save(output_path)

        print(f"Image saved: {output_path}")


if __name__ == "__main__":
        qwen_image()