

import torch
from diffusers import SanaSprintPipeline
from diffusers import Lumina2Pipeline


def sana():
    print("Loading Sana")

    pipeline = SanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()

    prompt = "A photorealistic architectural concept of a modern Australian backyard, viewed diagonally from a slightly elevated position, showing the complete pool and surrounding garden. A rectangular swimming pool with clear turquoise water occupies the centre. Three wide submerged entry steps are visible at the near end, with a long underwater bench along the right interior wall. Pale sandstone coping surrounds the pool. Exactly two timber sun loungers with white cushions stand side by side on a timber deck to the left of the pool. A small round table sits between them. A white patio umbrella shades the loungers. A frameless glass fence runs behind the pool, with native grasses and flowering shrubs beyond it. A single-storey white house stands in the background. Warm late-afternoon sunlight comes from the left, casting consistent shadows to the right. Natural water reflections, realistic materials and straight architectural lines. No people, text, logos or watermarks."

    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=2,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    image.save("C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/outputs/concept_exploration/sana_pool_concept.png")
    print("Saved concept image")

def lumina2():
    print("Loading Lumina")

    pipe = Lumina2Pipeline.from_pretrained(
        "Alpha-VLLM/Lumina-Image-2.0",
        dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    prompt = "A photorealistic architectural concept of a modern Australian backyard, viewed diagonally from a slightly elevated position, showing the complete pool and surrounding garden. A rectangular swimming pool with clear turquoise water occupies the centre. Three wide submerged entry steps are visible at the near end, with a long underwater bench along the right interior wall. Pale sandstone coping surrounds the pool. Exactly two timber sun loungers with white cushions stand side by side on a timber deck to the left of the pool. A small round table sits between them. A white patio umbrella shades the loungers. A frameless glass fence runs behind the pool, with native grasses and flowering shrubs beyond it. A single-storey white house stands in the background. Warm late-afternoon sunlight comes from the left, casting consistent shadows to the right. Natural water reflections, realistic materials and straight architectural lines. No people, text, logos or watermarks."

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=50,
        cfg_trunc_ratio=0.25,
        cfg_normalization=True,

    generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

    image.save("C:/Users/Mineth/Documents/GitHub/Advanced Topics in AI-Model Comparison Assignment/outputs/concept_exploration/lumina_concept_concept.png")
    print("Saved lumina concept")



if __name__ == "__main__":
    lumina2()
