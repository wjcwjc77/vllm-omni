# run the baseline -> Hugging Face Diffusers
import torch
import time
import argparse
import logging
from diffusers import ZImagePipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="text to image benchmark.")
    parser.add_argument("--device", type=str, default="cuda:7", help="Device to use for generation.")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--prompt", help="file path of the prompt text file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0,
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=9,
        help="Number of denoising steps for the diffusion sampler.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # 1. Load the pipeline
    # Use bfloat16 for optimal performance on supported GPUs
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(args.device)

    # [Optional] Attention Backend
    # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
    # pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
    # pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

    # [Optional] Model Compilation
    # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
    # pipe.transformer.compile()

    # [Optional] CPU Offloading
    # Enable CPU offloading for memory-constrained devices.
    # pipe.enable_model_cpu_offload()

    with open(args.prompt, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    total_images = len(prompts)
    t0 = time.time()
    # 2. Generate Image
    images = pipe(
        prompt=prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.cfg_scale,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator(args.device).manual_seed(args.seed),
    ).images
    total_time = time.time() - t0
    avg_time = total_time / total_images
    logging.info(f"[Profiler] Total generate time: {total_time:.3f}s")
    logging.info(f"[Profiler] Average time per image: {avg_time:.3f}s")
    for idx, image in enumerate(images):
        output_path = args.output
        if len(images) > 1:
            output_path = args.output.replace('.png', f'_{idx}.png')
        image.save(output_path)

if __name__ == "__main__":
    main()