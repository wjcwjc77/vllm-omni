# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import time
import logging
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="text to image benchmark.")
    parser.add_argument("--device", type=str, default="cuda:7", help="Device to use for generation.")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--prompt", help="file path of the prompt text file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="vllm_omni_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
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

    with open(args.prompt, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    device = args.device
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )
    t0 = time.time()
    images = []
    for img in omni.generate(
        prompts,
        height=args.height,
        width=args.width,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        num_outputs_per_prompt=args.num_images_per_prompt,
    ):
        images.append(img)
    total_time = time.time() - t0
    avg_time = total_time / len(images)
    logging.info(f"[Profiler] Total generate time: {total_time:.3f}s")
    logging.info(f"[Profiler] Average time per image: {avg_time:.3f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "vllm_omni_output"
    if len(prompts) <= 1:
        images[0].save(output_path)
        logging.info(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            logging.info(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
