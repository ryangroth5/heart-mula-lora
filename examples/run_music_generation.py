#!/usr/bin/env python3
"""CLI wrapper for HeartMuLa music generation."""
import argparse
import os
import sys
import numpy as np
import torch
from heartlib.pipelines.music_generation import HeartMuLaGenPipeline

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    p = argparse.ArgumentParser(description="HeartMuLa music generation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--version", default="3B", choices=["3B", "7B", "3B-happy-new-year"])
    p.add_argument("--lyrics", required=True, help="Lyrics text or path to file")
    p.add_argument("--tags", required=True, help="Tags text or path to file")
    p.add_argument("--save_path", default="./assets/output.mp3")
    p.add_argument("--max_audio_length_ms", type=int, default=240000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--mula_device", default="cuda")
    p.add_argument("--codec_device", default="cuda")
    p.add_argument("--mula_dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--codec_dtype", default="fp32", choices=list(DTYPE_MAP))
    p.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    p.add_argument("--lazy_load", action="store_true")
    p.add_argument("--lora_path", default=None, help="Path to lora.pt adapter weights (optional)")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank used during training (must match)")
    p.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha used during training (must match)")
    return p.parse_args()


def main():
    args = parse_args()

    seed = args.seed if args.seed != -1 else int(np.random.randint(0, 2147483647))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"[heartmula] Loading pipeline from {args.model_path} (version={args.version})")
    print(f"[heartmula] Devices: mula={args.mula_device}, codec={args.codec_device}")
    print(f"[heartmula] Dtypes: mula={args.mula_dtype}, codec={args.codec_dtype}")
    print(f"[heartmula] lazy_load={args.lazy_load}")
    if args.lora_path:
        print(f"[heartmula] LoRA adapter: {args.lora_path} (rank={args.lora_rank}, alpha={args.lora_alpha})")
    sys.stdout.flush()

    # Resolve devices
    mula_device = torch.device(args.mula_device)
    codec_device = torch.device(args.codec_device)
    device = {"mula": mula_device, "codec": codec_device}
    dtype = {"mula": DTYPE_MAP[args.mula_dtype], "codec": DTYPE_MAP[args.codec_dtype]}

    pipeline = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=args.model_path,
        device=device,
        dtype=dtype,
        version=args.version,
        lazy_load=args.lazy_load,
    )

    if args.lora_path:
        if not os.path.exists(args.lora_path):
            print(f"[heartmula] WARNING: lora_path {args.lora_path} not found, skipping LoRA load")
        else:
            import importlib.util, pathlib
            _spec = importlib.util.spec_from_file_location(
                "train_lora",
                pathlib.Path(__file__).parent / "train_lora.py"
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            apply_lora = _mod.apply_lora
            model = pipeline.mula
            apply_lora(model, args.lora_rank, args.lora_alpha, lora_dropout=0.0)
            lora_state = torch.load(args.lora_path, map_location="cpu", weights_only=True)
            model.load_state_dict(lora_state, strict=False)
            model.to(mula_device)
            print(f"[heartmula] LoRA weights loaded from {args.lora_path}")
            sys.stdout.flush()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # Resolve and echo the actual tags/lyrics content (may be file paths)
    tags_content = args.tags
    if os.path.isfile(tags_content):
        with open(tags_content, encoding="utf-8") as f:
            tags_content = f.read()
    lyrics_content = args.lyrics
    if os.path.isfile(lyrics_content):
        with open(lyrics_content, encoding="utf-8") as f:
            lyrics_content = f.read()

    print(f"[heartmula] Tags:   {tags_content!r}")
    print(f"[heartmula] Lyrics: {lyrics_content[:120]!r}{'...' if len(lyrics_content) > 120 else ''}")
    print(f"[heartmula] Starting generation...")
    print(f"[heartmula] max_audio_length_ms={args.max_audio_length_ms}, temperature={args.temperature}, topk={args.topk}, cfg_scale={args.cfg_scale}, seed={seed}")
    sys.stdout.flush()

    with torch.no_grad():
        pipeline(
            inputs={"lyrics": args.lyrics, "tags": args.tags},
            max_audio_length_ms=args.max_audio_length_ms,
            temperature=args.temperature,
            topk=args.topk,
            cfg_scale=args.cfg_scale,
            save_path=args.save_path,
        )

    print(f"[heartmula] Done! Audio saved to {args.save_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
