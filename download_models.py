#!/usr/bin/env python3
"""Download HeartMuLa model weights from Hugging Face into ./ckpt/"""
import argparse
import os
from huggingface_hub import snapshot_download, hf_hub_download

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ckpt")


def download(version="3B"):
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[1/3] Downloading HeartMuLa-oss-{version}...")
    snapshot_download(
        repo_id=f"HeartMuLa/HeartMuLa-oss-{version}",
        local_dir=os.path.join(CKPT_DIR, f"HeartMuLa-oss-{version}"),
        ignore_patterns=["*.gitattributes", "README.md"],
    )

    print("[2/3] Downloading HeartCodec-oss...")
    snapshot_download(
        repo_id="HeartMuLa/HeartCodec-oss-20260123",
        local_dir=os.path.join(CKPT_DIR, "HeartCodec-oss"),
        ignore_patterns=["*.gitattributes", "README.md"],
    )

    print("[3/3] Downloading tokenizer.json and gen_config.json...")
    for filename in ["tokenizer.json", "gen_config.json"]:
        hf_hub_download(
            repo_id="HeartMuLa/HeartMuLaGen",
            filename=filename,
            local_dir=CKPT_DIR,
        )

    print(f"\nDone! Models saved to {CKPT_DIR}")
    print("You can now set model_path to ./ckpt in the UI.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="3B", choices=["3B", "7B", "3B-happy-new-year"])
    args = p.parse_args()
    download(args.version)
