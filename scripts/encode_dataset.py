"""
encode_dataset.py — Convert raw audio + lyrics/tags pairs into pre-tokenized .pt files.

Usage:
  venv/bin/python scripts/encode_dataset.py \
    --manifest dataset.json \
    --output_dir ./data/tokens \
    --model_path ./ckpt

dataset.json format:
  [{"audio": "./data/song1.mp3", "lyrics": "verse 1...", "tags": "pop, female vocals"}, ...]
"""

import argparse
import json
import os
import sys

import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description="Encode audio + lyrics/tags into token .pt files")
    parser.add_argument("--manifest", required=True, help="Path to dataset.json")
    parser.add_argument("--output_dir", default="./data/tokens", help="Output directory for .pt files")
    parser.add_argument("--model_path", default="./ckpt", help="Path to checkpoint directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        entries = json.load(f)
    print(f"Found {len(entries)} entries in manifest.")

    # Load codec
    print("Loading HeartCodec...")
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    codec_path = os.path.join(args.model_path, "HeartCodec-oss")
    codec = HeartCodec.from_pretrained(codec_path)
    codec.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    BOS_ID = 128000
    EOS_ID = 128001
    TARGET_SR = 48000

    for idx, entry in enumerate(entries):
        audio_path = entry["audio"]
        lyrics_text = entry.get("lyrics", "")
        tags_text = entry.get("tags", "")

        print(f"[{idx+1}/{len(entries)}] Processing {audio_path} ...")

        # Load and resample audio to 48 kHz mono
        waveform, sr = torchaudio.load(audio_path)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Shape: (1, T_samples)

        # Encode audio → discrete tokens via HeartCodec vq_embed.
        # The vq_embed expects 512-dim acoustic features at 12.5 Hz.
        # We use 128-dim log-mel at 50 fps (hop=960) stacked across 4 frames → 512-dim @ 12.5 fps.
        # This matches the approximate input distribution the vq_embed was trained on.
        with torch.no_grad():
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SR,
                n_fft=4096,
                hop_length=960,
                n_mels=128,
            )
            mel = mel_transform(waveform.squeeze(0))  # (128, T_mel)
            mel = (mel + 1e-9).log()
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)
            # Stack 4 consecutive mel frames → (T_mel//4, 512)
            T_mel = mel.shape[1]
            T_trim = (T_mel // 4) * 4
            features = mel[:, :T_trim].reshape(128, -1, 4).permute(1, 0, 2).reshape(-1, 512)
            _, indices, _ = codec.flow_matching.vq_embed(features.unsqueeze(0))
            audio_tokens = indices.squeeze(0).cpu()  # (T_frames, 8)

        print(f"  Audio tokens shape: {audio_tokens.shape}")

        # Tokenize tags: wrap with <tag>…</tag>, prepend BOS, append EOS
        tags_wrapped = f"<tag>{tags_text}</tag>" if tags_text else "<tag></tag>"
        tags_enc = tokenizer.encode(tags_wrapped)
        tags_ids = [BOS_ID] + tags_enc.ids + [EOS_ID]

        # Tokenize lyrics: prepend BOS, append EOS
        lyrics_enc = tokenizer.encode(lyrics_text)
        lyrics_ids = [BOS_ID] + lyrics_enc.ids + [EOS_ID]

        # Save output
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(args.output_dir, f"{idx:04d}_{stem}.pt")
        torch.save({
            "audio_tokens": audio_tokens,  # Tensor[T, 8]
            "tags_ids": tags_ids,          # List[int]
            "lyrics_ids": lyrics_ids,      # List[int]
        }, out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone. {len(entries)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()
