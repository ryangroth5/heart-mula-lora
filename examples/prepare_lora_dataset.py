#!/usr/bin/env python3
"""
Dataset preparation tool for LoRA fine-tuning.

Converts raw audio files into dataset.json entries with audio, lyrics, and tags fields.
Pipeline:
  1. Source separation via torchaudio HDemucs (isolates vocals)
  2. Lyrics transcription via HeartTranscriptorPipeline (on vocals track)
  3. Tag generation via Audio Flamingo 3 (subprocess to audio-flamingo conda env)

Usage:
    /home/ubooty/local/heart-mula/venv/bin/python \
        /home/ubooty/local/heart-mula/examples/prepare_lora_dataset.py \
        --input_dir ./data/raw \
        --output_dir ./data/processed \
        --output_json ./dataset.json \
        --model_path /home/ubooty/local/heart-mula/ckpt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
AF_TAG_HELPER = "/home/ubooty/local/audio-flamingo/audio_flamingo_tag.py"
HDEMUCS_SAMPLE_RATE = 44100


def separate_vocals(audio_path: Path, output_dir: Path, device: str) -> Path:
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device)
    model.eval()

    waveform, sr = torchaudio.load(str(audio_path))

    # Resample to 44100 Hz if needed
    if sr != HDEMUCS_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, HDEMUCS_SAMPLE_RATE)

    # HDemucs expects stereo; duplicate mono channel if needed
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.unsqueeze(0).to(device)  # (1, 2, T)

    with torch.no_grad():
        sources = model(waveform)  # (1, 4, 2, T) — drums, bass, other, vocals

    source_names = model.sources  # ["drums", "bass", "other", "vocals"]
    vocals_idx = source_names.index("vocals")
    vocals = sources[0, vocals_idx]  # (2, T)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
    torchaudio.save(str(vocals_path), vocals.cpu(), HDEMUCS_SAMPLE_RATE)
    return vocals_path


def transcribe_lyrics(vocals_path: Path, model_path: str, device: str, dtype: str) -> str:
    from heartlib.pipelines.lyrics_transcription import HeartTranscriptorPipeline

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    pipeline = HeartTranscriptorPipeline.from_pretrained(model_path, device=device, dtype=torch_dtype)
    result = pipeline(str(vocals_path), return_timestamps=False)
    return result["text"].strip()


AF_ENV = "/home/ubooty/miniconda3/envs/audio-flamingo"
AF_LD_LIBRARY_PATH = f"{AF_ENV}/lib/python3.11/site-packages/nvidia/cusparselt/lib"


def generate_tags(wav_path: Path, af_python: str) -> str:
    import os
    env = os.environ.copy()
    env["CUDA_HOME"] = AF_ENV
    env["LD_LIBRARY_PATH"] = AF_LD_LIBRARY_PATH + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    proc = subprocess.run(
        [af_python, AF_TAG_HELPER, str(wav_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    # llava may pollute stdout — find the last line starting with '{'
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise RuntimeError(f"No JSON in stdout: {proc.stdout[:200]}")
    data = json.loads(lines[-1])
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["tags"]


def main():
    parser = argparse.ArgumentParser(description="Prepare LoRA training dataset from audio files")
    parser.add_argument("--input_dir", required=True, help="Directory of audio files")
    parser.add_argument("--output_dir", default="./data/processed", help="Where to save vocals wav files")
    parser.add_argument("--output_json", default="./dataset.json", help="Output manifest path")
    parser.add_argument("--model_path", default="./ckpt", help="HeartMuLa checkpoint dir")
    parser.add_argument("--heartmula_path", default="/home/ubooty/local/heart-mula",
                        help="heart-mula repo root (prepended to sys.path for heartlib)")
    parser.add_argument("--af_python",
                        default="/home/ubooty/miniconda3/envs/audio-flamingo/bin/python",
                        help="Python interpreter for the audio-flamingo conda env")
    parser.add_argument("--skip_tags", action="store_true", help="Skip tag generation")
    parser.add_argument("--skip_lyrics", action="store_true", help="Skip lyrics transcription")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already have both tags and lyrics in sidecar .json or output_json")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                        help="dtype for HeartTranscriptor")
    args = parser.parse_args()

    # Ensure heartlib is importable
    heartmula_path = str(Path(args.heartmula_path).resolve())
    if heartmula_path not in sys.path:
        sys.path.insert(0, heartmula_path)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json)

    audio_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) in {input_dir}")

    # Build lookup from existing output_json for skip_existing
    existing = {}
    if args.skip_existing and output_json.exists():
        for e in json.loads(output_json.read_text()):
            existing[Path(e["audio"]).name] = e

    dataset = []
    for audio_file in audio_files:
        # Check sidecar first, then output_json, for existing metadata
        sidecar = audio_file.with_suffix(".json")
        if sidecar.exists():
            prev = json.loads(sidecar.read_text())
        else:
            prev = existing.get(audio_file.name, {})

        has_lyrics = bool(prev.get("lyrics", "").strip())
        has_tags = bool(prev.get("tags", "").strip())

        if args.skip_existing and has_lyrics and has_tags:
            print(f"\nSkipping (already complete): {audio_file.name}")
            dataset.append({"audio": str(audio_file.resolve()),
                            "lyrics": prev["lyrics"], "tags": prev["tags"]})
            continue

        print(f"\nProcessing: {audio_file.name}")
        entry = {
            "audio": str(audio_file.resolve()),
            "lyrics": prev.get("lyrics", ""),
            "tags": prev.get("tags", ""),
        }

        # Step 1: Tag generation — runs on the full mix wav
        if not args.skip_tags and not has_tags:
            # audio-flamingo requires wav; convert m4a on the fly if needed
            if audio_file.suffix.lower() == ".wav":
                mix_wav = audio_file
            else:
                mix_wav = output_dir / f"{audio_file.stem}_mix.wav"
                if not mix_wav.exists():
                    print("  Converting to wav for tagger...")
                    try:
                        waveform, sr = torchaudio.load(str(audio_file))
                        output_dir.mkdir(parents=True, exist_ok=True)
                        torchaudio.save(str(mix_wav), waveform, sr)
                    except Exception as e:
                        print(f"  WARNING: wav conversion failed for {audio_file.name}: {e}", file=sys.stderr)
                        mix_wav = None
            if mix_wav:
                print("  Generating tags...")
                try:
                    entry["tags"] = generate_tags(mix_wav, args.af_python)
                    print(f"  Tags: {entry['tags']}")
                except Exception as e:
                    print(f"  WARNING: tags failed for {audio_file.name}: {e}", file=sys.stderr)
        elif has_tags:
            print("  Tags: (using existing)")

        # Step 2: Source separation — vocals needed for lyrics transcription only
        vocals_path = None
        if not args.skip_lyrics and not has_lyrics:
            existing_vocals = output_dir / f"{audio_file.stem}_vocals.wav"
            if existing_vocals.exists():
                print(f"  Vocals: (reusing {existing_vocals.name})")
                vocals_path = existing_vocals
            else:
                print("  Separating vocals...")
                try:
                    vocals_path = separate_vocals(audio_file, output_dir, args.device)
                    print(f"  Vocals saved to: {vocals_path}")
                except Exception as e:
                    print(f"  WARNING: separation failed for {audio_file.name}: {e}", file=sys.stderr)

        # Step 3: Lyrics transcription
        if not args.skip_lyrics and not has_lyrics:
            if vocals_path:
                print("  Transcribing lyrics...")
                try:
                    entry["lyrics"] = transcribe_lyrics(vocals_path, args.model_path, args.device, args.dtype)
                    print(f"  Lyrics: {entry['lyrics'][:80]}{'...' if len(entry['lyrics']) > 80 else ''}")
                except Exception as e:
                    print(f"  WARNING: lyrics failed for {audio_file.name}: {e}", file=sys.stderr)
            else:
                print("  Lyrics: (skipped — separation failed)")
        elif has_lyrics:
            print("  Lyrics: (using existing)")

        dataset.append(entry)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nWrote {len(dataset)} entries to {output_json}")


if __name__ == "__main__":
    main()
