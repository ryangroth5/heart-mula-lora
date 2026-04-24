# HeartMuLa LoRA Fine-Tuning Pipeline

A fork of [HeartMuLa](https://heartmula.github.io/) that adds a complete LoRA fine-tuning pipeline and web UI. Bring your own audio — the tools here handle everything from source separation and transcription through tokenization, training, and generation.

---

## What this fork adds

| Component | Description |
|---|---|
| `scripts/encode_dataset.py` | Converts raw audio to pre-tokenized `.pt` files using a validated mel-spectrogram → VQ workaround (the public `heartlib` API has no audio tokenization path) |
| `scripts/train_lora.py` | LoRA fine-tuning loop with mixed-dtype handling, gradient checkpointing, and multi-GPU support via `accelerate` |
| `scripts/run_music_generation.py` | CLI inference wrapper with LoRA adapter support |
| `scripts/prepare_lora_dataset.py` | End-to-end dataset prep: HDemucs source separation → HeartTranscriptor lyrics → Audio Flamingo tags |
| `server.js` + `public/index.html` | Node.js web UI with real-time SSE log streaming for generate / encode / train / dataset-prep jobs |
| `download_models.py` | One-command model download from Hugging Face |

---

## Requirements

- Python 3.10, CUDA GPU with ≥10 GB VRAM
- Node.js ≥18 (for the web UI)
- Python venv at `./venv` with heartlib and torch (see setup below)

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/heart-mula-lora
cd heart-mula-lora

# Python venv
python3.10 -m venv venv
venv/bin/pip install heartlib torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
  torchtune==0.4.0 tokenizers==0.22.1 transformers==4.57.0 \
  "huggingface_hub>=0.34.0,<1.0" accelerate tqdm

# Node dependencies (web UI only)
npm install

# Download models
venv/bin/python download_models.py --version 3B
# Downloads HeartMuLa-oss-3B, HeartCodec-oss, tokenizer.json to ./ckpt/
# For the happy-new-year variant: --version 3B-happy-new-year
```

> **Note on `heartlib.heartcodec.models`:** The public `heartlib` 0.1.0 package ships without the internal `heartcodec/models/` subpackage (FlowMatching, ScalarModel implementation files). Training uses pre-tokenized `.pt` files so never calls HeartCodec directly; the encoding step accesses `HeartCodec.flow_matching.vq_embed` after checkpoint load. If you get `No module named 'heartlib.heartcodec.models'`, see `NOTES.md` for the stub fix.

---

## Workflow

### 1. Prepare a dataset

The dataset prep pipeline stitches together three external tools that each live outside this repo. See [Cross-project dependencies](#cross-project-dependencies) below for setup details on each one.

Put your source audio files (`.mp3`, `.m4a`, `.wav`, `.flac`) in `./data/raw/`.

**Option A — fully automated (all three tools):**
```bash
venv/bin/python scripts/prepare_lora_dataset.py \
  --input_dir ./data/raw \
  --output_dir ./data/processed \
  --output_json ./data/dataset.json \
  --model_path ./ckpt
```

**Option B — skip Audio Flamingo tagging** (if the conda env isn't set up):
```bash
venv/bin/python scripts/prepare_lora_dataset.py \
  --input_dir ./data/raw \
  --output_dir ./data/processed \
  --output_json ./data/dataset.json \
  --model_path ./ckpt \
  --skip_tags
```

**Option C — skip everything, write the manifest by hand:**

Copy the template and edit it:
```bash
cp data/dataset.example.json data/dataset.json
# then edit data/dataset.json with your audio paths, lyrics, and tags
```

All three options produce the same `dataset.json` format. Tags and lyrics can be empty strings — the model trains on whatever conditioning you provide.

### 2. Encode audio to tokens

```bash
venv/bin/python scripts/encode_dataset.py \
  --manifest ./data/dataset.json \
  --output_dir ./data/tokens \
  --model_path ./ckpt
```

Run once per dataset. Writes one `.pt` file per entry into `./data/tokens/`.

### 3. Train

```bash
venv/bin/python scripts/train_lora.py \
  --model_path ./ckpt \
  --dataset_dir ./data/tokens \
  --output ./lora.pt \
  --epochs 50 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --grad_accum 4
```

See `TRAINING.md` for the full flag reference, expected loss curves, and OOM troubleshooting.

### 4. Generate

```bash
venv/bin/python scripts/run_music_generation.py \
  --model_path ./ckpt \
  --version 3B-happy-new-year \
  --lyrics "He is real, He is real / Everybody ought to know" \
  --tags "gospel, gospel choir, soul, spiritual" \
  --save_path ./assets/output.mp3 \
  --lora_path ./lora.pt \
  --lora_rank 8 \
  --lora_alpha 16
```

---

## Web UI

```bash
node server.js
# Open http://localhost:3000
```

Four tabs:

| Tab | What it does |
|---|---|
| **Generate** | Full form for generation — model config, sampling params, LoRA adapter |
| **Train (LoRA)** | Step 1: encode dataset → Step 2: train with live loss stream |
| **Dataset** | Download audio from YouTube, run prepare pipeline, edit per-file tags/lyrics |
| **Download** | Batch YouTube download via `yt-dlp` |

All GPU jobs stream logs in real time via Server-Sent Events. Starting a new generate or train job kills the previous one to free VRAM — this is intentional.

---

## Audio tokenization workaround

The `heartlib` public API does not expose an audio tokenization path. `HeartCodec` has a `flow_matching` module (ResidualVQ, 8 codebooks × 8192 entries, 512-dim input at 12.5 fps) but no shipped encoder that maps audio → 512-dim features.

**Validated workaround** (implemented in `encode_dataset.py`):

```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000, n_fft=4096, hop_length=960, n_mels=128,
)
mel = mel_transform(waveform.squeeze(0))
mel = (mel + 1e-9).log()
mel = (mel - mel.mean()) / (mel.std() + 1e-9)
T_trim = (mel.shape[1] // 4) * 4
features = mel[:, :T_trim].reshape(128, -1, 4).permute(1, 0, 2).reshape(-1, 512)
_, indices, _ = codec.flow_matching.vq_embed(features.unsqueeze(0))
audio_tokens = indices.squeeze(0).cpu()  # (T, 8)
```

Stacking 4 consecutive 128-dim mel frames gives 512-dim input at 12.5 fps matching the VQ's expected rate. The `project_in` layer inside `vq_embed` (512→32) is robust to this input distribution — validated at 0 dead codes across all 8 codebooks on a test dataset.

---

## LoRA details

- Adapters applied to `q/k/v/output_proj` in all 28 backbone attention layers
- Rank 8, alpha 16 → ~4.5 M trainable params (0.12% of 3B)
- Output is a `.pt` state dict (adapter weights only, ~4 MB at rank 8)
- Mixed dtype: backbone in bfloat16, heads in float32 — train_lora.py casts at each boundary

---

## Worked example: gospel choir fine-tuning

The pipeline was developed and validated on a set of gospel choir recordings (11 clips). The source audio is not included here as it is copyrighted material — this is documentation of the process.

**Dataset prep:** HDemucs separated vocals; HeartTranscriptor transcribed lyrics; tags were generated by Audio Flamingo and corrected with `prepend_tags.py` to ensure "gospel, gospel choir" appeared.

**Training observations** (11 clips, rank 8, `grad_accum=4`):

| Epoch | Loss range |
|---|---|
| 1 | ~8.3 (near-random baseline for 8197-token vocab) |
| ~10 | ~5.5–6.0 |
| ~50 | ~4.5–4.8 |

Loss plateaued around 4.5 after ~30 epochs. At loss < 5 the model predicts audio tokens significantly better than random (random baseline ≈ 9.01 for vocab size 8197). For more data or higher capacity, try increasing LoRA rank or adding clips.

---

## File layout

```
ckpt/
  HeartMuLa-oss-3B/          # base model weights (download with download_models.py)
  HeartMuLa-oss-3B-happy-new-year/
  HeartCodec-oss/             # codec weights
  HeartTranscriptor-oss/      # transcription model (optional, for dataset prep)
  tokenizer.json
  gen_config.json
data/
  raw/                        # your source audio (.m4a, .mp3, etc.)
  processed/                  # separated vocals (.wav)
  dataset.example.json        # template manifest (copy to dataset.json and edit)
  dataset.json                # your manifest — gitignored, generated at runtime
  tokens/                     # pre-tokenized .pt files (generated by encode_dataset.py)
scripts/
  encode_dataset.py
  train_lora.py
  run_music_generation.py
  prepare_lora_dataset.py
public/index.html             # single-page web UI
server.js                     # Express server with SSE streaming
download_models.py            # HuggingFace model downloader
TRAINING.md                   # user-facing training guide with all flag references
NOTES.md                      # developer notes on API gaps, dtype issues, and workarounds
```

---

## Cross-project dependencies

The dataset preparation step (`prepare_lora_dataset.py`) stitches together three external tools that each live outside this repo. None are bundled here — each has its own install path and environment requirements.

---

### 1. HDemucs — source separation

**Repo:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)  
**Used via:** `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS` (bundled in torchaudio ≥2.1)  
**What it does:** Separates a mixed audio file into stems (drums, bass, other, vocals). The pipeline passes the isolated vocals track to HeartTranscriptor and the full mix to Audio Flamingo.

No separate install needed — it ships with torchaudio and downloads weights (~200 MB) on first use to `~/.cache`. Input must be **44100 Hz stereo**; `prepare_lora_dataset.py` resamples automatically.

Key constraints:
- Short clips (<5 s) can produce near-silent vocals — the transcriber will return an empty string in that case
- Output stems are saved as float32 wav at 44100 Hz

---

### 2. HeartTranscriptor — lyrics transcription

**Weights:** [HeartMuLa/HeartTranscriptor-oss](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss) on Hugging Face  
**Part of:** [heartlib](https://heartmula.github.io/) (`HeartTranscriptorPipeline`)  
**What it does:** Transcribes lyrics from the separated vocals wav. Runs inside the same `./venv` as the rest of the pipeline.

Download:
```bash
venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartTranscriptor-oss', local_dir='./ckpt/HeartTranscriptor-oss')
"
```

**Path convention:** `HeartTranscriptorPipeline.from_pretrained(model_path)` appends `"HeartTranscriptor-oss"` to the path internally. Pass the base `./ckpt` directory — not `./ckpt/HeartTranscriptor-oss` — or you get a double-suffix error.

---

### 3. Audio Flamingo 3 — genre/mood/instrument tagging

**Repo:** [NVIDIA/audio-flamingo](https://github.com/NVIDIA/audio-flamingo)  
**Model weights:** [`nvidia/audio-flamingo-3`](https://huggingface.co/nvidia/audio-flamingo-3) on Hugging Face  
**What it does:** Generates a comma-separated tag string (genre, mood, tempo, instruments, vocals) from audio. Tags are used as conditioning text during LoRA training.

Audio Flamingo has heavy CUDA dependencies that conflict with the heartlib venv, so it **must run in its own conda environment**:

```bash
# One-time setup
conda create -n audio-flamingo python=3.11
conda activate audio-flamingo
pip install -r /path/to/audio-flamingo/requirements.txt
```

`prepare_lora_dataset.py` calls it via subprocess and injects the two env vars that `conda activate` normally sets but subprocess doesn't inherit:

| Env var | Value |
|---|---|
| `LD_LIBRARY_PATH` | `$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparselt/lib` |
| `CUDA_HOME` | path to the audio-flamingo conda env prefix |

You can override the conda env python path with `--af_python`:
```bash
venv/bin/python scripts/prepare_lora_dataset.py \
  --input_dir ./data/raw \
  --output_json ./data/dataset.json \
  --af_python /path/to/your/conda/envs/audio-flamingo/bin/python
```

**Known issue — hallucinated tags on degraded audio:** If HDemucs separation produces artifacts, Audio Flamingo can generate plausible-sounding but wrong tags. Use `--skip_tags` and add tags manually, or use the web UI's Dataset tab to edit them per-file before building the manifest.

**m4a not supported:** The underlying `llava.Sound()` loader rejects m4a. `prepare_lora_dataset.py` always passes the HDemucs-separated vocals wav (not the original m4a) to the tagger.

---

### How the three tools connect

```
data/raw/*.m4a  (or .mp3 / .wav / .flac)
       │
       ▼
 [HDemucs]  torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
       │
       ├──── vocals.wav ──▶ [HeartTranscriptor]  →  lyrics string
       │
       └──── full mix  ──▶ [Audio Flamingo 3]   →  tag string
                                    │
                                    ▼
                             dataset.json  [{audio, lyrics, tags}, ...]
                                    │
                                    ▼
                          [encode_dataset.py]  →  data/tokens/*.pt
                                    │
                                    ▼
                           [train_lora.py]  →  lora.pt
```

---

## Credits

Base model and `heartlib`: [HeartMuLa](https://heartmula.github.io/) by the HeartMuLa team.  
Source separation: [HDemucs](https://github.com/facebookresearch/demucs) by Facebook Research, via [torchaudio](https://github.com/pytorch/audio).  
Audio tagging: [Audio Flamingo 3](https://github.com/NVIDIA/audio-flamingo) by NVIDIA.
