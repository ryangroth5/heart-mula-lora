# LoRA Fine-Tuning Guide

Fine-tune HeartMuLa 3B on your own audio + lyrics/tags pairs to capture a specific style.

---

## Overview

Training happens in three stages:

1. **Prepare** — run source separation, lyrics transcription, and tag generation on raw clips → `dataset.json`
2. **Encode** — convert `dataset.json` into pre-tokenized `.pt` files (runs once per dataset)
3. **Train** — run the LoRA fine-tuning loop on those token files → `lora.pt`

LoRA adapters are applied to the backbone attention projections (`q/k/v/output_proj`) only. Base model weights stay frozen. The final output is a small `lora.pt` (~4 MB at rank 8).

---

## Requirements

- Python venv at `./venv`
- `./ckpt/HeartMuLa-oss-3B/` — base model weights
- `./ckpt/HeartCodec-oss/` — codec weights (for encoding)
- `./ckpt/HeartTranscriptor-oss/` — transcription model (for dataset prep)
- `./ckpt/tokenizer.json` — BPE tokenizer
- GPU with ≥10 GB VRAM (single 30s clip ~7–9 GB with grad checkpointing)

---

## Step 1 — Prepare the dataset manifest

A template manifest is included at `data/dataset.example.json`. If you prefer to write it by hand, copy it to `data/dataset.json` and fill in your audio paths, lyrics, and tags. `data/dataset.json` is gitignored — it's local to your machine.

`scripts/prepare_lora_dataset.py` builds `dataset.json` automatically:

1. **Source separation** (torchaudio HDemucs) — isolates vocals from full mix
2. **Lyrics transcription** (HeartTranscriptorPipeline) — transcribes isolated vocals
3. **Tag generation** (Audio Flamingo 3 via subprocess) — generates genre/mood/tempo/instrument tags

```bash
venv/bin/python scripts/prepare_lora_dataset.py \
  --input_dir ./data/raw \
  --output_dir ./data/processed \
  --output_json ./data/dataset.json \
  --model_path ./ckpt \
  --af_python /path/to/conda/envs/audio-flamingo/bin/python
```

**All flags:**

| Flag | Default | Notes |
|---|---|---|
| `--input_dir` | required | Directory of `.m4a` / `.mp3` / `.wav` / `.flac` files |
| `--output_dir` | `./data/processed` | Where vocals `.wav` files are saved |
| `--output_json` | `./dataset.json` | Output manifest |
| `--model_path` | `./ckpt` | HeartMuLa checkpoint dir (for HeartTranscriptor) |
| `--af_python` | required (unless `--skip_tags`) | Python interpreter for the audio-flamingo conda env |
| `--af_tag_helper` | auto-discovered next to `--af_python` | Path to `audio_flamingo_tag.py` |
| `--skip_tags` | false | Skip Audio Flamingo, leave tags blank |
| `--skip_lyrics` | false | Skip HeartTranscriptor, leave lyrics blank |
| `--device` | `cuda` | torch device |
| `--dtype` | `fp32` | dtype for HeartTranscriptor |

**Note on audio-flamingo tags:** The tagger subprocess requires two env vars that are only set when you `conda activate audio-flamingo`. `prepare_lora_dataset.py` derives and injects them automatically from the `--af_python` path:
- `LD_LIBRARY_PATH` — adds the conda env's `nvidia/cusparselt/lib` path
- `CUDA_HOME` — set to the conda env prefix (parent of the `bin/` directory)

**Prepend tags manually:** After generation, use `prepend_tags.py` (from the [audio-flamingo repo](https://github.com/NVIDIA/audio-flamingo)) to add tags that the auto-labeler missed:

```bash
python /path/to/audio-flamingo/prepend_tags.py \
  --manifest ./data/dataset.json \
  --tags "gospel, gospel choir"
```

---

## Step 2 — Encode audio to tokens

```bash
venv/bin/python scripts/encode_dataset.py \
  --manifest ./data/dataset.json \
  --output_dir ./data/tokens \
  --model_path ./ckpt
```

Writes one `.pt` file per manifest entry. Run once per dataset.

**What it does per entry:**
1. Load audio, resample to 48 kHz mono
2. Extract 128-dim log-mel spectrogram at 50 fps (hop=960 samples)
3. Stack 4 consecutive mel frames → 512-dim features at 12.5 fps
4. Quantize via `HeartCodec.flow_matching.vq_embed` → 8 discrete codebook indices per frame
5. Tokenize tags (wrapped in `<tag>…</tag>` + BOS/EOS) and lyrics (BOS/EOS)
6. Save `{audio_tokens: Tensor[T,8], tags_ids: List[int], lyrics_ids: List[int]}`

**Implementation note:** The `heartlib` public API does not expose an audio tokenization path. The approach above (mel-spectrogram → vq_embed) is an empirically validated approximation. All 8192 codebook entries in all 8 codebooks are utilized (0 dead codes), and the codebooks produce varied, non-degenerate assignments across different audio frames.

**Verify:**
```bash
venv/bin/python -c "
import torch, glob
for f in sorted(glob.glob('./data/tokens/*.pt'))[:3]:
    d = torch.load(f, weights_only=True)
    print(f, d['audio_tokens'].shape)
"
```

---

## Step 3 — Train

### Single GPU

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

### Via the web UI

Start the server (`node server.js`), open `http://localhost:3000`, go to the **Train (LoRA)** tab:
1. Click **Encode Dataset** (Step 1) — runs encode_dataset.py, streams logs
2. Fill in hyperparameters and click **Start Training** (Step 2) — streams loss per step

### All training flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `./ckpt` | Checkpoint directory |
| `--version` | `3B` | Model size (`3B`, `7B`, `3B-happy-new-year`) |
| `--dataset_dir` | `./data/tokens` | Directory of `.pt` token files |
| `--output` | `./lora.pt` | Where to save LoRA adapter weights |
| `--epochs` | `3` | Full passes over the dataset |
| `--lora_rank` | `8` | LoRA rank — higher = more capacity, more VRAM |
| `--lora_alpha` | `16` | Effective scale = alpha / rank |
| `--lora_dropout` | `0.05` | Dropout on LoRA layers |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--grad_accum` | `4` | Gradient accumulation steps (increase if OOM) |

### What healthy training looks like

- **Start:** ~8–9 (near-random prediction over 8197-token vocab)
- **After ~10 epochs on 11 clips:** ~5.5–6.0
- **Converged (50+ epochs, 11 clips):** ~3.5–5.0
- **Ideal (more data):** ~2–4

Loss should trend down monotonically. Some noise is normal with batch size 1 and grad accumulation. If loss stays flat at ~8 after 20+ steps, check that `.pt` files loaded and LoRA was applied (script prints trainable param count — should be ~4.5M / 0.12% for 3B at rank 8).

---

## Step 4 — Generate with LoRA

### CLI

```bash
venv/bin/python scripts/run_music_generation.py \
  --model_path ./ckpt \
  --version 3B-happy-new-year \
  --lyrics "He is real, He is real / Everybody ought to know" \
  --tags "gospel, gospel choir, soul, spiritual, choir, call and response" \
  --save_path ./assets/gospel_gen.mp3 \
  --lora_path ./lora.pt \
  --lora_rank 8 \
  --lora_alpha 16
```

### Via the web UI

In the **Generate** tab, Model section — fill in **LoRA Adapter Path** (`./lora.pt`), **LoRA Rank** (8), and **LoRA Alpha** (16), then click Generate.

---

## Output

`lora.pt` is a PyTorch state dict containing only the LoRA adapter parameters (A/B matrices). At rank 8 on the 3B backbone it is approximately 4 MB.

---

## Troubleshooting

**`No .pt files found in ./data/tokens`**
Run Step 2 (Encode) first.

**`RuntimeError: CUDA out of memory`**
Increase `--grad_accum` (try 8 or 16). Use shorter clips. On multi-GPU: `accelerate launch --num_processes=N`.

**`FileNotFoundError: HeartTranscriptor-oss`**
Download it: `venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('HeartMuLa/HeartTranscriptor-oss', local_dir='./ckpt/HeartTranscriptor-oss')"`

**`libcusparseLt.so.0: cannot open shared object file`** (in audio-flamingo tagger)
The conda activation script normally sets `LD_LIBRARY_PATH`. When calling the tagger directly (not via prepare_lora_dataset.py), set it manually:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
```

**`CUDA_HOME does not exist, unable to compile CUDA op(s)`** (in audio-flamingo tagger)
Set `CUDA_HOME=$CONDA_PREFIX` — nvcc lives at `$CONDA_PREFIX/bin/nvcc`.

**`RuntimeError: expected mat1 and mat2 to have the same dtype`** (during training)
Mixed bfloat16/float32 issue. The model loads in bfloat16 but some heads (`codebook0_head`, `audio_head`) remain float32. `train_lora.py` casts appropriately at each boundary — if you hit this on new code, cast inputs with `.to(head.weight.dtype)` before the linear call.

**Loss is NaN from step 1**
Usually a learning rate issue. Try reducing `--lr` by 10×. Also verify gradient flow: LoRA A matrices should have non-zero gradients after the first backward pass.

**Tags from audio-flamingo are wrong genre** (e.g. "screamo" for gospel choir)
The tagger operates on the separated vocals `.wav` rather than the original audio. HDemucs separation artifacts can confuse the model. Add correct tags manually using `prepend_tags.py` — prepended tags take precedence.
