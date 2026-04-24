# Developer Notes — HeartMuLa LoRA Pipeline

Internal debugging knowledge accumulated while building the gospel choir fine-tuning pipeline. Complements `TRAINING.md` (user docs) with the "why it's broken and how we fixed it" layer.

---

## heartlib API gaps

### No audio tokenization path in public API

The public `heartlib` package ships no method to convert audio → discrete tokens. The obvious candidate was `codec.sq_codec.encode()` — it doesn't exist. The actual children of `HeartCodec` are:

- `flow_matching` — `ResidualVQ` with 8 codebooks, 8192 entries each, expects **512-dim** input at ~12.5 fps
- `scalar_model` — CNN encoder/decoder, 128-dim latent at ~50 fps

`scalar_model.encode()` outputs 128-dim but `flow_matching.vq_embed` needs 512-dim. The BeSTRQ encoder that maps 512-dim features to codes was not shipped.

**Workaround (validated):** 128-dim log-mel at 50fps (hop=960 samples at 48kHz), stack 4 consecutive frames → 512-dim at 12.5fps, pass directly to `flow_matching.vq_embed`:

```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000, n_fft=4096, hop_length=960, n_mels=128,
)
mel = mel_transform(waveform.squeeze(0))        # (128, T_mel)
mel = (mel + 1e-9).log()
mel = (mel - mel.mean()) / (mel.std() + 1e-9)
T_trim = (mel.shape[1] // 4) * 4
features = mel[:, :T_trim].reshape(128, -1, 4).permute(1, 0, 2).reshape(-1, 512)
_, indices, _ = codec.flow_matching.vq_embed(features.unsqueeze(0))
audio_tokens = indices.squeeze(0).cpu()         # (T, 8)
```

Validated by checking that all 8192 entries in all 8 codebooks are utilized (0 dead codes) across 11 gospel choir clips. The `project_in` layer inside `vq_embed` (512→32) is robust to the exact input distribution.

### HeartTranscriptorPipeline path convention

`HeartTranscriptorPipeline.from_pretrained(model_path)` internally appends `"HeartTranscriptor-oss"` to the path. Do **not** pass `{model_path}/HeartTranscriptor-oss` — that produces a double-suffix error:
```
Expected to find checkpoint at ckpt/HeartTranscriptor-oss/HeartTranscriptor-oss
```

Correct call:
```python
pipeline = HeartTranscriptorPipeline.from_pretrained(
    model_path,   # just the base ckpt dir, e.g. "./ckpt"
    device=device, dtype=dtype
)
```

Download the checkpoint if missing:
```bash
venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('HeartMuLa/HeartTranscriptor-oss', local_dir='./ckpt/HeartTranscriptor-oss')
"
```

---

## Mixed dtype in HeartMuLa 3B

Loading the model with `torch_dtype=torch.bfloat16` does **not** convert all layers. At least these stay float32:
- `model.codebook0_head` (Linear)
- `model.audio_head` (Parameter of shape `(7, decoder_dim, audio_vocab_size)`)

This means the training loop must cast at each dtype boundary. Symptoms if missed:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
RuntimeError: expected scalar type Float but found BFloat16
```

Pattern to use throughout `forward_train`:
```python
# Cast inputs to the head's own dtype, not a fixed dtype
head0_dtype = model.codebook0_head.weight.dtype
c0_logits = model.codebook0_head(audio_hidden.to(head0_dtype)).squeeze(0)

proj_dtype = next(model.projection.parameters()).dtype
projected = model.projection(audio_hidden.to(proj_dtype))

head_weight = model.audio_head[cb_idx - 1]   # shape (decoder_dim, vocab)
logits_i = torch.matmul(decoder_out.squeeze(0).to(head_weight.dtype), head_weight)
```

For masks and other bool/int tensors, always cast with `.to(embeds.dtype)` not `.float()`:
```python
# Wrong — produces float32 when embeds are bfloat16:
embeds * tokens_mask.unsqueeze(-1).float()

# Right:
embeds * tokens_mask.unsqueeze(-1).to(embeds.dtype)
```

---

## LoRA layer dtype mismatch

`torchtune.modules.peft.LoRALinear` is created as float32 by default. The backbone weights are bfloat16. Copy weights **after** converting the LoRA layer:

```python
lora_layer = LoRALinear(...)
lora_layer = lora_layer.to(orig.weight.dtype)   # ← must come before weight copy
lora_layer.weight.data = orig.weight.data.clone()
```

Without the `.to()` call you get a dtype error on the first forward pass through the LoRA layer.

---

## audio-flamingo environment

audio-flamingo runs in a separate conda env (`$CONDA_PREFIX` (the audio-flamingo env)) and cannot be imported from the heart-mula venv. Always call it via subprocess.

Two env vars must be set for the subprocess or it fails:

```
LD_LIBRARY_PATH  must include:
  $CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparselt/lib

CUDA_HOME  must be:
  $CONDA_PREFIX
```

These are set by `conda activate audio-flamingo` but are **not** inherited by subprocess calls. `prepare_lora_dataset.py` sets them automatically for the subprocess. If calling `audio_flamingo_tag.py` directly:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
/path/to/conda/envs/audio-flamingo/bin/python \
  /path/to/audio-flamingo/audio_flamingo_tag.py <audio_file>
```

### llava stdout pollution

The audio-flamingo `llava` library prints debug lines (model path, conversation dict, etc.) to stdout before the JSON result. Parse defensively: scan `proc.stdout.splitlines()` in reverse for the first line starting with `{`.

### m4a not supported

`llava.Sound()` cannot load m4a files. Always pass a wav file. In `prepare_lora_dataset.py` we pass the HDemucs vocals wav (not the original m4a) to the tagger.

### Hallucinated tags on bad audio

If the audio file can't be decoded (or is badly separated), audio-flamingo produces plausible-sounding but wrong tags (e.g. "screamo", "electronic" for a gospel choir vocal stem). Use `prepend_tags.py` to force-inject correct genre tags at the front of the tag list after generation.

---

## Dataset pipeline file layout

```
/path/to/audio-flamingo/
  prepare_lora_dataset.py   # orchestrator: demucs → transcription → tags → dataset.json
  audio_flamingo_tag.py     # helper: runs inside audio-flamingo conda env
  prepend_tags.py           # post-process: prepend missing tags to all manifest entries

./
  data/raw/           # 01.m4a – 11.m4a  (gospel choir source clips)
  data/processed/     # *_vocals.wav from HDemucs separation
  data/dataset.json   # manifest: audio path, lyrics, tags per clip
  data/tokens/        # 0000_01.pt – 0010_11.pt  (pre-tokenized audio + text IDs)
  data/train_log.txt  # stdout+stderr from most recent training run
  lora.pt             # current LoRA adapter (rank 8, alpha 16)
```

---

## Training observations (gospel choir, 11 clips)

| Epoch | Loss range |
|-------|-----------|
| 1     | ~8.3 (random) |
| ~10   | ~5.5–6.0 |
| ~50   | ~4.5–4.8 |

Loss plateaued ~4.5 after ~30 epochs with 11 clips and grad_accum=4. Adding more clips or increasing rank would be the next lever. The model is memorizing the 11 examples well enough to condition on gospel choir style — loss below 5 means it is predicting audio tokens significantly better than chance (8197-token vocab → random baseline ~9.01).

Healthy check: after epoch 1, LoRA A matrices must have non-zero gradients. If loss stays flat at ~8 past step 20, the `.pt` files probably didn't load (check `--dataset_dir` path).

---

## LoRA inference: import path and device placement

Two issues to be aware of:

**Import path:** `from examples.train_lora import apply_lora` fails if `examples/` is not on the Python module path. Fixed in `run_music_generation.py` by loading the module directly via `importlib.util.spec_from_file_location` using `__file__` as the anchor.

**Device placement:** After `model.load_state_dict(lora_state, strict=False)`, the newly-created LoRA A/B matrices (loaded from a `map_location="cpu"` state dict) stay on CPU even though the rest of the model is on CUDA. Must call `model.to(mula_device)` after the state dict load, or you get:
```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu
```

---

## HDemucs notes

- Pipeline: `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`
- Expected input: **44100 Hz stereo** — resample before calling the model
- Output keys: `drums`, `bass`, `other`, `vocals`
- Save as float32 wav at 44100 Hz; HeartTranscriptor accepts this directly
- First run downloads ~200MB of model weights to `~/.cache`
- Short clips (<5s) sometimes produce near-silent vocals — the transcriber will return an empty string in that case

---

## Web UI endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/generate` | Start generation job → `{jobId}` |
| GET | `/stream/:jobId` | SSE log stream for generation |
| POST | `/encode` | Start encode_dataset.py → `{jobId}` |
| GET | `/encode-stream/:jobId` | SSE log stream for encoding |
| POST | `/train` | Start train_lora.py → `{jobId}` |
| GET | `/train-stream/:jobId` | SSE log stream for training |
| GET | `/audio?path=...` | Serve generated audio file |
| GET | `/library` | List assets/*.mp3 sorted by mtime |

All job state is in-memory (`jobs` object). Server restart clears all job history. Starting a new generate or train job kills the previous one (SIGKILL) to free GPU memory — this is intentional, not a bug.
