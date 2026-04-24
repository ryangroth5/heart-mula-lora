"""
train_lora.py — LoRA fine-tuning for HeartMuLa 3B on pre-tokenized audio + lyrics/tags pairs.

Single GPU:
  venv/bin/python examples/train_lora.py \
    --model_path ./ckpt \
    --dataset_dir ./data/tokens \
    --output ./lora.pt \
    --epochs 3 \
    --lora_rank 8

Multi-GPU:
  accelerate launch --num_processes=2 examples/train_lora.py \
    --model_path ./ckpt --dataset_dir ./data/tokens ...
"""

import argparse
import os
import sys
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted(glob(os.path.join(dataset_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {dataset_dir}. Run encode_dataset.py first.")
        print(f"Dataset: {len(self.files)} samples from {dataset_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        return data["audio_tokens"], data["tags_ids"], data["lyrics_ids"]


def collate_fn(batch):
    """Return list of items — handle variable-length sequences individually."""
    return batch


# ---------------------------------------------------------------------------
# Token embedding helpers (mirrors pipeline preprocess)
# ---------------------------------------------------------------------------

def build_prompt_tokens(tags_ids, lyrics_ids, config):
    """Build the text prompt token sequence: [tags_tokens] + [empty] + [lyrics_tokens]."""
    empty_id = getattr(config, "empty_id", 0)
    combined = tags_ids + [empty_id] + lyrics_ids
    # Shape: (S,) — will be expanded to (S, 9) inside forward_train
    return torch.tensor(combined, dtype=torch.long)


# ---------------------------------------------------------------------------
# Teacher-forcing forward pass
# ---------------------------------------------------------------------------

def forward_train(model, prompt_tokens, audio_tokens, device):
    """
    Full sequence teacher-forcing forward pass.

    Args:
        model: HeartMuLa instance (with LoRA applied)
        prompt_tokens: (S,) text token IDs
        audio_tokens: (T, 8) ground-truth audio token IDs
        device: torch device

    Returns:
        c0_logits: (T, audio_vocab_size)
        ci_logits: list of 7 tensors each (T, audio_vocab_size)
    """
    S = prompt_tokens.shape[0]
    T = audio_tokens.shape[0]
    total_len = S + T

    # Build combined token tensor of shape (1, S+T, 9)
    # Text positions: columns 0-7 = empty_id (0), column 8 = text token ID
    # Audio positions: columns 0-7 = audio codebook tokens, column 8 = empty_id (0)
    empty_id = 0
    tokens = torch.full((1, total_len, 9), empty_id, dtype=torch.long, device=device)

    # Text prompt — last column holds text token IDs
    tokens[0, :S, 8] = prompt_tokens.to(device)

    # Audio frames — first 8 columns hold codebook tokens
    tokens[0, S:, :8] = audio_tokens.to(device)

    # Mask: 1 where the slot is valid
    tokens_mask = torch.ones(1, total_len, 9, dtype=torch.bool, device=device)

    # Embed tokens
    embeds = model._embed_tokens(tokens, uncond_mask=None)  # (1, S+T, 9, dim)
    # Sum across the parallel_number=9 dimension after masking
    embeds = (embeds * tokens_mask.unsqueeze(-1).to(embeds.dtype)).sum(dim=2)  # (1, S+T, dim)

    # Positional indices
    positions = torch.arange(total_len, device=device)

    # Backbone forward (no KV cache — full sequence at once)
    backbone_out = model.backbone(embeds, input_pos=positions)  # (1, S+T, dim)

    # Slice audio frame positions
    audio_hidden = backbone_out[:, S:, :]  # (1, T, dim) — keep model dtype (bfloat16)

    # Codebook 0 logits — cast to head's dtype (may differ from backbone dtype)
    head0_dtype = model.codebook0_head.weight.dtype
    c0_logits = model.codebook0_head(audio_hidden.to(head0_dtype)).squeeze(0)  # (T, vocab)

    # Codebooks 1-7 via decoder (one frame at a time with gradient checkpointing)
    ci_logits = []
    proj_dtype = next(model.projection.parameters()).dtype
    projected = model.projection(audio_hidden.to(proj_dtype))  # (1, T, decoder_dim)

    for cb_idx in range(1, 8):
        # Embed codebook cb_idx-1 ground-truth tokens as decoder input
        # audio_head is a weight matrix: (7, decoder_dim, audio_vocab_size)
        # We need embeddings: use audio_embeddings with offset
        prev_cb_tokens = audio_tokens[:, cb_idx - 1].to(device)  # (T,)
        offset = (cb_idx - 1) * model.config.audio_vocab_size
        prev_embeds = model.audio_embeddings(prev_cb_tokens.unsqueeze(0) + offset)  # (1, T, dim)

        # Concatenate with projected backbone output as decoder input
        decoder_in = projected + prev_embeds  # (1, T, dim) simple fusion

        def make_checkpoint_fn(d_in):
            def fn(x):
                return model.decoder(x, input_pos=positions[S:S + T])
            return fn

        decoder_out = torch.utils.checkpoint.checkpoint(
            make_checkpoint_fn(decoder_in), decoder_in, use_reentrant=False
        )  # (1, T, decoder_dim)

        # audio_head: Parameter of shape (7, decoder_dim, audio_vocab_size)
        head_weight = model.audio_head[cb_idx - 1]  # (decoder_dim, audio_vocab_size)
        logits_i = torch.matmul(decoder_out.squeeze(0).to(head_weight.dtype), head_weight)  # (T, vocab)
        ci_logits.append(logits_i)

    return c0_logits, ci_logits


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(model, lora_rank, lora_alpha, lora_dropout):
    from torchtune.modules.peft import LoRALinear, get_adapter_params, set_trainable_params

    replaced = 0
    for layer in model.backbone.layers:
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'output_proj']:
            orig = getattr(layer.attn, proj_name, None)
            if orig is None:
                continue
            lora_layer = LoRALinear(
                in_dim=orig.in_features,
                out_dim=orig.out_features,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            # Copy original weights and match dtype
            lora_layer = lora_layer.to(orig.weight.dtype)
            lora_layer.weight.data = orig.weight.data.clone()
            if orig.bias is not None:
                lora_layer.bias = orig.bias
            setattr(layer.attn, proj_name, lora_layer)
            replaced += 1

    print(f"Replaced {replaced} projections with LoRALinear (rank={lora_rank}, alpha={lora_alpha})")

    set_trainable_params(model, get_adapter_params(model))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load model
    print("Loading HeartMuLa model...")
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    model_path = os.path.join(args.model_path, f"HeartMuLa-oss-{args.version}")
    model = HeartMuLa.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.train()

    # Apply LoRA
    apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Enable gradient checkpointing on backbone layers
    for layer in model.backbone.layers:
        if hasattr(layer, 'gradient_checkpointing_enable'):
            layer.gradient_checkpointing_enable()

    # Dataset & dataloader
    dataset = TokenDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Optimizer (AdamW on LoRA params only)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Accelerate wrapping
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    audio_vocab_size = model.module.config.audio_vocab_size if hasattr(model, 'module') else model.config.audio_vocab_size

    global_step = 0
    optimizer.zero_grad()

    # Checkpoint directory sits next to the output file
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "lora_checkpoints")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(tag):
        from torchtune.modules.peft import get_adapter_params
        inner = model.module if hasattr(model, 'module') else model
        path = os.path.join(ckpt_dir, f"lora_{tag}.pt")
        torch.save(get_adapter_params(inner), path)
        print(f"[ckpt] saved {path}", flush=True)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(dataloader):
            item = batch[0]
            audio_tokens, tags_ids, lyrics_ids = item
            audio_tokens = audio_tokens.long()

            # Build text prompt
            inner_model = model.module if hasattr(model, 'module') else model
            prompt_tokens = build_prompt_tokens(tags_ids, lyrics_ids, inner_model.config)

            # Skip sequences that exceed the decoder's max_seq_len (2048)
            decoder_max = getattr(inner_model.decoder, 'max_seq_len', 2048)
            if prompt_tokens.shape[0] + audio_tokens.shape[0] > decoder_max:
                print(f"  SKIP: S+T={prompt_tokens.shape[0]+audio_tokens.shape[0]} > decoder max {decoder_max}", flush=True)
                continue

            # Forward
            c0_logits, ci_logits = forward_train(inner_model, prompt_tokens, audio_tokens, device)

            # Targets: audio_tokens columns 0-7
            targets = [audio_tokens[:, cb].to(device) for cb in range(8)]

            # Loss: cross-entropy over all 8 codebooks
            all_logits = [c0_logits] + ci_logits
            loss = sum(
                F.cross_entropy(all_logits[i].float(), targets[i])
                for i in range(8)
            ) / 8

            # Gradient accumulation
            loss = loss / args.grad_accum
            accelerator.backward(loss)

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print(f"step={global_step} epoch={epoch} loss={loss.item() * args.grad_accum:.4f}", flush=True)

        # End-of-epoch step if remainder
        if (len(dataloader) % args.grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(f"step={global_step} epoch={epoch} (end-of-epoch flush)", flush=True)

        # Periodic checkpoint
        if accelerator.is_main_process and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            save_checkpoint(f"epoch{epoch:04d}_step{global_step}")

    # Save final LoRA weights
    if accelerator.is_main_process:
        from torchtune.modules.peft import get_adapter_params
        inner_model = model.module if hasattr(model, 'module') else model
        lora_state = get_adapter_params(inner_model)
        torch.save(lora_state, args.output)
        print(f"LoRA weights saved to {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for HeartMuLa")
    parser.add_argument("--model_path", default="./ckpt")
    parser.add_argument("--version", default="3B", choices=["3B", "7B", "3B-happy-new-year"])
    parser.add_argument("--dataset_dir", default="./data/tokens")
    parser.add_argument("--output", default="./lora.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=100,
                        help="Save a checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
