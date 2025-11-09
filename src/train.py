import os
import time
import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import sentencepiece as spm

from src.models.transformer import Transformer
from src.data.dataset import get_loader
from src.utils import (
    set_seed, NoamLR, LabelSmoothingLoss,
    plot_series, plot_steps, append_csv, safe_bleu
)


def make_pad_mask(x, pad_id):
    return x.ne(pad_id)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device, bos_id, pad_id):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for src, tgt, _, _ in tqdm(loader, desc="Valid", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        bos_col = torch.full((tgt.size(0), 1), bos_id, dtype=tgt.dtype, device=device)
        tgt_in = torch.cat([bos_col, tgt[:, :-1]], dim=1)

        src_mask = Transformer.build_src_mask(src.ne(pad_id))
        tgt_mask = Transformer.build_tgt_mask(tgt_in.ne(pad_id))

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(logits, tgt)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_bleu_on_valid(model, loader, sp, device, pad_id, bos_id, eos_id,
                           max_len=100, max_samples=500):
    import itertools
    model.eval()
    hyps, refs, seen = [], [], 0

    for src, tgt, _, _ in loader:
        src = src.to(device)
        src_mask = Transformer.build_src_mask(src.ne(pad_id))
        mem = model.encoder(src, src_mask)                   # [B,L,d]

        # 批量贪心
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = Transformer.build_tgt_mask(ys.ne(pad_id))
            logits = model.decoder(ys, mem, tgt_mask, src_mask)   # [B,L,V]
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_tok], 1)
            finished |= (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break

        for b in range(B):
            out = ys[b].tolist()
            if out and out[0] == bos_id: out = out[1:]
            if eos_id in out: out = out[:out.index(eos_id)]
            hyps.append(sp.DecodeIds(out).strip())

            ref_ids = tgt[b].tolist()
            if eos_id in ref_ids: ref_ids = ref_ids[:ref_ids.index(eos_id)]
            refs.append(sp.DecodeIds(ref_ids).strip())

            seen += 1
            if max_samples > 0 and seen >= max_samples:
                break
        if max_samples > 0 and seen >= max_samples:
            break

    from src.utils import safe_bleu
    return safe_bleu(hyps, refs)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")

    torch.backends.cudnn.benchmark = True

    # SentencePiece
    spm_path = os.path.join(args.data_dir, "spm_bpe.model")
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)

    pad_id = sp.pad_id()   # 0
    bos_id = sp.bos_id()   # 2
    eos_id = sp.eos_id()   # 3
    vocab_size = sp.get_piece_size()

    # DataLoaders
    train_loader = get_loader(args.data_dir, "train", args.batch_size, args.max_len, pad_id, shuffle=True)
    valid_loader = get_loader(args.data_dir, "valid", args.batch_size, args.max_len, pad_id, shuffle=False)

    # Model
    try:
        model = Transformer(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            pos_enc=args.pos_enc,
            ln_style=args.ln_style
        ).to(device)
    except TypeError:
        model = Transformer(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)

    total_params, trainable_params = count_params(model)
    print(f"Params: total={total_params:,} | trainable={trainable_params:,}")

    # Optimizer (AdamW) + NoamLR
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, betas=(0.9, 0.98), eps=1e-9
        )
    scheduler = NoamLR(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    # Loss
    loss_fn = LabelSmoothingLoss(args.label_smoothing, vocab_size, ignore_index=pad_id)

    # Logging containers
    train_losses, valid_losses, valid_bleus = [], [], []
    epoch_times_min = []
    toks_per_sec = [] 
    lr_steps_x, lr_steps_y = [], []  
    grad_steps_x, grad_norms = [], [] 
    global_step = 0
    best_valid = float("inf")
    start_epoch = 1
    metrics_csv = os.path.join("results", "metrics.csv")

    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        train_losses = ckpt.get("train_losses", [])
        valid_losses = ckpt.get("valid_losses", [])
        valid_bleus  = ckpt.get("valid_bleus", [])
        epoch_times_min = ckpt.get("epoch_times_min", [])
        toks_per_sec = ckpt.get("toks_per_sec", [])
        lr_steps_x = ckpt.get("lr_steps_x", [])
        lr_steps_y = ckpt.get("lr_steps_y", [])
        grad_steps_x = ckpt.get("grad_steps_x", [])
        grad_norms = ckpt.get("grad_norms", [])
        global_step = ckpt.get("global_step", 0)
        best_valid = ckpt.get("best_valid", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] Start from epoch {start_epoch}, global_step {global_step}, best_valid {best_valid:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        total_tokens = 0

        for src, tgt, _, _ in tqdm(train_loader, desc="Train", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            bos_col = torch.full((tgt.size(0), 1), bos_id, dtype=tgt.dtype, device=device)
            tgt_in = torch.cat([bos_col, tgt[:, :-1]], dim=1)

            src_mask = Transformer.build_src_mask(src.ne(pad_id))
            tgt_mask = Transformer.build_tgt_mask(tgt_in.ne(pad_id))

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)
            grad_steps_x.append(global_step + 1)
            grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            lr_steps_x.append(global_step + 1)
            lr_steps_y.append(current_lr)

            total_tokens += int(src.ne(pad_id).sum().item() + tgt.ne(pad_id).sum().item() - src.size(0))
            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        tr_loss = total_loss / max(1, n_batches)

        va_loss = evaluate_loss(model, valid_loader, loss_fn, device, bos_id, pad_id)

        va_bleu = evaluate_bleu_on_valid(
            model, valid_loader, sp, device, pad_id, bos_id, eos_id,
            max_len=args.max_len, max_samples=args.bleu_max_samples  # 子集评估，加速
        )

        train_losses.append(tr_loss)
        valid_losses.append(va_loss)
        valid_bleus.append(va_bleu)
        epoch_minutes = (time.time() - t0) / 60.0
        epoch_times_min.append(epoch_minutes)
        toks_per_sec.append(total_tokens / max(1e-6, (time.time() - t0)))

        print(f"Train Loss: {tr_loss:.4f} | Valid Loss: {va_loss:.4f} | Valid BLEU: {va_bleu:.2f} | epoch {epoch_minutes:.2f} min")

        is_best = va_loss <= best_valid
        if is_best:
            best_valid = va_loss

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_valid": best_valid,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "valid_bleus": valid_bleus,
            "epoch_times_min": epoch_times_min,
            "toks_per_sec": toks_per_sec,
            "lr_steps_x": lr_steps_x,
            "lr_steps_y": lr_steps_y,
            "grad_steps_x": grad_steps_x,
            "grad_norms": grad_norms,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        save_checkpoint(state, os.path.join(args.save_dir, "last.pt"))
        if is_best:
            save_checkpoint(state, os.path.join(args.save_dir, "best.pt"))

        append_csv(
            os.path.join("results", "metrics.csv"),
            header=["epoch", "train_loss", "valid_loss", "train_ppl", "valid_ppl", "valid_bleu",
                    "epoch_minutes", "tokens_per_sec", "steps_total", "lr_last"],
            row=[epoch, tr_loss, va_loss, math.exp(tr_loss), math.exp(va_loss), va_bleu,
                 epoch_minutes, toks_per_sec[-1], global_step, scheduler.get_last_lr()[0]]
        )

        xs = list(range(1, len(train_losses) + 1))
        plot_series(xs, train_losses, "Train Loss", os.path.join("results", "loss_train.png"))
        plot_series(xs, valid_losses, "Valid Loss", os.path.join("results", "loss_valid.png"))
        plot_series(xs, [math.exp(x) for x in train_losses], "Train Perplexity", os.path.join("results", "ppl_train.png"))
        plot_series(xs, [math.exp(x) for x in valid_losses], "Valid Perplexity", os.path.join("results", "ppl_valid.png"))
        plot_series(xs, valid_bleus, "Valid BLEU", os.path.join("results", "bleu_valid.png"))
        plot_series(xs, toks_per_sec, "Tokens/sec (epoch)", os.path.join("results", "throughput_toks_per_s.png"))
        plot_series(xs, epoch_times_min, "Minutes", os.path.join("results", "epoch_time_min.png"), title="Epoch Time")
        plot_steps(lr_steps_x, lr_steps_y, "Learning Rate", os.path.join("results", "lr_steps.png"))
        plot_steps(grad_steps_x, grad_norms, "Grad Norm (global)", os.path.join("results", "grad_norm_steps.png"))

    print("Training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bleu_max_samples", type=int, default=2000,
                        help="验证集计算BLEU时最多使用多少句(0=用完整验证集)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--resume", type=str, default="", help="加载断点继续训练(best/last 任一)")
    parser.add_argument("--pos_enc", choices=["sin", "learned", "none"], default="sin",
                    help="位置编码:sin(默认) / learned(可学习) / none(不使用)")
    parser.add_argument("--ln_style", choices=["pre", "post"], default="pre",
                        help="LayerNorm 位置:pre(默认) / post")
    parser.add_argument("--optim", choices=["adamw", "adam"], default="adamw",
                        help="优化器选择:adamw(默认) / adam")
    args = parser.parse_args()
    main(args)
