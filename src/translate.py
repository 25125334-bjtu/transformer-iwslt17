import os
import argparse
import torch
import sentencepiece as spm
from src.models.transformer import Transformer


@torch.no_grad()
def greedy_decode(model, enc, src_mask, bos_id, eos_id, max_len):
    B = enc.size(0)
    device = enc.device
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = ys.ne(0)  # 非pad
        tgt_mask = Transformer.build_tgt_mask(tgt_pad_mask)
        logits = model.decoder(ys, enc, tgt_mask, src_mask)  # [B,L,V]
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
        ys = torch.cat([ys, next_token], dim=1)
        if torch.all(next_token.squeeze(-1) == eos_id):
            break
    return ys


@torch.no_grad()
def beam_search(model, enc, src_mask, bos_id, eos_id, max_len,
                beam_size=5, alpha=0.6, min_len=5, no_repeat_ngram=3):
    """
    Beam search with length penalty (GNMT), min length, and no-repeat ngram.
    Works for batch size = 1.
    """
    device = enc.device

    def length_penalty(length, alpha):
        # GNMT length penalty
        return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

    def violates_no_repeat(ids, n):
        if n <= 0 or len(ids) < n*2:  # too short to repeat
            return False
        ngrams = set()
        for i in range(len(ids) - n + 1):
            ng = tuple(ids[i:i+n])
            if ng in ngrams:
                return True
            ngrams.add(ng)
        return False

    # beam: list of (ids_tensor[[...]], sum_logprob, finished_bool)
    ys0 = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    beams = [(ys0, 0.0, False)]

    for step in range(1, max_len):
        all_cands = []
        for seq, score, finished in beams:
            last = seq[0, -1].item()
            if finished:
                # keep finished as candidates (no expansion)
                all_cands.append((seq, score, True))
                continue
            if step < min_len and last == eos_id:
                # block premature EOS
                continue

            tgt_mask = Transformer.build_tgt_mask(seq.ne(0))  # pad_id=0
            logits = model.decoder(seq, enc, tgt_mask, src_mask)  # [1,L,V]
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)    # [1,V]

            topk = torch.topk(logp, beam_size, dim=-1)
            for i in range(beam_size):
                tok = topk.indices[0, i].view(1, 1)
                new_seq = torch.cat([seq, tok], dim=1)
                new_score = score + topk.values[0, i].item()

                ids_list = new_seq[0].tolist()
                # no-repeat-ngram
                if no_repeat_ngram > 0 and violates_no_repeat(ids_list[1:], no_repeat_ngram):
                    continue

                done = tok.item() == eos_id and step >= min_len
                all_cands.append((new_seq, new_score, done))

        def normed_score(tpl):
            seq, score, finished = tpl
            ids = seq[0].tolist()[1:]
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            lp = length_penalty(max(1, len(ids)), alpha)
            return score / lp

        ordered = sorted(all_cands, key=normed_score, reverse=True)
        beams = ordered[:beam_size]

        if all(fin for _, _, fin in beams):
            break

    # 返回归一化分数最好的那条
    best = max(beams, key=lambda x: (x[1] / length_penalty(
        max(1, len(x[0][0].tolist()[1: x[0][0].tolist().index(eos_id)]
                   if eos_id in x[0][0].tolist() else x[0][0].tolist()) ), alpha)))
    return best[0]



def ids_to_text(sp, ids):
    ids = ids.tolist()
    if ids and ids[0] == sp.bos_id():
        ids = ids[1:]
    if sp.eos_id() in ids:
        idx = ids.index(sp.eos_id())
        ids = ids[:idx]
    return sp.DecodeIds(ids)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.data_dir, "spm_bpe.model"))
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # 模型
    model = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.0
    ).to(device)
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt.get("model", ckpt) 
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with open(args.input, "r", encoding="utf-8") as fi, open(args.output, "w", encoding="utf-8") as fo:
        for line in fi:
            src_text = line.strip()
            src_ids = sp.EncodeAsIds(src_text) + [eos_id]
            src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,L]
            src_mask = Transformer.build_src_mask(src.ne(pad_id))
            mem = model.encoder(src, src_mask)

            if args.beam_size <= 1:
                out = greedy_decode(model, mem, src_mask, bos_id, eos_id, args.max_len)
            else:
                out = beam_search(model, mem, src_mask, bos_id, eos_id, args.max_len,
                                beam_size=args.beam_size, alpha=args.alpha,
                                min_len=args.min_len, no_repeat_ngram=args.no_repeat_ngram)

            hyp = ids_to_text(sp, out[0])
            fo.write(hyp + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.6, help="length penalty for beam search")
    parser.add_argument("--min_len", type=int, default=5, help="minimum generation length before EOS")
    parser.add_argument("--no_repeat_ngram", type=int, default=3, help="block repeated n-grams")

    args = parser.parse_args()
    main(args)
