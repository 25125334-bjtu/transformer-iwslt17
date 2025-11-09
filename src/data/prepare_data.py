import os
import argparse
import io
from datasets import load_dataset
import sentencepiece as spm


def write_parallel_txt(split, ds, src_lang, tgt_lang, out_dir):
    src_path = os.path.join(out_dir, f"{split}.{src_lang}")
    tgt_path = os.path.join(out_dir, f"{split}.{tgt_lang}")
    with io.open(src_path, "w", encoding="utf-8") as fs, io.open(tgt_path, "w", encoding="utf-8") as ft:
        for ex in ds:
            src = ex["translation"][src_lang].strip()
            tgt = ex["translation"][tgt_lang].strip()
            if src and tgt:
                fs.write(src + "\n")
                ft.write(tgt + "\n")
    return src_path, tgt_path


def train_joint_bpe(src_files, tgt_files, model_prefix, vocab_size):
    tmp_all = model_prefix + ".train.all"
    with io.open(tmp_all, "w", encoding="utf-8") as f:
        for p in src_files + tgt_files:
            with io.open(p, "r", encoding="utf-8") as fr:
                for line in fr:
                    f.write(line.strip() + "\n")

    spm.SentencePieceTrainer.Train(
        input=tmp_all,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0, pad_piece="<pad>",
        unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<bos>",
        eos_id=3, eos_piece="<eos>"
    )
    os.remove(tmp_all)


def encode_file(sp, infile, outfile, add_bos=False, add_eos=True):
    with io.open(infile, "r", encoding="utf-8") as fi, io.open(outfile, "w", encoding="utf-8") as fo:
        for line in fi:
            text = line.strip()
            ids = sp.EncodeAsIds(text)
            if add_bos:
                ids = [sp.bos_id()] + ids
            if add_eos:
                ids = ids + [sp.eos_id()]
            fo.write(" ".join(map(str, ids)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/iwslt17")
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--src_lang", type=str, default="en")
    ap.add_argument("--tgt_lang", type=str, default="de")
    ap.add_argument("--from_local", action="store_true",
                    help="使用本地 raw/train|valid|test.{en,de}，跳过下载")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if not args.from_local:
        print("Downloading IWSLT2017 (en-de) via datasets ...")
        data = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de")
        print("Writing raw txt ...")
        train_src, train_tgt = write_parallel_txt("train", data["train"], args.src_lang, args.tgt_lang, raw_dir)
        valid_src, valid_tgt = write_parallel_txt("valid", data["validation"], args.src_lang, args.tgt_lang, raw_dir)
        test_src,  test_tgt  = write_parallel_txt("test",  data["test"], args.src_lang, args.tgt_lang, raw_dir)
    else:
        # 本地模式：要求已存在 raw/train|valid|test.{en,de}
        def _ck(p):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing file: {p}")
            return p
        print("Using local raw files (skip download).")
        train_src = _ck(os.path.join(raw_dir, f"train.{args.src_lang}"))
        train_tgt = _ck(os.path.join(raw_dir, f"train.{args.tgt_lang}"))
        valid_src = _ck(os.path.join(raw_dir, f"valid.{args.src_lang}"))
        valid_tgt = _ck(os.path.join(raw_dir, f"valid.{args.tgt_lang}"))
        test_src  = _ck(os.path.join(raw_dir, f"test.{args.src_lang}"))
        test_tgt  = _ck(os.path.join(raw_dir, f"test.{args.tgt_lang}"))

    print("Training joint BPE ...")
    spm_prefix = os.path.join(args.data_dir, "spm_bpe")
    train_joint_bpe([train_src], [train_tgt], spm_prefix, args.vocab_size)

    sp = spm.SentencePieceProcessor()
    sp.load(spm_prefix + ".model")

    print("Encoding to id files ...")
    bin_dir = os.path.join(args.data_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    pairs = [
        (train_src, os.path.join(bin_dir, "train.src")),
        (train_tgt, os.path.join(bin_dir, "train.tgt")),
        (valid_src, os.path.join(bin_dir, "valid.src")),
        (valid_tgt, os.path.join(bin_dir, "valid.tgt")),
        (test_src,  os.path.join(bin_dir, "test.src")),
        (test_tgt,  os.path.join(bin_dir, "test.tgt")),
    ]
    for inp, outp in pairs:
        encode_file(sp, inp, outp, add_bos=False, add_eos=True)

    with open(os.path.join(args.data_dir, "vocab_size.txt"), "w") as f:
        f.write(str(sp.get_piece_size()))
    print("Done. Data prepared at:", args.data_dir)


if __name__ == "__main__":
    main()
