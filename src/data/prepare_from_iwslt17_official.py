import os
import re
import io
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple

def _read_train_tags(en_path: str, de_path: str) -> Tuple[List[str], List[str]]:
    """train.tags.en-de.{en,de}: keep only real sentences (lines that do NOT start with '<')."""
    def _read_one(p):
        out = []
        with io.open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("<"):  # drop tags like <url> <talkid> <title> etc.
                    continue
                out.append(line)
        return out
    en = _read_one(en_path)
    de = _read_one(de_path)
    assert len(en) == len(de), f"train tags length mismatch: {len(en)} vs {len(de)}"
    return en, de

def _read_xml_pair(en_xml: str, de_xml: str) -> Tuple[List[str], List[str]]:
    """Read aligned segments from *.en.xml and *.de.xml (same doc layout)."""
    def _extract(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # IWSLT TED XML typically: <mteval><doc ...><seg id="1"> ... </seg>...</doc>...</mteval>
        segs = []
        for seg in root.iterfind(".//seg"):
            text = (seg.text or "").strip()
            if text:
                segs.append(text)
        return segs
    en = _extract(en_xml)
    de = _extract(de_xml)
    assert len(en) == len(de), f"xml length mismatch for {en_xml} vs {de_xml}: {len(en)} vs {len(de)}"
    return en, de

def _write_parallel(out_prefix: str, src: List[str], tgt: List[str]):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with io.open(out_prefix + ".en", "w", encoding="utf-8") as f_en, \
         io.open(out_prefix + ".de", "w", encoding="utf-8") as f_de:
        for e, d in zip(src, tgt):
            f_en.write(e.strip() + "\n")
            f_de.write(d.strip() + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/iwslt17/offical",
                    help="folder that contains train.tags.* and *.xml")
    ap.add_argument("--out_raw", type=str, default="data/iwslt17/raw",
                    help="output folder for raw/{train,valid,test}.{en,de}")
    # 默认划分：train = train.tags，valid = tst2013，test = tst2014
    ap.add_argument("--valid_year", type=str, default="tst2013", choices=[
        "dev2010","tst2010","tst2011","tst2012","tst2013","tst2014","tst2015"
    ])
    ap.add_argument("--test_year", type=str, default="tst2014", choices=[
        "dev2010","tst2010","tst2011","tst2012","tst2013","tst2014","tst2015"
    ])
    args = ap.parse_args()

    os.makedirs(args.out_raw, exist_ok=True)

    # 1) train: train.tags.en-de.{en,de}
    train_en = os.path.join(args.root, "train.tags.en-de.en")
    train_de = os.path.join(args.root, "train.tags.en-de.de")
    assert os.path.isfile(train_en) and os.path.isfile(train_de), "train.tags files not found."
    tr_en, tr_de = _read_train_tags(train_en, train_de)
    print(f"[train] {len(tr_en)} pairs")

    # 2) valid: choose one year split (default: tst2013)
    valid_en_xml = sorted(glob.glob(os.path.join(args.root, f"IWSLT17.TED.{args.valid_year}.en-de.en.xml")))
    valid_de_xml = sorted(glob.glob(os.path.join(args.root, f"IWSLT17.TED.{args.valid_year}.en-de.de.xml")))
    assert len(valid_en_xml)==1 and len(valid_de_xml)==1, f"valid xml files for {args.valid_year} not found."
    va_en, va_de = _read_xml_pair(valid_en_xml[0], valid_de_xml[0])
    print(f"[valid:{args.valid_year}] {len(va_en)} pairs")

    # 3) test: choose one year split (default: tst2014)
    test_en_xml = sorted(glob.glob(os.path.join(args.root, f"IWSLT17.TED.{args.test_year}.en-de.en.xml")))
    test_de_xml = sorted(glob.glob(os.path.join(args.root, f"IWSLT17.TED.{args.test_year}.en-de.de.xml")))
    assert len(test_en_xml)==1 and len(test_de_xml)==1, f"test xml files for {args.test_year} not found."
    te_en, te_de = _read_xml_pair(test_en_xml[0], test_de_xml[0])
    print(f"[test:{args.test_year}] {len(te_en)} pairs")

    # 写出
    _write_parallel(os.path.join(args.out_raw, "train"), tr_en, tr_de)
    _write_parallel(os.path.join(args.out_raw, "valid"), va_en, va_de)
    _write_parallel(os.path.join(args.out_raw, "test"),  te_en, te_de)
    print(f"✓ Wrote raw files to {args.out_raw}")

if __name__ == "__main__":
    main()
