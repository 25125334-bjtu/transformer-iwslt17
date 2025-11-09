import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#windows
class PadCollate:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id
    def __call__(self, batch):
        return pad_batch(batch, pad_id=self.pad_id)

class ParallelIdsDataset(Dataset):
    def __init__(self, src_path, tgt_path, max_len=100):
        self.src = self._load_ids(src_path, max_len)
        self.tgt = self._load_ids(tgt_path, max_len)
        assert len(self.src) == len(self.tgt)

    @staticmethod
    def _load_ids(path, max_len):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ids = list(map(int, line.strip().split()))
                if len(ids) == 0:
                    continue
                if len(ids) > max_len:
                    ids = ids[:max_len-1] + [ids[-1]]
                data.append(ids)
        return data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def pad_batch(examples, pad_id=0):
    # examples: [(src_ids, tgt_ids), ...]
    src_seqs = [torch.tensor(s, dtype=torch.long) for s, _ in examples]
    tgt_seqs = [torch.tensor(t, dtype=torch.long) for _, t in examples]
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]

    src_pad = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_id)
    return src_pad, tgt_pad, torch.tensor(src_lens), torch.tensor(tgt_lens)


def get_loader(data_dir, split, batch_size, max_len, pad_id, shuffle):
    src_path = os.path.join(data_dir, "bin", f"{split}.src")
    tgt_path = os.path.join(data_dir, "bin", f"{split}.tgt")
    ds = ParallelIdsDataset(src_path, tgt_path, max_len=max_len)

    return DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0,                # Windows 下设为 0，避免多进程序列化问题
                    collate_fn=PadCollate(pad_id)
    )