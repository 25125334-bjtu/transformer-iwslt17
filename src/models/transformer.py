import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent  # True=可见, False=屏蔽


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:L, :].unsqueeze(0)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size()
        B, Lk, _ = k.size()
        B, Lv, _ = v.size()

        Q = self.w_q(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lq, Dh]
        K = self.w_k(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lk, Dh]
        V = self.w_v(v).view(B, Lv, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lv, Dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Lq,Lk]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [B,H,Lq,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # [B,Lq,D]
        return self.w_o(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # self-attn
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop1(attn_out))
        # ffn
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask, src_mask):
        # masked self-attn
        out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.drop1(out))
        # enc-dec attn
        out = self.enc_attn(x, mem, mem, mask=src_mask)
        x = self.norm2(x + self.drop2(out))
        # ffn
        out = self.ffn(x)
        x = self.norm3(x + self.drop3(out))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, src_ids, src_mask):
        x = self.embed(src_ids)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, mem, tgt_mask, src_mask):
        x = self.embed(tgt_ids)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mem, tgt_mask, src_mask)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, dropout)

    def forward(self, src_ids, tgt_in_ids, src_mask, tgt_mask):
        mem = self.encoder(src_ids, src_mask)
        logits = self.decoder(tgt_in_ids, mem, tgt_mask, src_mask)
        return logits

    @staticmethod
    def build_src_mask(src_pad_mask: torch.Tensor):
        # src_pad_mask: [B, L] True=非pad
        # attn mask shape -> [B, 1, 1, L] broadcast to [B,H,Lq,Lk]
        return src_pad_mask.unsqueeze(1).unsqueeze(1)

    @staticmethod
    def build_tgt_mask(tgt_pad_mask: torch.Tensor):
        # pad mask: [B,1,1,L]
        pad = tgt_pad_mask.unsqueeze(1).unsqueeze(1)
        # subsequent: [1,L,L]
        L = tgt_pad_mask.size(1)
        sub = subsequent_mask(L).to(tgt_pad_mask.device).unsqueeze(0)  # [1,L,L]
        return pad & sub