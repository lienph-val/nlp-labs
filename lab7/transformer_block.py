import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, verbose=False):
        B, T, d_model = x.shape

        Q = self.split_heads(self.W_q(x))  # (B, h, T, d_k)
        K = self.split_heads(self.W_k(x))  # (B, h, T, d_k)
        V = self.split_heads(self.W_v(x))  # (B, h, T, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, T, T)
        weights = self.softmax(scores)                          # (B, h, T, T)
        out = weights @ V                                       # (B, h, T, d_k)

        # Ghép các head lại: (B, h, T, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        out = self.W_o(out)                                     # (B, T, d_model)

        if verbose:
            print(f"  Q               : {tuple(Q.shape)}")
            print(f"  K               : {tuple(K.shape)}")
            print(f"  V               : {tuple(V.shape)}")
            print(f"  Attention scores: {tuple(scores.shape)}")
            print(f"  Attention output: {tuple(out.shape)} (sau concat heads + W_o)")
        return out

# Feed-Forward Network: Linear -> ReLU -> Linear
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, verbose=False):
        if verbose:
            print(f"X (input)            : {tuple(x.shape)}")

        # 1) Multi-Head Self-Attention -> Residual + LayerNorm
        attn_out = self.attention(x, verbose)
        x = self.norm1(x + self.dropout(attn_out))
        if verbose:
            print(f"Sau Add & Norm 1     : {tuple(x.shape)}")

        # 2) Feed-Forward -> Residual + LayerNorm
        ffn_out = self.ffn(x)
        if verbose:
            print(f"FFN output           : {tuple(ffn_out.shape)}")
        x = self.norm2(x + self.dropout(ffn_out))
        if verbose:
            print(f"Output (Add & Norm 2): {tuple(x.shape)}")
        return x

#18: Kiểm tra shape
if __name__ == "__main__":
    B, T, d_model, n_heads, d_ff = 8, 100, 128, 4, 512
    d_k = d_model // n_heads
    print(f"B={B}, T={T}, d_model={d_model}, n_heads={n_heads}, d_k={d_k}, d_ff={d_ff}\n")

    block = TransformerBlock(d_model, n_heads, d_ff)
    block.eval()  

    x = torch.randn(B, T, d_model)
    out = block(x, verbose=True)

    # Kiểm tra bằng assert
    assert out.shape == (B, T, d_model)
    q = block.attention.split_heads(block.attention.W_q(x))
    assert q.shape == (B, n_heads, T, d_k)
    print("\nTất cả shape đều đúng như dự kiến.")
