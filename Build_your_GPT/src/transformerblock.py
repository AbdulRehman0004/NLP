import torch
import torch.nn as nn
import math
from src.multiattention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        #  Complete TransformerBlock.forward()
        att,_ = self.attention(x,x,x)
        x = self.norm1(x + self.dropout(att))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x
        
