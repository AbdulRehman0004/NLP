import torch
import torch.nn as nn
import math

from src.multiattention import MultiHeadAttention
from src.transformerblock import TransformerBlock



class ARGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        x = self.token_embedding(x) + self.position_embedding(torch.arange(seq_length, device=x.device))
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = self.fc_out(x)
        return x
        
    
