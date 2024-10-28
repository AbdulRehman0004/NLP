import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # the output dimensionality
        self.num_heads = num_heads # the number of attention heads
        self.head_dim = d_model // num_heads # dimensionality of each head output

        # Making sure that each head will process a portion of the output
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Initialize weights for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        # self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output, attention_probs