import torch
from torch import nn
from torch import Tensor
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class SelfAttention(nn.Module):
    def __init__(self, inp_size, out_size, kernel_size=3):
        super(SelfAttention, self).__init__()

        self.out_size = out_size
        self.qw = nn.Linear(inp_size, out_size)
        self.kw = nn.Linear(inp_size, out_size)
        self.vw = nn.Linear(inp_size, out_size)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        q = self.qw(q)
        k = torch.transpose(self.kw(k), 1, 2)
        v = self.vw(v)
        out = torch.softmax(torch.matmul(q, k) / math.sqrt(self.out_size), dim=-1)
        # print("q", q.shape)
        # print("k", k.shape)
        # print("v", v.shape)
        # print("out", out.shape)
        out = torch.matmul(out, v)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, inp_size, num_heads, kernel_size=3):
        super(MultiheadAttention, self).__init__()
        assert inp_size % num_heads == 0
        # self.inp_size =  inp_size
        out_head_size = inp_size // num_heads


        self.query_emb = nn.Sequential(
            nn.Linear(inp_size, inp_size),
            nn.ReLU())
        self.key_emb = nn.Sequential(
            nn.Linear(inp_size, inp_size),
            nn.ReLU())
        self.value_emb = nn.Sequential(
            nn.Linear(inp_size, inp_size),
            nn.ReLU())

        self.heads = [SelfAttention(inp_size, out_head_size, kernel_size) for _ in range(num_heads)]
        self.out_linear = nn.Linear(inp_size, inp_size)

    def forward(self, input: Tensor):
        # input.shape : bs, seq_len, emb_dim
        q = self.query_emb(input)
        k = self.key_emb(input)
        v = self.value_emb(input)
        # print("q", q.shape)
        # print("k", k.shape)
        # print("v", v.shape)

        out = []
        for i in range(len(self.heads)):
            out.append(self.heads[i](q, k, v))
        out = torch.cat(out, dim=-1)
        out = self.out_linear(out)
        return out


class FFTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, kernel_size):
        super(FFTBlock, self).__init__()
        self.multihead_attn = MultiheadAttention(hidden_size, num_heads, kernel_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'),
                                  nn.ReLU(),
                                  nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'))
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, input: Tensor):
        # input.shape : bs, seq_len, emb_dim
        x = self.multihead_attn(input)
        x = self.layer_norm1(x + input)
        # x.shape : bs, seq_len, emb_dim
        out = torch.transpose(self.conv(torch.transpose(x, 1, 2)), 1, 2)
        out = self.layer_norm1(out + x)
        return out


class Aligner(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(Aligner, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'),
                                   nn.ReLU())
        self.layer_norm1 = nn.LayerNorm(hidden_size, hidden_size)
        self.conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'),
                                   nn.ReLU())
        self.layer_norm2 = nn.LayerNorm(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input: Tensor):
        # input.shape : bs, seq_len, emb_dim
        out = torch.transpose(self.conv1(torch.transpose(input, 1, 2)), 1, 2)
        out = self.layer_norm1(out)
        out = torch.transpose(self.conv1(torch.transpose(out, 1, 2)), 1, 2)
        self.layer_norm2(out)
        out = self.linear(out)
        return out
