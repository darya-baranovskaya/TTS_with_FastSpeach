import torch
from torch import nn
from models.model_layers import *
from models.grapheme_aligner import GraphemeAligner
from torch.nn.utils.rnn import pad_sequence

class FastSpeechEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, hidden_size2, num_heads, kernel_size, n_fft_blocks, device):
        super(FastSpeechEncoder, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(emb_size=hidden_size, dropout=0)
        self.fft_blocks = torch.nn.ModuleList([FFTBlock(hidden_size, hidden_size2, num_heads, kernel_size, device).to(device) for _ in range(n_fft_blocks)])

    def forward(self, input: Tensor):
        out = self.pos_encoding(self.tok_emb(input))
        # out.shape : bs, seq_len, emb_dim
        for i in range(len(self.fft_blocks)):
            out = self.fft_blocks[i](out)
        return out

class FastSpeechDecoder(nn.Module):
    def __init__(self, hidden_size, hidden_size2, num_heads, kernel_size, n_fft_blocks, device):
        super(FastSpeechDecoder, self).__init__()
        self.pos_encoding = PositionalEncoding(emb_size=hidden_size, dropout=0)
        self.fft_blocks = torch.nn.ModuleList([FFTBlock(hidden_size, hidden_size2, num_heads, kernel_size, device).to(device) for _ in range(n_fft_blocks)])
        self.linear = nn.Linear(hidden_size, 80)

    def forward(self, input: Tensor):
        out = self.pos_encoding(input)
        for i in range(len(self.fft_blocks)):
            out = self.fft_blocks[i](out)
        out = self.linear(out)
        return out

class FastSpeech(nn.Module):
    def __init__(self,config):
        super(FastSpeech, self).__init__()
        self.encoder = FastSpeechEncoder(config.vocab_size, config.hidden_size,
                                         config.hidden_size_fft,
                                         config.num_heads, config.kernel_size,
                                         config.n_fft_blocks, config.device)
        self.aligner = Aligner(config.hidden_size, config.kernel_size)
        self.decoder = FastSpeechDecoder(config.hidden_size, config.hidden_size_fft,
                                         config.num_heads,
                                         config.kernel_size, config.n_fft_blocks,
                                         config.device)

    def forward(self, input, alignes=None):
        input = self.encoder(input)
        length_pred = self.aligner(input).squeeze(-1)
        out = []
        if alignes is None:
            ones = torch.ones(length_pred.shape).to(input.device)
            alignes = length_pred
            alignes, _ = torch.max(torch.stack([ones, alignes], dim=0), dim=0)
            alignes = alignes.type(torch.LongTensor).to(input.device)
        for i in range(input.shape[0]):
            curr_elem = torch.repeat_interleave(input[i], alignes[i], dim=0)
            out.append(curr_elem)
        out = pad_sequence(out, batch_first=True)
        out = self.decoder(out)
        out = torch.transpose(out, 1, 2)
        return out, length_pred
