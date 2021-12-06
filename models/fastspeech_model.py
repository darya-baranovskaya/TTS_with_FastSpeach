import torch
from torch import nn
from models.model_layers import *
from models.grapheme_aligner import GraphemeAligner
from torch.nn.utils.rnn import pad_sequence

class FastSpeechEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, hidden_size2, num_heads, kernel_size, n_fft_blocks, dropout):
        super(FastSpeechEncoder, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(emb_size=hidden_size, dropout=0)
        self.fft_blocks = nn.Sequential(*[FFTBlock(hidden_size, hidden_size2, num_heads, kernel_size, dropout) for _ in range(n_fft_blocks)])

    def forward(self, input: Tensor,  mask: Tensor):
        out = self.pos_encoding(self.tok_emb(input))
        # out.shape : bs, seq_len, emb_dims
        out, _ = self.fft_blocks([out, mask])
        return out

class FastSpeechDecoder(nn.Module):
    def __init__(self, hidden_size, hidden_size2, num_heads, kernel_size, n_fft_blocks, dropout):
        super(FastSpeechDecoder, self).__init__()
        self.pos_encoding = PositionalEncoding(emb_size=hidden_size, dropout=0)
        self.fft_blocks = nn.Sequential(*[FFTBlock(hidden_size, hidden_size2, num_heads, kernel_size, dropout) for _ in range(n_fft_blocks)])
        self.linear = nn.Linear(hidden_size, 80)

    def forward(self, input: Tensor, mask: Tensor):
        out = self.pos_encoding(input)
        out, _ = self.fft_blocks([out, mask])
        out = self.linear(out)
        return out

class FastSpeech(nn.Module):
    def __init__(self,config):
        super(FastSpeech, self).__init__()
        self.encoder = FastSpeechEncoder(config.vocab_size, config.hidden_size,
                                         config.hidden_size_fft,
                                         config.num_heads, config.kernel_size,
                                         config.n_fft_blocks, config.dropout)
        self.aligner = Aligner(config.hidden_size, config.kernel_size, config.dropout)
        self.decoder = FastSpeechDecoder(config.hidden_size, config.hidden_size_fft,
                                         config.num_heads,
                                         config.kernel_size, config.n_fft_blocks,
                                         config.dropout)

    def forward(self, input, mask:Tensor=None, alignes=None):
        input = self.encoder(input, mask)
        length_pred = self.aligner(input).squeeze(-1)
        out = []
        out_mask = []
        if alignes is None:
            # alignes = length_pred
            zeros = torch.zeros(length_pred.shape).to(input.device)
            alignes, _ = torch.max(torch.stack([zeros, length_pred], dim=0), dim=0)
        alignes = (alignes + 0.5).type(torch.LongTensor).to(input.device)
        for i in range(input.shape[0]):
            curr_elem = torch.repeat_interleave(input[i], alignes[i], dim=0)
            out.append(curr_elem)
        if mask is None:
            out_mask = None
        else:
            for i in range(input.shape[0]):
                curr_mask = torch.repeat_interleave(mask[i], alignes[i], dim=0)
                out_mask.append(curr_mask)
            out_mask = pad_sequence(out_mask, batch_first=True)
        out = pad_sequence(out, batch_first=True)
        out = self.decoder(out, out_mask)
        out = torch.transpose(out, 1, 2)
        return out, length_pred
