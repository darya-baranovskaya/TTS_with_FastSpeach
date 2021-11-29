from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Union
from itertools import islice
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

@dataclass
class Batch:
    waveform: torch.Tensor
    waveforn_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    melspec: torch.Tensor = None


    def to(self, device: torch.device) -> 'Batch':
        raise NotImplementedError