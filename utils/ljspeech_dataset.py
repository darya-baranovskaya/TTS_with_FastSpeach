import torch
import torchaudio
from typing import Tuple, Dict, Optional, List, Union
from torch.nn.utils.rnn import pad_sequence
from utils.batch_sampler import Batch
from models.grapheme_aligner import GraphemeAligner
from configs.melspectrogram_config import MelSpectrogramConfig
from utils.melspectrogram import MelSpectrogram

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result

device = torch.device('cuda:0')
aligner = GraphemeAligner().to(device)
featurizer = MelSpectrogram(MelSpectrogramConfig())

class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveforn_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveforn_length = torch.cat(waveforn_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)
        batch = Batch(waveform, waveforn_length, transcript, tokens, token_lengths)
        batch.melspec = featurizer(batch.waveform)
        lengths = []
        for i in range(batch.melspec.shape[0]):
            lengths.append(featurizer(batch.waveform[i:i + 1, :batch.waveforn_length[i]]).shape[-1])
        lengths = torch.Tensor(lengths).unsqueeze(1)
        alignes = aligner(
            batch.waveform.to(device), batch.waveforn_length, batch.transcript
        )
        batch.durations = lengths * alignes
        return batch