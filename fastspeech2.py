import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FFTBlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.pos_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size=9, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_inner, d_model, kernel_size=1),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        residual = x
        x, attn = self.slf_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(residual + self.dropout(x))
        residual = x
        x = self.pos_ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(residual + self.dropout(x))
        return x, attn


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        d_inner: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                FFTBlock(
                    d_model=hidden_dim, d_inner=d_inner, n_head=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=max_len)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.pos_encoder(x)
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask=mask)
        return x


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        num_phones: int,
        num_speakers: int,
        num_mel_bins: int,
        num_tones: int = 7,
        tone_embedding: int = 16,
        d_model: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_inner: int = 1024,
        transformer_dropout: float = 0.1,
        max_len: int = 1024,
        duration_layers: int = 1,
        duration_kernel_size: int = 3,
        duration_dropout: float = 0.25,
        pitch_layers: int = 6,
        pitch_kernel_size: int = 5,
        pitch_dropout: float = 0.25,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_phones, d_model - tone_embedding, padding_idx=self.padding_idx
        )

        # 0 = padding
        # 1 = silence
        # 2,3,4,5 = tones 1-4
        # 6 = neutral tone
        self.embed_tones = nn.Embedding(
            num_tones, tone_embedding, padding_idx=self.padding_idx
        )
        self.speaker_embedding = nn.Embedding(num_speakers, d_model)
        self.embed_pitch = nn.Linear(2, d_model)

        self.encoder = Transformer(
            d_model,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            d_inner=transformer_inner,
            dropout=transformer_dropout,
            max_len=max_len,
        )
        self.decoder = Transformer(
            d_model,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            d_inner=transformer_inner,
            dropout=transformer_dropout,
            max_len=max_len * 2,
        )

        self.duration_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=1,
            num_layers=duration_layers,
            kernel_size=duration_kernel_size,
            dropout=duration_dropout,
        )
        self.pitch_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=2,
            num_layers=pitch_layers,
            kernel_size=pitch_kernel_size,
            dropout=pitch_dropout,
        )
        self.mel_out = nn.Linear(d_model, num_mel_bins)

    @staticmethod
    def _make_predictor(
        hidden_size: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ):
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nn.ReLU(inplace=True),
                    LayerNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Conv1d(hidden_size, out_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def _length_regulator(self, x: Tensor, mel_time: int, durations: Tensor) -> Tensor:
        bsz, time, feats = x.size()
        indices = torch.arange(time, device=x.device)
        repeated_indices = torch.full(
            (bsz, mel_time, feats),
            fill_value=time - 1,
            dtype=torch.long,
            device=x.device,
        )
        for k in range(bsz):
            res = torch.repeat_interleave(indices, durations[k].long(), dim=0)[
                :mel_time
            ]
            repeated_indices[k, : res.shape[0]] = res[..., None]
        return torch.gather(x, 1, repeated_indices)

    def forward(
        self,
        speakers: Tensor,
        tokens: Tensor,
        tones: Tensor,
        pitches: Optional[Tensor] = None,
        periodicity: Optional[Tensor] = None,
        durations: Optional[Tensor] = None,
        mels: Optional[Tensor] = None,
    ):
        text_embed = torch.cat(
            (self.embed_tokens(tokens), self.embed_tones(tones)), dim=-1
        )
        padding_mask = tokens == self.padding_idx
        encoder_outputs = (
            self.encoder(text_embed, padding_mask)
            + self.speaker_embedding(speakers.long())[:, None]
        )
        duration_prediction = self.duration_predictor(
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        if mels is not None and durations is not None:
            durations = torch.clamp(torch.round(durations), min=0).long()
            mel_time = mels.shape[1]
        else:
            duration_prediction = torch.exp(duration_prediction) - 1
            durations = torch.clamp(torch.round(duration_prediction), min=0).long()
            mel_time = torch.max(torch.sum(durations, dim=1)).long()

        decoder_inp = self._length_regulator(encoder_outputs, mel_time, durations)

        pitch_feat = self.pitch_predictor(decoder_inp.transpose(1, 2)).transpose(1, 2)
        if pitches is not None:
            new_feat = torch.stack((pitches, periodicity), dim=2)
        else:
            new_feat = pitch_feat.detach()

        decoder_inp += self.embed_pitch(new_feat)

        padding_mask = (
            (mels.sum(dim=2) == self.padding_idx) if mels is not None else None
        )
        decoder_outputs = self.decoder(decoder_inp, padding_mask)
        mel_outputs = self.mel_out(decoder_outputs)

        return mel_outputs, duration_prediction, pitch_feat[..., 0], pitch_feat[..., 1]
