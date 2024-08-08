from typing import List, Optional, Tuple

import math
import torch
from torch import nn, Tensor


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSeparable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))
        nn.init.normal_(self.depthwise_conv.weight, mean=0, std=std)
        nn.init.normal_(self.pointwise_conv.weight, mean=0, std=std)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class SepConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = LayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.ReLU(inplace=True)
        self.conv1 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.activation_fn(self.conv1(x))
        x = self.dropout(x)
        x = self.activation_fn(self.conv2(x))
        x = self.dropout(x)
        return residual + x


class Model(nn.Module):
    def __init__(
        self,
        num_phones: int,
        num_speakers: int,
        num_mel_bins: int,
        num_tones: int = 7,
        tone_embedding: int = 16,
        d_model: int = 256,
        layer_dropout: float = 0.2,
        encoder_kernel_sizes: List[int] = [5, 25, 13, 9],
        decoder_kernel_sizes: List[int] = [17, 21, 9, 3],
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

        self.num_speakers = num_speakers
        if self.num_speakers > 1:
            self.speaker_embedding = nn.Embedding(self.num_speakers, d_model)
        self.embed_tokens = nn.Embedding(
            num_phones, d_model - tone_embedding, padding_idx=self.padding_idx
        )
        self.embed_tones = nn.Embedding(
            num_tones, tone_embedding, padding_idx=self.padding_idx
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.embed_pitch = nn.Conv1d(2, d_model, kernel_size=1)

        self.encoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in encoder_kernel_sizes
            ]
        )
        self.decoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in decoder_kernel_sizes
            ]
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

        self.layer_norm = LayerNorm1d(d_model)
        self.layer_norm2 = LayerNorm1d(d_model)
        self.mel_out = nn.Conv1d(d_model, num_mel_bins, kernel_size=1)

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
            repeated_indices[k, : res.shape[0]] = res.unsqueeze(-1)

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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat(
            (self.embed_tokens(tokens), self.embed_tones(tones)), dim=-1
        ).transpose(1, 2)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        encoder_outputs = self.layer_norm(x).transpose(1, 2)

        if self.num_speakers > 1:
            encoder_outputs += self.speaker_embedding(speakers.long()).unsqueeze(1)

        duration_prediction = self.duration_predictor(
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        if mels is not None and durations is not None:
            durations = torch.maximum(torch.round(durations), torch.tensor(0)).long()
            mel_time = mels.shape[1]
            assert torch.max(torch.sum(durations, dim=1)).item() == mel_time
        else:
            duration_prediction = torch.exp(duration_prediction) - 1
            durations = torch.maximum(
                torch.round(duration_prediction), torch.tensor(0)
            ).long()
            mel_time = torch.max(torch.sum(durations, dim=1)).long()

        decoder_inp = self._length_regulator(encoder_outputs, mel_time, durations)
        decoder_inp = self.dropout(decoder_inp).transpose(1, 2)

        pitch_feat = self.pitch_predictor(decoder_inp)
        new_feat = (
            torch.stack((pitches, periodicity), dim=2).transpose(1, 2)
            if pitches is not None
            else pitch_feat.clone()
        )
        new_feat = new_feat.detach()

        decoder_inp += self.embed_pitch(new_feat)

        for decoder_layer in self.decoder:
            decoder_inp = decoder_layer(decoder_inp)
        decoder_outputs = self.layer_norm2(decoder_inp)

        decoder_outputs = self.mel_out(decoder_outputs).transpose(1, 2)

        return decoder_outputs, duration_prediction, pitch_feat[:, 0], pitch_feat[:, 1]
