from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def compatible_state_dict(payload: dict) -> dict:
    state = payload.get("model_state_dict", payload)
    normalized = {}
    for key, value in state.items():
        name = key[5:] if key.startswith("impl.") else key
        if name.startswith("fc_final."):
            continue
        normalized[name] = value
    return normalized


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"Model configuration must be a mapping, got {type(config)!r}.")
    model = config.get("model", config)
    if not isinstance(model, dict):
        raise TypeError("The model section must be a mapping.")
    if str(model.get("variant", "simple")).lower() != "simple":
        raise ValueError("This release contains only the ModernTCN simple variant.")
    if "encoder" not in model or "decoder" not in model:
        raise KeyError("The model configuration requires encoder and decoder sections.")
    return model


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(inputs))))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            DepthwiseSeparableConv1d(channels, kernel_size),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class ModernTCN(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = normalize_model_config(config)
        encoder = model["encoder"]
        decoder = model["decoder"]

        self.input_channels = int(model.get("input_channels", 3))
        self.sequence_length = int(model.get("sequence_length", 100))
        self.hidden_dim = int(encoder["hidden_dim"])
        self.num_blocks = int(encoder["num_blocks"])
        self.kernel_size = int(encoder["kernel_size"])
        self.dropout = float(encoder["dropout"])
        self.decoder_hidden_dim = int(decoder["hidden_dim"])
        self.output_dim = int(decoder.get("output_dim", 1))

        self.input_proj = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=1)
        self.encoder = nn.Sequential(
            *[
                TemporalBlock(self.hidden_dim, self.kernel_size, self.dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_hidden_dim, self.output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected a three-dimensional input, got {tuple(inputs.shape)}.")
        if inputs.shape[1] != self.input_channels and inputs.shape[2] == self.input_channels:
            inputs = inputs.permute(0, 2, 1)
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got input shape {tuple(inputs.shape)}."
            )
        encoded = self.encoder(self.input_proj(inputs))
        return self.global_pool(encoded).squeeze(-1)

    def forward(
        self, inputs: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self._encode(inputs)
        predictions = self.decoder(features)
        if return_features:
            return predictions, features
        return predictions

    def get_feature_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._encode(inputs)

    def get_projected_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self._encode(inputs)
        for layer in list(self.decoder.children())[:-1]:
            features = layer(features)
        return features

    def get_last_layer(self) -> nn.Linear:
        layer = self.decoder[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError("The final decoder layer must be linear.")
        return layer

    def get_config_summary(self) -> dict[str, Any]:
        return {
            "variant": "simple",
            "input_channels": self.input_channels,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "output_dim": self.output_dim,
        }
