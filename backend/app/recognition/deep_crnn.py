"""
backend/app/recognition/deep_crnn.py
Ultra-Deep Convolutional Recurrent Neural Network (Deep CRNN + Attention) for Handwriting Text Recognition (HTR).

Architecture Breakdown & Enhanced Hidden Layers:
1. Deep Convolutional Backbone (CNN):
   - 9 Convolutional stages with Batch Normalization and LeakyReLU
   - 6 Residual Bottleneck Blocks with identity skip connections
   - Squeeze-and-Excitation (SE) Attention Hidden Blocks after each residual stage for dynamic channel recalibration (marker stroke enhancement)
   - Asymmetric Max Pooling (2x2, then 2x1) preserving horizontal sequence resolution
2. Map-to-Sequence Feature Flattening:
   - Preserves horizontal spatial sequence order while projecting feature depth
3. Deep Sequence Recurrent Hidden Layers:
   - 3-Layer Deep Bidirectional Long Short-Term Memory (BiLSTM) with 512 hidden units per direction (total 1024 dimensional state)
   - Layer Normalization and Inter-layer Dropout (0.3)
4. Multi-Head Self-Attention Context Layer:
   - 8-Head Multi-Head Attention layer to model long-range handwritten word and stroke dependencies across whiteboard text lines
5. Deep Multi-Layer Perceptron (MLP) Projection Head:
   - Dense Hidden Layer 1: Linear(1024 -> 512) + GELU + LayerNorm + Dropout(0.2)
   - Dense Hidden Layer 2: Linear(512 -> 256) + GELU + Dropout(0.1)
   - CTC Classification Projection: Linear(256 -> num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Hidden Layer.
    Dynamically recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between convolutional channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling over spatial dimensions H x W
        squeeze = x.view(b, c, -1).mean(dim=2)
        # Excitation: Two fully-connected hidden layers with gating
        excitation = self.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        # Scale: Channel-wise multiplication
        return x * excitation


class DeepResidualBlock(nn.Module):
    """
    Deep Residual Bottleneck Block with Batch Normalization, LeakyReLU,
    and integrated Squeeze-and-Excitation channel attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class DeepCRNN(nn.Module):
    """
    Ultra-Deep Convolutional Recurrent Network with SE Attention,
    3-Layer BiLSTM, Multi-Head Self-Attention, and Deep MLP Head.
    """
    def __init__(
        self,
        img_channel: int = 1,
        num_classes: int = 96,
        hidden_size: int = 512,
        num_lstm_layers: int = 3,
        num_attention_heads: int = 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # =====================================================================
        # 1. DEEP CONVOLUTIONAL BACKBONE (CNN) + RESIDUAL & SE HIDDEN LAYERS
        # =====================================================================
        # Stage 1: Initial low-level stroke extraction
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # [B, 64, H/2, W/2]

        # Stage 2: Intermediate features + 2 Residual Blocks
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res2_1 = DeepResidualBlock(128)
        self.res2_2 = DeepResidualBlock(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # [B, 128, H/4, W/4]

        # Stage 3: Asymmetric pooling to preserve horizontal text sequence resolution
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.res3_1 = DeepResidualBlock(256)
        self.res3_2 = DeepResidualBlock(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # [B, 256, H/8, W/4]

        # Stage 4: High-level semantic features + 2 Residual Blocks
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res4_1 = DeepResidualBlock(512)
        self.res4_2 = DeepResidualBlock(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # [B, 512, H/16, W/4]

        # Final Conv layer squashing height dimension
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(2, 1))
        self.bn5 = nn.BatchNorm2d(512)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(0.25)

        # =====================================================================
        # 2. DEEP RECURRENT HIDDEN LAYERS (3-Layer BiLSTM)
        # =====================================================================
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
            batch_first=False
        )

        lstm_output_dim = hidden_size * 2  # 1024 dimensions (forward + backward)

        # =====================================================================
        # 3. MULTI-HEAD SELF-ATTENTION HIDDEN LAYER
        # =====================================================================
        self.mha_ln = nn.LayerNorm(lstm_output_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=0.1
        )

        # =====================================================================
        # 4. DEEP MULTI-LAYER PERCEPTRON (MLP) CLASSIFICATION HEAD
        # =====================================================================
        self.mlp_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  [Batch, Channel, Height, Width] e.g. [B, 1, 32, 256]
        Output shape: [Sequence_Length, Batch, Num_Classes] for CTC loss
        """
        # --- 1. Convolutional & Residual Backbone ---
        feat = self.relu(self.bn1(self.conv1(x)))
        feat = self.pool1(feat)

        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = self.res2_1(feat)
        feat = self.res2_2(feat)
        feat = self.pool2(feat)

        feat = self.relu(self.bn3(self.conv3(feat)))
        feat = self.res3_1(feat)
        feat = self.res3_2(feat)
        feat = self.pool3(feat)

        feat = self.relu(self.bn4(self.conv4(feat)))
        feat = self.res4_1(feat)
        feat = self.res4_2(feat)
        feat = self.pool4(feat)

        feat = self.relu(self.bn5(self.conv5(feat)))
        feat = self.dropout(feat)

        # --- 2. Map-to-Sequence ---
        # Shape: [B, Channels=512, H=1, W_seq] -> [B, 512, W_seq]
        b, c, h, w = feat.size()
        feat = feat.squeeze(2)              # [B, 512, W_seq]
        seq = feat.permute(2, 0, 1)         # [W_seq, B, 512] for PyTorch recurrent layers

        # --- 3. Deep BiLSTM Hidden Sequence Modeling ---
        lstm_out, _ = self.bilstm(seq)      # [W_seq, B, 1024]

        # --- 4. Multi-Head Self-Attention Hidden Context ---
        normed_lstm = self.mha_ln(lstm_out)
        attn_out, _ = self.self_attention(normed_lstm, normed_lstm, normed_lstm)
        context_features = lstm_out + attn_out  # Residual attention skip connection

        # --- 5. Deep MLP Classification Projection ---
        logits = self.mlp_head(context_features)  # [W_seq, B, Num_Classes]

        # Output log probabilities for CTC Loss
        return F.log_softmax(logits, dim=2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alphabet vocabulary for CTC decoding
CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC Blank token
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
IDX2CHAR[0] = ""  # Blank token


def ctc_greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC decoder with blank collapsing.
    """
    with torch.no_grad():
        preds = torch.argmax(log_probs, dim=2)  # [T, B]
        preds = preds.transpose(0, 1).cpu().numpy()  # [B, T]

    transcriptions = []
    for pred_seq in preds:
        tokens = []
        prev_token = 0
        for token in pred_seq:
            if token != 0 and token != prev_token:
                tokens.append(IDX2CHAR.get(token, ""))
            prev_token = token
        transcriptions.append("".join(tokens))
    return transcriptions
