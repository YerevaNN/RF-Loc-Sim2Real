"""
Flamingo-style RF-conditioned DINOv2 for dense spatial prediction.

AdaptsFlamingo mechanism (Alayrac et al., NeurIPS 2022) to inject
RF base-station signals into a *frozen* DINOv2 backbone via gated cross-attention

-----------------
* DINOv2 weights are entirely frozen; only the cross-attention blocks, the RF
  token encoder, and the decoder are trained.
* Each cross-attention block is initialised as an identity function (tanh-gated
  scalars start at 0) so the pretrained feature geometry is preserved at the
  beginning of training.
* RF tokens carry 2-D sinusoidal positional encoding of base-station map
  coordinates.

The forward signature ``(image, sequence)`` is compatible with the existing
``RomeTransformerUnet`` algorithm — no changes to the training loop, data
pipeline, or loss computation are required.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

from src.networks.gated_cross_attention import GatedCrossAttentionBlock
from src.networks.poe import PositionalEncoding2D
from src.networks.upernet import FPN_fuse, PSPModule
from src.utils import unpatch


class ViTPlusPlusFlamingoUPerNet(nn.Module):

    def __init__(
        self,
        # --- task ---
        num_classes: int,
        image_size: int,
        mlp_input_dim: int,
        # --- DINOv2 backbone ---
        v_num_channels: int,
        v_patch_size: int,
        v_hidden_size: int,
        v_num_hidden_layers: int,
        v_num_attention_heads: int,
        pretrained: str,
        # --- Flamingo cross-attention ---
        xattn_layers: list[int],
        d_cross: int,
        xattn_n_heads: int,
        xattn_ffn_ratio: float,
        xattn_dropout: float,
        d_rf: int,
        # --- decoder ---
        decoder_dim: int,
        up_pool_scales: list[int],
        pre_out_channels: int,
    ):
        super().__init__()

        self.v_hidden_size = v_hidden_size
        self.v_patch_size = v_patch_size
        self.num_tokens = (image_size // v_patch_size) ** 2
        self.xattn_layer_set = set(xattn_layers)
        self.xattn_layers_sorted = sorted(xattn_layers)

        # ------------------------------------------------------------------ #
        #  Frozen DINOv2 backbone                                             #
        # ------------------------------------------------------------------ #
        if pretrained:
            # Load at the checkpoint's native resolution; DINOv2 interpolates
            # position embeddings on the fly for any input size.
            self.vit: Dinov2Model = Dinov2Model.from_pretrained(pretrained)
        else:
            vit_config = Dinov2Config(
                image_size=image_size,
                num_channels=v_num_channels,
                patch_size=v_patch_size,
                hidden_size=v_hidden_size,
                num_hidden_layers=v_num_hidden_layers,
                num_attention_heads=v_num_attention_heads,
            )
            self.vit = Dinov2Model(vit_config)
        self.vit.requires_grad_(False)

        # ------------------------------------------------------------------ #
        #  RF token encoder:  [coords || features] -> PE(coords) || features  #
        #                     -> MLP -> d_rf                                  #
        # ------------------------------------------------------------------ #
        rf_feature_dim = mlp_input_dim - 2          # strip (y, x) coordinates
        self.rf_pe = PositionalEncoding2D(d_model=d_rf)
        self.rf_mlp = nn.Sequential(
            nn.Linear(rf_feature_dim + d_rf, 256),
            nn.GELU(),
            nn.Linear(256, d_rf),
        )

        # ------------------------------------------------------------------ #
        #  Gated cross-attention blocks (one per injection layer)             #
        # ------------------------------------------------------------------ #
        self.xattn_blocks = nn.ModuleDict({
            str(layer_idx): GatedCrossAttentionBlock(
                d_model=v_hidden_size,
                d_cross=d_cross,
                n_heads=xattn_n_heads,
                d_rf=d_rf,
                ffn_ratio=xattn_ffn_ratio,
                dropout=xattn_dropout,
            )
            for layer_idx in xattn_layers
        })

        # ------------------------------------------------------------------ #
        #  UPerNet decoder (reuses PSPModule & FPN_fuse from upernet.py)      #
        # ------------------------------------------------------------------ #
        n_levels = len(xattn_layers)
        self.pre_projs = nn.ModuleList([
            nn.Linear(v_hidden_size, decoder_dim) for _ in range(n_levels)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(decoder_dim) for _ in range(n_levels)
        ])

        feature_channels = [decoder_dim] * n_levels
        self.PPN = PSPModule(feature_channels[-1], bin_sizes=up_pool_scales)
        self.FPN = FPN_fuse(feature_channels)

        self.unpatch_conv = nn.Conv2d(
            decoder_dim,
            pre_out_channels * v_patch_size ** 2,
            kernel_size=1,
            padding="same",
        )
        self.head = nn.Conv2d(
            pre_out_channels, num_classes, kernel_size=3, padding="same"
        )

    # --------------------------------------------------------------------- #
    #  RF token preparation                                                  #
    # --------------------------------------------------------------------- #
    def encode_rf(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build RF tokens from padded base-station data.

        Args:
            sequence: [B, N_rf, mlp_input_dim]  (first 2 cols = normalised y,x)

        Returns:
            rf_tokens: [B, N_rf, d_rf]
            rf_mask:   [B, N_rf]  True for real tokens, False for padding.
        """
        rf_mask = (sequence != 0).all(dim=-1)       # padding rows are all-zero
        coords = sequence[:, :, :2]                  # [B, N_rf, 2]
        features = sequence[:, :, 2:]                # [B, N_rf, rf_feature_dim]
        pe = self.rf_pe(coords)                      # [B, N_rf, d_rf]
        rf_tokens = self.rf_mlp(
            torch.cat([features, pe], dim=-1)        # [B, N_rf, rf_feature_dim + d_rf]
        )                                            # [B, N_rf, d_rf]
        return rf_tokens, rf_mask

    # --------------------------------------------------------------------- #
    #  Forward pass                                                          #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        image: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image:    [B, 3, H, W]  map crop (buildings / roads / BS overlay).
            sequence: [B, N_rf, mlp_input_dim]  padded base-station data.

        Returns:
            [B, 1, H, W]  localization logits (same spatial size as input).
        """
        # --- RF token encoding ---
        rf_tokens = rf_mask = None
        if sequence is not None:
            rf_tokens, rf_mask = self.encode_rf(sequence)

        # --- Image patch embedding (frozen) ---
        h = self.vit.embeddings(image, bool_masked_pos=None)

        # --- Transformer layers with cross-attention injection ---
        extracted: list[torch.Tensor] = []
        for i, layer in enumerate(self.vit.encoder.layer):
            h = layer(h, head_mask=None, output_attentions=False)[0]

            if i in self.xattn_layer_set:
                if rf_tokens is not None:
                    h = self.xattn_blocks[str(i)](h, rf_tokens, rf_mask)
                extracted.append(h)

        # --- Decode multi-level features ---
        hw = int(self.num_tokens ** 0.5)
        outputs_2d: list[torch.Tensor] = []
        for j, feat in enumerate(extracted):
            o = feat[:, 1 : self.num_tokens + 1]    # [B, 256, d_model]  (skip CLS)
            o = self.pre_projs[j](o)                 # [B, 256, decoder_dim]
            o = (
                o.reshape(-1, hw, hw, self.pre_projs[j].out_features)
                 .permute(0, 3, 1, 2)
            )                                        # [B, decoder_dim, 16, 16]
            o = self.bns[j](o)
            outputs_2d.append(o)

        outputs_2d[-1] = self.PPN(outputs_2d[-1])
        feats = self.FPN(outputs_2d)                 # [B, decoder_dim, 16, 16]

        feats = self.unpatch_conv(feats)             # [B, pre_out * p², 16, 16]
        feats = unpatch(feats, hw, hw, self.head.in_channels, self.v_patch_size)
        return self.head(feats)                      # [B, 1, 224, 224]
