import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionBlock(nn.Module):
    """
    Flamingo-style gated cross-attention block (Alayrac et al., NeurIPS 2022).


    Args:
        d_model:   Dimensionality of the visual (query) tokens.
        d_cross:   Bottleneck dimensionality for cross-attention projections.
        n_heads:   Number of attention heads (must divide d_cross evenly).
        d_rf:      Dimensionality of the RF (key/value) tokens.  Defaults to d_cross.
        ffn_ratio: FFN hidden-layer size as a fraction of d_model.
        dropout:   Dropout probability applied in attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_cross: int,
        n_heads: int,
        d_rf: int = None,
        ffn_ratio: float = 0.125,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_rf = d_rf or d_cross
        assert d_cross % n_heads == 0, f"d_cross ({d_cross}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_head = d_cross // n_heads

        # Pre-attention layer norms
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_rf)

        # Bottleneck projections
        self.q_proj = nn.Linear(d_model, d_cross)
        self.k_proj = nn.Linear(d_rf, d_cross)
        self.v_proj = nn.Linear(d_rf, d_cross)
        self.out_proj = nn.Linear(d_cross, d_model)

        # Pre-FFN layer norm
        self.ln_ffn = nn.LayerNorm(d_model)

        # FFN with narrow bottleneck
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

        # Tanh-gated scalars (zero-init => identity at start)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_visual: torch.Tensor,
        rf_tokens: torch.Tensor,
        rf_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h_visual:  [B, N_vis, d_model]  visual tokens from a frozen DINOv2 layer.
            rf_tokens: [B, N_rf,  d_rf]     RF token embeddings.
            rf_mask:   [B, N_rf]            True = real token, False = padding.

        Returns:
            [B, N_vis, d_model]  RF-conditioned visual tokens.
        """
        B, N_vis, _ = h_visual.shape

        # --- Bottleneck cross-attention ---
        q = self.q_proj(self.ln_q(h_visual))          # [B, N_vis, d_cross]
        kv_in = self.ln_kv(rf_tokens)
        k = self.k_proj(kv_in)                         # [B, N_rf,  d_cross]
        v = self.v_proj(kv_in)                         # [B, N_rf,  d_cross]

        # Multi-head reshape
        q = q.view(B, N_vis, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, N_vis, d_head]
        k = k.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)     # [B, H, N_rf,  d_head]
        v = v.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)     # [B, H, N_rf,  d_head]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Mask padding RF tokens (key dimension)
        if rf_mask is not None:
            # rf_mask [B, N_rf] -> [B, 1, 1, N_rf]
            attn = attn.masked_fill(~rf_mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                                      # [B, H, N_vis, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N_vis, -1)        # [B, N_vis, d_cross]
        out = self.out_proj(out)                                          # [B, N_vis, d_model]

        # Gated cross-attention residual
        h = h_visual + torch.tanh(self.gate_attn) * out

        # Gated FFN residual
        h = h + torch.tanh(self.gate_ffn) * self.ffn(self.ln_ffn(h))

        return h
