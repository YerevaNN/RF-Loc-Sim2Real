import math
import types
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel, Dinov2Config, Dinov2Model
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.networks.poe import PositionalEncoding2D


def _dinov2_self_attention_forward(self, hidden_states, head_mask=None, output_attentions=False):
    """
    Replacement forward for Dinov2SelfAttention that applies an additive
    attention bias (pre-softmax) instead of the default post-softmax head_mask.

    The bias is read from self._attn_bias, which is set/cleared by
    ViTPlusPlus.forward() around each encoder call.
    """
    mixed_query_layer = self.query(hidden_states)

    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Pre-softmax additive mask: 0 for real tokens, -inf for padding.
    # This ensures padding positions receive exactly 0 attention probability
    # after softmax, and all probability mass stays on real tokens.
    if self._attn_bias is not None:
        attention_scores = attention_scores + self._attn_bias

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs


def _patch_dinov2_attention(vit: Dinov2Model):
    """
    Replace the forward of every Dinov2SelfAttention layer so it reads
    an additive ``_attn_bias`` attribute instead of using head_mask.
    """
    for layer in vit.encoder.layer:
        self_attn = layer.attention.attention
        self_attn._attn_bias = None
        self_attn.forward = types.MethodType(_dinov2_self_attention_forward, self_attn)


class ViTPlusPlus(nn.Module):

    def __init__(
        self, mlp_input_dim: int, image_size: int, v_num_channels: int, v_patch_size: int,
        v_hidden_size: int, v_num_hidden_layers: int, v_num_attention_heads: int, pretrained: str,
        model_type: Literal["clip", "dino_v2"], use_pe: bool
    ):
        super().__init__()
        self.model_type = model_type
        self.v_num_layers = v_num_hidden_layers
        vit_config = dict(
            image_size=image_size,
            num_channels=v_num_channels, patch_size=v_patch_size,
            hidden_size=v_hidden_size, num_hidden_layers=v_num_hidden_layers,
            num_attention_heads=v_num_attention_heads, output_hidden_states=True
        )
        if model_type == "dino_v2":
            self.vit = Dinov2Model(Dinov2Config(**vit_config))
            if pretrained:
                self.vit: Dinov2Model = self.vit.from_pretrained(pretrained, output_hidden_states=True)
            _patch_dinov2_attention(self.vit)
        else:
            self.vit = CLIPVisionModel(CLIPVisionConfig(**vit_config))
            if pretrained:
                self.vit: CLIPVisionModel = self.vit.from_pretrained(pretrained, output_hidden_states=True)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, v_hidden_size),
        )

        self.pe = PositionalEncoding2D(d_model=v_hidden_size)
        self.use_pe = use_pe

    def _set_attn_bias(self, bias):
        """Broadcast an additive attention bias to every patched layer."""
        for layer in self.vit.encoder.layer:
            layer.attention.attention._attn_bias = bias

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        sequence: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions or self.vit.config.output_attentions
        output_hidden_states = output_hidden_states or self.vit.config.output_hidden_states
        return_dict = return_dict or self.vit.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if sequence is not None:
            if not self.use_pe:
                sequence_embedding = self.mlp(sequence)
            else:
                coords = sequence[:, :, :2]
                radio_info = sequence[:, :, 2:]
                sequence_embedding = self.mlp(radio_info)
                sequence_embedding = self.pe(coords) + sequence_embedding

        if self.model_type == "clip":
            img_embeddings = self.vit.vision_model.embeddings(pixel_values)
            img_embeddings = self.vit.vision_model.pre_layrnorm(img_embeddings)
        else:
            img_embeddings = self.vit.embeddings(pixel_values, bool_masked_pos=None)

        attn_bias = None
        if sequence is not None:
            # noinspection PyUnboundLocalVariable
            embeddings = torch.cat([img_embeddings, sequence_embedding], dim=1)
            # noinspection PyArgumentList,PyUnresolvedReferences
            # 1 = real token, 0 = padding
            token_mask = torch.cat(
                [
                    torch.ones_like(img_embeddings[:, :, 0], dtype=torch.float),
                    (sequence != 0).all(axis=-1).float()
                ], axis=1
            )  # [B, seq_len]

            if self.model_type == "clip":
                # CLIP uses a proper attention_mask (pre-softmax), build boolean [B, 1, seq, seq]
                pair_mask = token_mask.unsqueeze(1)  # [B, 1, seq]
                pair_mask = torch.matmul(pair_mask.transpose(1, 2), pair_mask)  # [B, seq, seq]
                attn_bias = pair_mask.unsqueeze(1).to(torch.bool)  # [B, 1, seq, seq]
            else:
                # DINOv2: build additive bias — 0 for real, -inf for padding
                # We only need to mask the *key* dimension: a real query attending
                # to a padding key should be blocked.  Padding queries are discarded
                # after the encoder so their output doesnt matter, but masking them
                # too is harmless and keeps things symmetric.
                # Shape: [B, 1, 1, seq_len] — broadcasts over (heads, query_pos)
                attn_bias = (1.0 - token_mask).unsqueeze(1).unsqueeze(1) * (-1e9)
                # attn_bias[b, :, :, j] == -1e9  when token j is padding
        else:
            embeddings = img_embeddings

        encoder_params = dict(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.model_type == "clip":
            encoder_params["inputs_embeds"] = embeddings
            encoder_params["attention_mask"] = attn_bias
            encoder_outputs = self.vit.vision_model.encoder(**encoder_params)
        else:
            encoder_params["hidden_states"] = embeddings
            self._set_attn_bias(attn_bias)
            encoder_outputs = self.vit.encoder(**encoder_params)
            self._set_attn_bias(None)

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        if self.model_type == "clip":
            pooled_output = self.vit.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
