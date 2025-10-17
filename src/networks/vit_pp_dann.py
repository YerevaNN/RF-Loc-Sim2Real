from typing import Optional, Literal

import torch
import torch.nn as nn

from src.networks.vit_pp_upernet import ViTPlusPlusUPerNet


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 0.0):
        super().__init__()
        self.lambd = lambd

    def set_lambda(self, lambd: float):
        self.lambd = lambd

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)


class ViTPlusPlusDANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        mlp_input_dim: int,
        v_num_channels: int,
        v_patch_size: int,
        v_hidden_size: int,
        v_num_hidden_layers: int,
        v_num_attention_heads: int,
        res_hidden_states: list[int],
        use_upernet: bool,
        up_pool_scales: list[int],
        neck_input_dim: int,
        neck_scales: list[float],
        neck_size: list[int],
        pre_out_channels: int,
        model_type: Literal["clip", "dino_v2"],
        pretrained: str,
        use_pe: bool,
        domain_hidden_dim: int = 128,
    ):
        super().__init__()

        # Backbone that produces task logits via a head and exposes features
        self.backbone = ViTPlusPlusUPerNet(
            num_classes=num_classes,
            image_size=image_size,
            mlp_input_dim=mlp_input_dim,
            v_num_channels=v_num_channels,
            v_patch_size=v_patch_size,
            v_hidden_size=v_hidden_size,
            v_num_hidden_layers=v_num_hidden_layers,
            v_num_attention_heads=v_num_attention_heads,
            res_hidden_states=res_hidden_states,
            use_upernet=use_upernet,
            up_pool_scales=up_pool_scales,
            neck_input_dim=neck_input_dim,
            neck_scales=neck_scales,
            neck_size=neck_size,
            pre_out_channels=pre_out_channels,
            model_type=model_type,
            pretrained=pretrained,
            use_pe=use_pe,
        )

        # GRL for domain adaptation
        self.grl = GRL(lambd=0.0)

        # Domain classifier on CLS vector: MLP -> 1-logit domain score
        in_dim = self.backbone.v_hidden_size
        self.domain_head = nn.Sequential(
            nn.Linear(in_dim, domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(domain_hidden_dim, 1),
        )

    # ----- Parameter groups for separate optimizers -----
    def task_parameters(self):
        """Parameters for the task branch (feature extractor + task head)."""
        return self.backbone.parameters()

    def domain_parameters(self):
        """Parameters for the domain discriminator branch only."""
        return self.domain_head.parameters()

    @torch.no_grad()
    def set_lambda(self, lambd: float):
        self.grl.set_lambda(lambd)

    def forward(
        self,
        image,
        sequence=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_features: bool = False,
    ):
        # Extract dense features and get task logits from backbone
        feats = self.backbone.extract_features(
            image,
            sequence,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits_y = self.backbone.predict_from_features(feats)

        # Domain prediction from CLS token with gradient reversal
        cls = self.backbone.extract_cls_feature(feats)
        cls_rev = self.grl(cls)
        logits_d = self.domain_head(cls_rev).view(cls.size(0))

        if return_features:
            return logits_y, logits_d, feats
        return logits_y, logits_d
