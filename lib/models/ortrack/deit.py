from functools import partial
from typing import Sequence, Union

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ====== 关键兼容 ======
try:
    from timm.models._builder import build_model_with_cfg
except:
    try:
        from timm.models import build_model_with_cfg
    except:
        def build_model_with_cfg(cls, variant, pretrained, **kwargs):
            kwargs.pop('pretrained_filter_fn',None)
            kwargs.pop('pretrained_strict', None)
            kwargs.pop('pretrained_cfg',None)
            return cls(**kwargs)

try:
    from timm.layers import resample_abs_pos_embed
except:
    try:
        from timm.models.layers import resample_abs_pos_embed
    except:
        def resample_abs_pos_embed(*args, **kwargs):
            return args[0]

try:
    from timm.models._registry import generate_default_cfgs, register_model
except:
    try:
        from timm.models.registry import register_model
        def generate_default_cfgs(cfgs):
            return cfgs
    except:
        # 最老版本 fallback
        def register_model(fn):
            return fn
        def generate_default_cfgs(cfgs):
            return cfgs

from lib.models.ortrack.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn


# ====== distilled ViT ======
class VisionTransformerDistilled(VisionTransformer):
    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim)
        )
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    def forward_head(self, x, pre_logits=False):
        x, x_dist = x[:, 0], x[:, 1]
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


# ====== builder ======
def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        pretrained_strict=False,
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
})


# ====== model ======
@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return _create_deit('deit_tiny_patch16_224', pretrained, **model_args)


def deit_tiny_patch16_224_distill(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=6, num_heads=3)
    return _create_deit('deit_tiny_patch16_224', pretrained, **model_args)