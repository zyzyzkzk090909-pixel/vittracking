import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .base_backbone import collect_sgla_outputs, finalize_sgla_outputs, configure_cropr
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, num_patches_template=None, num_template=1):
        super().__init__()
        self.img_size = to_2tuple(img_size) if not isinstance(img_size, tuple) else img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = num_patches_template
        self.num_template = num_template

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i, num_patches_template=num_patches_template)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device).repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device).repeat(B, 1)

        removed_indexes_s = []
        sgla_logits = []
        sgla_cos_values = []
        cropr_scores = []
        cropr_targets = []

        for i, blk in enumerate(self.blocks):
            block_out = blk(
                x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, num_template=self.num_template)
            if len(block_out) == 7:
                x, global_index_t, global_index_s, removed_index_s, attn, score_logits, score_target = block_out
            elif len(block_out) == 6:
                x, global_index_t, global_index_s, removed_index_s, attn, score_logits = block_out
                score_target = None
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = block_out
                score_logits = None
                score_target = None
            collect_sgla_outputs(self, i, x, global_index_t.shape[1], global_index_s.shape[1], sgla_logits, sgla_cos_values)
            if score_logits is not None:
                cropr_scores.append(score_logits)
            if score_target is not None:
                cropr_targets.append(score_target)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        x = torch.cat([z, x], dim=1)

        aux_dict = {"attn": attn, "removed_indexes_s": removed_indexes_s, "cropr_scores": cropr_scores, "cropr_targets": cropr_targets}
        finalize_sgla_outputs(aux_dict, sgla_logits, sgla_cos_values)
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None, tnc_keep_rate=None, return_last_attn=False):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
    return model


def vit_tiny_patch16_224_ce(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = VisionTransformerCE(**model_kwargs)
    if pretrained and not isinstance(pretrained, str):
        try:
            import timm
            timm_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
            model.load_state_dict(timm_model.state_dict(), strict=False)
        except Exception:
            pass
    return model
