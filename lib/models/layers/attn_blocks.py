import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor = None, num_template: int = 1, ori_lens_1z: int = None, attn_t_fusion=None):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    """
    lens_s = attn.shape[-1] - lens_t
    bs = attn.shape[0]

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep >= lens_s:
        return tokens, global_index, None

    if num_template > 1:
        if ori_lens_1z is None:
            raise ValueError("ori_lens_1z (the number of patches in a single template) must be provided when num_template > 1 to correctly extract attention from the first template")
        static_attn = attn[:, :, :ori_lens_1z, lens_t:]
        dynamic_attn = attn[:, :, ori_lens_1z:lens_t, lens_t:] if lens_t > ori_lens_1z else static_attn
        if isinstance(attn_t_fusion, (tuple, list)) and len(attn_t_fusion) == 2:
            w_static, w_dynamic = attn_t_fusion
        else:
            w_static, w_dynamic = 0.7, 0.3
        attn_t = w_static * static_attn + w_dynamic * dynamic_attn
    else:
        attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        if box_mask_z.dim() != 2:
            raise ValueError(f"box_mask_z must be [B, L_t], but got {tuple(box_mask_z.shape)}")
        if box_mask_z.shape[0] != bs:
            raise ValueError(f"box_mask_z batch size {box_mask_z.shape[0]} does not match attention batch size {bs}")
        box_mask = box_mask_z[:, None, :, None].float()
        sum_attn = (attn_t * box_mask).sum(dim=2)
        valid_cnt = box_mask.sum(dim=2).clamp(min=1.0)
        attn_t = (sum_attn / valid_cnt).mean(dim=1)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)

    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    _, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    _, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]
    B, _, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


def cropr_score_tokens(tokens: torch.Tensor, lens_t: int, lens_s: int, keep_ratio: float, scorer: nn.Module = None, global_index: torch.Tensor = None, score_target: torch.Tensor = None):
    """Cropr-lite token scorer for search tokens only."""
    if scorer is None or keep_ratio >= 1.0 or lens_s <= 0:
        return tokens, global_index, None, None, score_target

    lens_keep = max(1, math.ceil(keep_ratio * lens_s))
    if lens_keep >= lens_s:
        return tokens, global_index, None, None, score_target

    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:lens_t + lens_s]

    score_logits = scorer(tokens_s).squeeze(-1)  # [B, L_s]
    sorted_scores, indices = torch.sort(score_logits, dim=1, descending=True)
    _, topk_idx = sorted_scores[:, :lens_keep], indices[:, :lens_keep]
    _, non_topk_idx = sorted_scores[:, lens_keep:], indices[:, lens_keep:]

    if global_index is None:
        global_index = torch.arange(lens_s, device=tokens.device)[None, :].repeat(tokens.shape[0], 1)

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    B, _, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
    return tokens_new, keep_index, removed_index, score_logits, score_target


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                 num_patches_template=None, cropr_enable=False, cropr_keep_ratio=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.num_patches_template = num_patches_template
        self.cropr_enable = cropr_enable
        self.cropr_keep_ratio = cropr_keep_ratio
        self.cropr_scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(1, dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, dim // 2), 1),
        ) if cropr_enable else None

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, num_template=1):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        score_logits = None
        score_target = None
        if self.cropr_enable and self.cropr_keep_ratio < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.cropr_keep_ratio if keep_ratio_search is None else min(keep_ratio_search, self.cropr_keep_ratio)
            # supervision target from template->search attention; detached to keep it stable
            score_target = torch.softmax(attn[:, :, :lens_t, lens_t:].mean(dim=1).mean(dim=1).detach(), dim=-1)
            x, global_index_search, removed_index_search, score_logits, _ = cropr_score_tokens(
                x, lens_t, global_index_search.shape[1], keep_ratio_search, self.cropr_scorer, global_index_search, score_target)
        elif self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(
                attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask,
                num_template=num_template, ori_lens_1z=self.num_patches_template)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn, score_logits, score_target


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
