from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F

EPSILON = 1e-6


class ORTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])
        if len(template_list) == 1:
            template_list = template_list[0]

        ce_template_mask = None
        ce_keep_rate = None
        use_ce = bool(getattr(self.cfg.MODEL, 'USE_CE', False)) and bool(getattr(self.cfg.MODEL.BACKBONE, 'CE_LOC', []))
        if use_ce:
            bs = template_list[0].shape[0]
            device = template_list[0].device
            ce_template_mask = generate_mask_cond(self.cfg, bs, device, data['template_anno'][0])
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0],
            )

        if getattr(self.net, 'is_distill_training', False):
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template=template_list, search=search_img, is_distill=True)

        if ce_template_mask is not None:
            real_bs = template_list[0].shape[0] if isinstance(template_list, list) else template_list.shape[0]
            if ce_template_mask.shape[0] != real_bs:
                ce_template_mask = ce_template_mask[0].unsqueeze(0).repeat(real_bs, 1)

        out_dict = self.net(template=template_list, search=search_img, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, is_distill=False)

        if getattr(self.net, 'is_distill_training', False):
            feat_teacher = out_dict_teacher['backbone_feat']
            feat_student = out_dict['backbone_feat']
            distill_loss = torch.stack([
                torch.nn.functional.mse_loss(feat_teacher[i], feat_student[i]) for i in range(feat_student.shape[0])
            ]).to(feat_student.device)
            out_dict['distill_loss'] = distill_loss

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

        try:
            giou_loss, iou, giou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou, giou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps) if 'score_map' in pred_dict else torch.tensor(0.0, device=l1_loss.device)

        sim_loss = pred_dict.get('sim_loss', torch.tensor(0.0, device=l1_loss.device))
        if not bool(getattr(self.cfg.MODEL, 'USE_SIM_LOSS', False)):
            sim_loss = torch.tensor(0.0, device=l1_loss.device)

        cropr_loss = torch.tensor(0.0, device=l1_loss.device)
        if bool(getattr(self.cfg.MODEL, 'USE_CROPR', False)):
            cropr_scores = pred_dict.get('cropr_scores', [])
            cropr_targets = pred_dict.get('cropr_targets', [])
            if len(cropr_scores) > 0 and len(cropr_scores) == len(cropr_targets):
                losses = []
                for score_logits, score_target in zip(cropr_scores, cropr_targets):
                    if torch.is_tensor(score_logits) and torch.is_tensor(score_target) and score_logits.shape == score_target.shape:
                        target = score_target / score_target.sum(dim=-1, keepdim=True).clamp_min(EPSILON)
                        log_prob = torch.log_softmax(score_logits, dim=-1)
                        ce_term = -(target * log_prob).sum(dim=-1).mean()
                        pred_prob = torch.softmax(score_logits, dim=-1)
                        entropy = -(pred_prob * torch.log(pred_prob.clamp_min(EPSILON))).sum(dim=-1).mean()
                        losses.append(ce_term - 0.01 * entropy)
                if len(losses) > 0:
                    cropr_loss = torch.stack(losses).mean()

        pro_loss = torch.tensor(0.0, device=l1_loss.device)
        pro_loss_weight = self.loss_weight.get('pro_loss', 0.0)
        if not bool(getattr(self.cfg.MODEL, 'USE_PRO_LOSS', False)):
            pro_loss_weight = 0.0
        if pro_loss_weight > 0 and 'pro' in pred_dict and 'cos_tensor' in pred_dict:
            pro = pred_dict['pro']
            cos_tensor = pred_dict['cos_tensor']
            if torch.is_tensor(pro) and torch.is_tensor(cos_tensor) and pro.shape == cos_tensor.shape and pro.numel() > 0:
                target = torch.softmax(cos_tensor.detach(), dim=1)
                pro_loss = F.kl_div(torch.log(pro.clamp_min(EPSILON)), target, reduction='batchmean')

        cropr_loss_weight = self.loss_weight.get('cropr_loss', 0.0) if bool(getattr(self.cfg.MODEL, 'USE_CROPR', False)) else 0.0

        if getattr(self.net, 'is_distill_training', False):
            distill_loss = pred_dict['distill_loss']
            tau_0 = 10
            rho = 10
            coef = self.loss_weight['distill_loss'] * (tau_0 + rho * ((1 - giou) - (1 - giou).mean()))
            distill_loss = (coef * distill_loss).mean()
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight.get('sim_loss', 0.0) * sim_loss + cropr_loss_weight * cropr_loss + pro_loss_weight * pro_loss + distill_loss
        else:
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight.get('sim_loss', 0.0) * sim_loss + cropr_loss_weight * cropr_loss + pro_loss_weight * pro_loss

        if return_status:
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/pro": pro_loss.item(),
                "Loss/cropr": cropr_loss.item(),
                "IoU": mean_iou.item(),
            }
            return loss, status
        return loss
