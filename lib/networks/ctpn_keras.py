import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

from lib.fast_rcnn.config import cfg
from lib.rpn_msr.anchor_target_layer_tf import anchor_target_layer
from lib.rpn_msr.proposal_layer_tf import proposal_layer


class CTPNModel(nn.Module):
    def __init__(self, name="ctpn_model"):
        super().__init__()
        self.name = name
        self.num_anchors = len(cfg.ANCHOR_SCALES) * 10

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # block5_conv3 output (after ReLU)
        self.vgg_backbone = nn.Sequential(*list(backbone.children())[:30])

        self.rpn_conv_3x3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.lstm_o_bilstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_o_fc = nn.Linear(256, 512)

        self.rpn_bbox_pred = nn.Conv2d(512, self.num_anchors * 4, kernel_size=1)
        self.rpn_cls_score = nn.Conv2d(512, self.num_anchors * 2, kernel_size=1)

    @staticmethod
    def spatial_reshape_layer(inputs, d):
        n, h = inputs.shape[:2]
        return inputs.reshape(n, h, -1, int(d))

    @staticmethod
    def smooth_l1_dist(deltas, sigma2=9.0):
        deltas_abs = deltas.abs()
        smooth_l1_sign = (deltas_abs < (1.0 / sigma2)).float()
        return (deltas * deltas) * 0.5 * sigma2 * smooth_l1_sign + (deltas_abs - 0.5 / sigma2) * (
            1.0 - smooth_l1_sign
        )

    def forward(self, images, training=False):
        x = images.float()
        if x.dim() != 4:
            raise ValueError("images must be a 4D tensor")

        # Input blobs are NHWC in this codebase.
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.vgg_backbone(x)
        x = F.relu(self.rpn_conv_3x3(x), inplace=True)

        n, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(n * h, w, c)
        seq, _ = self.lstm_o_bilstm(seq)
        seq = self.lstm_o_fc(seq)
        lstm_o = seq.view(n, h, w, 512)

        lstm_o_nchw = lstm_o.permute(0, 3, 1, 2).contiguous()
        rpn_bbox_pred = self.rpn_bbox_pred(lstm_o_nchw).permute(0, 2, 3, 1).contiguous()
        rpn_cls_score = self.rpn_cls_score(lstm_o_nchw).permute(0, 2, 3, 1).contiguous()

        rpn_cls_score_reshape = self.spatial_reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=-1)
        rpn_cls_prob_reshape = self.spatial_reshape_layer(rpn_cls_prob, self.num_anchors * 2)

        return {
            "rpn_bbox_pred": rpn_bbox_pred,
            "rpn_cls_score": rpn_cls_score,
            "rpn_cls_score_reshape": rpn_cls_score_reshape,
            "rpn_cls_prob": rpn_cls_prob,
            "rpn_cls_prob_reshape": rpn_cls_prob_reshape,
            "lstm_o": lstm_o,
        }

    def _anchor_targets(self, rpn_cls_score, gt_boxes_list, gt_ishard_list, dontcare_areas_list, im_info):
        feat_h = rpn_cls_score.shape[1]
        feat_w = rpn_cls_score.shape[2]
        batch_size = rpn_cls_score.shape[0]
        device = rpn_cls_score.device

        score_np = rpn_cls_score.detach().cpu().numpy()
        info_np = im_info.detach().cpu().numpy()

        labels_list = []
        targets_list = []
        inside_list = []
        outside_list = []

        for i in range(batch_size):
            boxes_np = np.asarray(gt_boxes_list[i], dtype=np.float32)
            ishard_np = np.asarray(gt_ishard_list[i], dtype=np.int32)
            dontcare_np = np.asarray(dontcare_areas_list[i], dtype=np.float32)

            if boxes_np.ndim != 2 or boxes_np.shape[1] != 5:
                boxes_np = np.zeros((0, 5), dtype=np.float32)
            if ishard_np.ndim != 1:
                ishard_np = np.zeros((boxes_np.shape[0],), dtype=np.int32)
            if dontcare_np.ndim != 2 or dontcare_np.shape[1] != 4:
                dontcare_np = np.zeros((0, 4), dtype=np.float32)

            rpn_labels_i, rpn_bbox_targets_i, rpn_bbox_inside_i, rpn_bbox_outside_i = anchor_target_layer(
                score_np[i:i + 1],
                boxes_np,
                ishard_np,
                dontcare_np,
                info_np[i:i + 1],
                _feat_stride=[16],
                anchor_scales=cfg.ANCHOR_SCALES,
            )

            labels_list.append(
                torch.from_numpy(rpn_labels_i).to(device=device, dtype=torch.float32).view(feat_h, feat_w, self.num_anchors)
            )
            targets_list.append(
                torch.from_numpy(rpn_bbox_targets_i)
                .to(device=device, dtype=torch.float32)
                .view(feat_h, feat_w, self.num_anchors * 4)
            )
            inside_list.append(
                torch.from_numpy(rpn_bbox_inside_i)
                .to(device=device, dtype=torch.float32)
                .view(feat_h, feat_w, self.num_anchors * 4)
            )
            outside_list.append(
                torch.from_numpy(rpn_bbox_outside_i)
                .to(device=device, dtype=torch.float32)
                .view(feat_h, feat_w, self.num_anchors * 4)
            )

        rpn_labels = torch.stack(labels_list, dim=0)
        rpn_bbox_targets = torch.stack(targets_list, dim=0)
        rpn_bbox_inside = torch.stack(inside_list, dim=0)
        rpn_bbox_outside = torch.stack(outside_list, dim=0)
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside, rpn_bbox_outside

    def compute_losses(self, images, im_info, gt_boxes, gt_ishard, dontcare_areas):
        outputs = self(images, training=True)
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside, rpn_bbox_outside = self._anchor_targets(
            outputs["rpn_cls_score"], gt_boxes, gt_ishard, dontcare_areas, im_info
        )

        rpn_cls_score = outputs["rpn_cls_score_reshape"].reshape(-1, 2)
        rpn_label = rpn_labels.reshape(-1)
        rpn_keep = torch.nonzero(rpn_label != -1, as_tuple=False).squeeze(1)

        rpn_cls_score = rpn_cls_score.index_select(0, rpn_keep)
        rpn_label = rpn_label.index_select(0, rpn_keep).long()

        if rpn_label.numel() > 0:
            rpn_cross_entropy_n = F.cross_entropy(rpn_cls_score, rpn_label, reduction="none")
            rpn_cross_entropy = rpn_cross_entropy_n.mean()
        else:
            rpn_cross_entropy_n = torch.zeros(1, device=images.device, dtype=torch.float32)
            rpn_cross_entropy = torch.zeros((), device=images.device, dtype=torch.float32)

        fg_keep = rpn_labels.reshape(-1) == 1

        rpn_bbox_pred = outputs["rpn_bbox_pred"].reshape(-1, 4).index_select(0, rpn_keep)
        rpn_bbox_targets = rpn_bbox_targets.reshape(-1, 4).index_select(0, rpn_keep)
        rpn_bbox_inside = rpn_bbox_inside.reshape(-1, 4).index_select(0, rpn_keep)
        rpn_bbox_outside = rpn_bbox_outside.reshape(-1, 4).index_select(0, rpn_keep)

        rpn_loss_box_n = torch.sum(
            rpn_bbox_outside * self.smooth_l1_dist(rpn_bbox_inside * (rpn_bbox_pred - rpn_bbox_targets)), dim=1
        )
        rpn_loss_box = torch.sum(rpn_loss_box_n) / (fg_keep.float().sum() + 1.0)

        model_loss = rpn_cross_entropy + rpn_loss_box
        reg_loss = torch.zeros((), device=images.device, dtype=torch.float32)
        total_loss = model_loss + reg_loss

        return {
            "total_loss": total_loss,
            "model_loss": model_loss,
            "rpn_cross_entropy": rpn_cross_entropy,
            "rpn_loss_box": rpn_loss_box,
        }

    @torch.no_grad()
    def predict_rois(self, images, im_info, cfg_key="TEST"):
        outputs = self(images, training=False)

        cls_prob_np = outputs["rpn_cls_prob_reshape"].detach().cpu().numpy()
        bbox_pred_np = outputs["rpn_bbox_pred"].detach().cpu().numpy()
        im_info_np = im_info.detach().cpu().numpy()

        rois, bbox_delta = proposal_layer(
            cls_prob_np,
            bbox_pred_np,
            im_info_np,
            cfg_key,
            anchor_scales=cfg.ANCHOR_SCALES,
        )

        device = images.device
        rois = torch.from_numpy(rois.astype(np.float32)).to(device=device)
        bbox_delta = torch.from_numpy(bbox_delta.astype(np.float32)).to(device=device)
        return rois.view(-1, 5), bbox_delta.view(-1, 4), outputs

    def load_pretrained(self, data_path):
        if data_path is None:
            print("Using torchvision VGG16 ImageNet weights; no external preload is applied.")
            return
        print("Ignoring external pretrained model {} because torchvision VGG16 weights are loaded.".format(data_path))
