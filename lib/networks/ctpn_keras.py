import numpy as np
import tensorflow as tf
import inspect

from lib.fast_rcnn.config import cfg
from lib.rpn_msr.anchor_target_layer_tf import anchor_target_layer
from lib.rpn_msr.proposal_layer_tf import proposal_layer


class CTPNModel(tf.keras.Model):
    def __init__(self, name="ctpn_model"):
        super().__init__(name=name)
        weight_decay = cfg.TRAIN.WEIGHT_DECAY
        regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay else None
        self.num_anchors = len(cfg.ANCHOR_SCALES) * 10

        # Use ImageNet-pretrained VGG16 features up to block5_conv3.
        vgg16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        self.vgg_backbone = tf.keras.Model(
            inputs=vgg16.input,
            outputs=vgg16.get_layer("block5_conv3").output,
            name="vgg16_backbone",
        )

        self.rpn_conv_3x3 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu", name="rpn_conv_3x3", kernel_regularizer=regularizer
        )
        lstm_kwargs = dict(units=128, return_sequences=True)
        lstm_sig = inspect.signature(tf.keras.layers.LSTM)
        if "use_cudnn" in lstm_sig.parameters:
            # Newer Keras: explicitly avoid the cuDNN kernel path.
            lstm_kwargs["use_cudnn"] = False
        else:
            # tf_keras in NGC TF 2.17 does not expose `use_cudnn`; setting
            # recurrent_dropout>0 disables the cuDNN implementation path.
            # This has no effect during inference (`training=False`).
            lstm_kwargs["recurrent_dropout"] = 1e-6

        self.lstm_o_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(**lstm_kwargs),
            merge_mode="concat",
            name="lstm_o_bilstm",
        )
        self.lstm_o_fc = tf.keras.layers.Dense(512, name="lstm_o_fc", kernel_regularizer=regularizer)
        self.rpn_bbox_pred = tf.keras.layers.Conv2D(
            self.num_anchors * 4, 1, padding="same", name="rpn_bbox_pred", kernel_regularizer=regularizer
        )
        self.rpn_cls_score = tf.keras.layers.Conv2D(
            self.num_anchors * 2, 1, padding="same", name="rpn_cls_score", kernel_regularizer=regularizer
        )

    @staticmethod
    def spatial_reshape_layer(inputs, d):
        input_shape = tf.shape(inputs)
        return tf.reshape(inputs, [input_shape[0], input_shape[1], -1, int(d)])

    @staticmethod
    def smooth_l1_dist(deltas, sigma2=9.0):
        deltas_abs = tf.abs(deltas)
        smooth_l1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smooth_l1_sign + (deltas_abs - 0.5 / sigma2) * tf.abs(
            smooth_l1_sign - 1
        )

    def call(self, images, training=False):
        x = tf.convert_to_tensor(images, dtype=tf.float32)
        x = self.vgg_backbone(x, training=training)

        x = self.rpn_conv_3x3(x)
        shape = tf.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        seq = tf.reshape(x, [n * h, w, c])
        seq = self.lstm_o_bilstm(seq, training=training)
        seq = self.lstm_o_fc(seq)
        lstm_o = tf.reshape(seq, [n, h, w, 512])

        rpn_bbox_pred = self.rpn_bbox_pred(lstm_o)
        rpn_cls_score = self.rpn_cls_score(lstm_o)

        rpn_cls_score_reshape = self.spatial_reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape, axis=-1)
        rpn_cls_prob_reshape = self.spatial_reshape_layer(rpn_cls_prob, self.num_anchors * 2)

        return {
            "rpn_bbox_pred": rpn_bbox_pred,
            "rpn_cls_score": rpn_cls_score,
            "rpn_cls_score_reshape": rpn_cls_score_reshape,
            "rpn_cls_prob": rpn_cls_prob,
            "rpn_cls_prob_reshape": rpn_cls_prob_reshape,
            "lstm_o": lstm_o,
        }

    def _anchor_targets(self, rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info):
        feat_h = tf.shape(rpn_cls_score)[1]
        feat_w = tf.shape(rpn_cls_score)[2]

        def _anchor_target_np(score, boxes, ishard, dontcare, info):
            return anchor_target_layer(
                score,
                boxes,
                ishard,
                dontcare,
                info,
                _feat_stride=[16],
                anchor_scales=cfg.ANCHOR_SCALES,
            )

        rpn_labels, rpn_bbox_targets, rpn_bbox_inside, rpn_bbox_outside = tf.numpy_function(
            _anchor_target_np,
            [rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )
        rpn_labels = tf.reshape(rpn_labels, [1, feat_h, feat_w, self.num_anchors])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [1, feat_h, feat_w, self.num_anchors * 4])
        rpn_bbox_inside = tf.reshape(rpn_bbox_inside, [1, feat_h, feat_w, self.num_anchors * 4])
        rpn_bbox_outside = tf.reshape(rpn_bbox_outside, [1, feat_h, feat_w, self.num_anchors * 4])
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside, rpn_bbox_outside

    def compute_losses(self, images, im_info, gt_boxes, gt_ishard, dontcare_areas):
        outputs = self(images, training=True)
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside, rpn_bbox_outside = self._anchor_targets(
            outputs["rpn_cls_score"], gt_boxes, gt_ishard, dontcare_areas, im_info
        )

        rpn_cls_score = tf.reshape(outputs["rpn_cls_score_reshape"], [-1, 2])
        rpn_label = tf.reshape(rpn_labels, [-1])
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.gather_nd(rpn_cls_score, rpn_keep)
        rpn_label = tf.cast(tf.gather_nd(rpn_label, rpn_keep), tf.int32)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

        fg_keep = tf.equal(tf.reshape(rpn_labels, [-1]), 1)
        rpn_bbox_pred = tf.gather_nd(tf.reshape(outputs["rpn_bbox_pred"], [-1, 4]), rpn_keep)
        rpn_bbox_targets = tf.gather_nd(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside = tf.gather_nd(tf.reshape(rpn_bbox_inside, [-1, 4]), rpn_keep)
        rpn_bbox_outside = tf.gather_nd(tf.reshape(rpn_bbox_outside, [-1, 4]), rpn_keep)

        rpn_loss_box_n = tf.reduce_sum(
            rpn_bbox_outside * self.smooth_l1_dist(rpn_bbox_inside * (rpn_bbox_pred - rpn_bbox_targets)),
            axis=[1],
        )

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
        model_loss = rpn_cross_entropy + rpn_loss_box

        if self.losses:
            reg_loss = tf.add_n(self.losses)
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = model_loss + reg_loss

        return {
            "total_loss": total_loss,
            "model_loss": model_loss,
            "rpn_cross_entropy": rpn_cross_entropy,
            "rpn_loss_box": rpn_loss_box,
        }

    def predict_rois(self, images, im_info, cfg_key="TEST"):
        outputs = self(images, training=False)

        def _proposal_np(rpn_cls_prob_reshape, rpn_bbox_pred, im_info_val):
            rois, bbox_delta = proposal_layer(
                rpn_cls_prob_reshape,
                rpn_bbox_pred,
                im_info_val,
                cfg_key,
                anchor_scales=cfg.ANCHOR_SCALES,
            )
            return rois.astype(np.float32), bbox_delta.astype(np.float32)

        rois, bbox_delta = tf.numpy_function(
            _proposal_np,
            [outputs["rpn_cls_prob_reshape"], outputs["rpn_bbox_pred"], im_info],
            [tf.float32, tf.float32],
        )
        rois = tf.reshape(rois, [-1, 5])
        bbox_delta = tf.reshape(bbox_delta, [-1, 4])
        return rois, bbox_delta, outputs

    def load_pretrained(self, data_path):
        if data_path is None:
            print("Using tf.keras.applications.VGG16(weights='imagenet'); no external preload is applied.")
            return
        print(
            "Ignoring external pretrained model {} because VGG16 ImageNet weights are loaded by tf.keras."
            .format(data_path)
        )
