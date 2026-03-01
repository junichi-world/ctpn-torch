from __future__ import print_function

import os
import os.path as osp
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg
from lib.roi_data_layer import roidb as rdl_roidb
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer

_DEBUG = False


class SolverWrapper(object):
    def __init__(self, network, imdb, roidb, output_dir, logdir, pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print("Computing bounding-box regression targets...")
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print("done")

        self.lr = float(cfg.TRAIN.LEARNING_RATE)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
        self.optimizer = self._create_optimizer()

        self.ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=self.optimizer, model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=output_dir, max_to_keep=100)
        self.writer = tf.summary.create_file_writer(logdir)

    def _create_optimizer(self):
        if cfg.TRAIN.SOLVER == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
        if cfg.TRAIN.SOLVER == "RMS":
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        return tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=cfg.TRAIN.MOMENTUM)

    def _get_lr(self):
        try:
            return float(tf.keras.backend.get_value(self.optimizer.learning_rate))
        except Exception:
            return float(self.optimizer.learning_rate)

    def _set_lr(self, value):
        value = float(value)
        self.lr = value
        try:
            self.optimizer.learning_rate.assign(value)
        except Exception:
            self.optimizer.learning_rate = value

    def _ensure_model_built(self):
        _ = self.net(tf.zeros([1, 64, 64, 3], dtype=tf.float32), training=False)

    def snapshot(self):
        save_path = self.manager.save(checkpoint_number=int(self.global_step.numpy()))
        print("Wrote snapshot to: {:s}".format(save_path))

    def train_model(self, max_iters, restore=False):
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        self._ensure_model_built()

        if self.pretrained_model is not None and not restore:
            try:
                print(("Loading pretrained model weights from {:s}").format(self.pretrained_model))
                self.net.load_pretrained(self.pretrained_model)
            except Exception as e:
                raise Exception("Check your pretrained model {:s}: {}".format(self.pretrained_model, e)) from e

        if restore:
            latest_ckpt = self.manager.latest_checkpoint
            if latest_ckpt is None:
                raise RuntimeError("No checkpoint found under {}".format(self.output_dir))
            print("Restoring from {}...".format(latest_ckpt), end=" ")
            self.ckpt.restore(latest_ckpt).expect_partial()
            print("done")

        restore_iter = int(self.global_step.numpy())
        if restore_iter >= max_iters:
            print("restore_iter ({}) >= max_iters ({}), skipping training".format(restore_iter, max_iters))
            return

        last_snapshot_iter = -1
        timer = Timer()

        for iter in range(restore_iter, max_iters):
            timer.tic()

            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                self._set_lr(self._get_lr() * cfg.TRAIN.GAMMA)
                print("lr -> {:.8f}".format(self._get_lr()))

            blobs = data_layer.forward()

            images = tf.convert_to_tensor(blobs["data"], dtype=tf.float32)
            im_info = tf.convert_to_tensor(blobs["im_info"], dtype=tf.float32)
            gt_boxes = tf.convert_to_tensor(blobs["gt_boxes"], dtype=tf.float32)
            gt_ishard = tf.convert_to_tensor(blobs["gt_ishard"], dtype=tf.int32)
            dontcare_areas = tf.convert_to_tensor(blobs["dontcare_areas"], dtype=tf.float32)

            with tf.GradientTape() as tape:
                losses = self.net.compute_losses(images, im_info, gt_boxes, gt_ishard, dontcare_areas)

            tvars = self.net.trainable_variables
            grads = tape.gradient(losses["total_loss"], tvars)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            grads_and_vars = [(g, v) for g, v in zip(grads, tvars) if g is not None]
            self.optimizer.apply_gradients(grads_and_vars)
            self.global_step.assign_add(1)

            with self.writer.as_default():
                tf.summary.scalar("rpn_reg_loss", losses["rpn_loss_box"], step=self.global_step)
                tf.summary.scalar("rpn_cls_loss", losses["rpn_cross_entropy"], step=self.global_step)
                tf.summary.scalar("model_loss", losses["model_loss"], step=self.global_step)
                tf.summary.scalar("total_loss", losses["total_loss"], step=self.global_step)
                tf.summary.scalar("lr", self._get_lr(), step=self.global_step)

            _diff_time = timer.toc(average=False)

            if iter % cfg.TRAIN.DISPLAY == 0:
                print(
                    "iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, "
                    "rpn_loss_box: %.4f, lr: %f"
                    % (
                        iter,
                        max_iters,
                        float(losses["total_loss"].numpy()),
                        float(losses["model_loss"].numpy()),
                        float(losses["rpn_cross_entropy"].numpy()),
                        float(losses["rpn_loss_box"].numpy()),
                        self._get_lr(),
                    )
                )
                print("speed: {:.3f}s / iter".format(_diff_time))

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot()

        if last_snapshot_iter != iter:
            self.snapshot()


def get_training_roidb(imdb):
    if cfg.TRAIN.USE_FLIPPED:
        print("Appending horizontally-flipped training examples...")
        imdb.append_flipped_images()
        print("done")

    print("Preparing training data...")
    if cfg.TRAIN.HAS_RPN:
        rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print("done")
    return imdb.roidb


def get_data_layer(roidb, num_classes):
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            raise RuntimeError("Calling caffe modules...")
        return RoIDataLayer(roidb, num_classes)
    return RoIDataLayer(roidb, num_classes)


def train_net(network, imdb, roidb, output_dir, log_dir, pretrained_model=None, max_iters=40000, restore=False):
    sw = SolverWrapper(network, imdb, roidb, output_dir, logdir=log_dir, pretrained_model=pretrained_model)
    print("Solving...")
    sw.train_model(max_iters, restore=restore)
    print("done solving")
