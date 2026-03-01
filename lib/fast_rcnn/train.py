from __future__ import print_function

import glob
import os
import os.path as osp
import re
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg
from lib.roi_data_layer import roidb as rdl_roidb
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer

_DEBUG = False


class SolverWrapper(object):
    def __init__(self, network, imdb, roidb, output_dir, logdir, pretrained_model=None, require_cuda=False):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print("Computing bounding-box regression targets...")
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print("done")

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install CUDA-enabled PyTorch and verify GPU drivers. "
                "Current torch version: {}, torch.version.cuda: {}".format(torch.__version__, torch.version.cuda)
            )

        self.device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.lr = float(cfg.TRAIN.LEARNING_RATE)
        self.global_step = 0
        self.optimizer = self._create_optimizer()

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def _create_optimizer(self):
        if cfg.TRAIN.SOLVER == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.SOLVER == "RMS":
            return torch.optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        return torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )

    def _get_lr(self):
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_lr(self, value):
        value = float(value)
        self.lr = value
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _checkpoint_path(self, step):
        return osp.join(self.output_dir, "ctpn_iter_{:07d}.pth".format(step))

    def _find_latest_checkpoint(self):
        files = glob.glob(osp.join(self.output_dir, "ctpn_iter_*.pth"))
        if not files:
            return None, 0

        def _extract_step(path):
            m = re.search(r"ctpn_iter_(\d+)\.pth$", osp.basename(path))
            return int(m.group(1)) if m else -1

        files.sort(key=_extract_step)
        latest = files[-1]
        return latest, _extract_step(latest)

    def snapshot(self):
        save_path = self._checkpoint_path(self.global_step)
        torch.save(
            {
                "step": self.global_step,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            save_path,
        )
        print("Wrote snapshot to: {:s}".format(save_path))

    def train_model(self, max_iters, restore=False):
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        self.net.train()

        if self.pretrained_model is not None and not restore:
            try:
                print(("Loading pretrained model weights from {:s}").format(self.pretrained_model))
                self.net.load_pretrained(self.pretrained_model)
            except Exception as e:
                raise Exception("Check your pretrained model {:s}: {}".format(self.pretrained_model, e)) from e

        if restore:
            latest_ckpt, latest_step = self._find_latest_checkpoint()
            if latest_ckpt is None:
                raise RuntimeError("No checkpoint found under {}".format(self.output_dir))
            print("Restoring from {}...".format(latest_ckpt), end=" ")
            payload = torch.load(latest_ckpt, map_location=self.device)
            self.net.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.global_step = int(payload.get("step", latest_step))
            print("done")

        restore_iter = int(self.global_step)
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

            images = torch.from_numpy(blobs["data"]).to(device=self.device, dtype=torch.float32)
            im_info = torch.from_numpy(blobs["im_info"]).to(device=self.device, dtype=torch.float32)
            gt_boxes = blobs["gt_boxes"]
            gt_ishard = blobs["gt_ishard"]
            dontcare_areas = blobs["dontcare_areas"]

            self.optimizer.zero_grad(set_to_none=True)
            losses = self.net.compute_losses(images, im_info, gt_boxes, gt_ishard, dontcare_areas)
            losses["total_loss"].backward()
            clip_grad_norm_(self.net.parameters(), 10.0)
            self.optimizer.step()
            self.global_step += 1

            self.writer.add_scalar("rpn_reg_loss", float(losses["rpn_loss_box"].detach().cpu().item()), self.global_step)
            self.writer.add_scalar("rpn_cls_loss", float(losses["rpn_cross_entropy"].detach().cpu().item()), self.global_step)
            self.writer.add_scalar("model_loss", float(losses["model_loss"].detach().cpu().item()), self.global_step)
            self.writer.add_scalar("total_loss", float(losses["total_loss"].detach().cpu().item()), self.global_step)
            self.writer.add_scalar("lr", self._get_lr(), self.global_step)

            _diff_time = timer.toc(average=False)

            if iter % cfg.TRAIN.DISPLAY == 0:
                print(
                    "iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, "
                    "rpn_loss_box: %.4f, lr: %f"
                    % (
                        iter,
                        max_iters,
                        float(losses["total_loss"].detach().cpu().item()),
                        float(losses["model_loss"].detach().cpu().item()),
                        float(losses["rpn_cross_entropy"].detach().cpu().item()),
                        float(losses["rpn_loss_box"].detach().cpu().item()),
                        self._get_lr(),
                    )
                )
                print("speed: {:.3f}s / iter".format(_diff_time))

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot()

        if last_snapshot_iter != iter:
            self.snapshot()

        self.writer.flush()
        self.writer.close()


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


def train_net(
    network,
    imdb,
    roidb,
    output_dir,
    log_dir,
    pretrained_model=None,
    max_iters=40000,
    restore=False,
    require_cuda=False,
):
    sw = SolverWrapper(
        network,
        imdb,
        roidb,
        output_dir,
        logdir=log_dir,
        pretrained_model=pretrained_model,
        require_cuda=require_cuda,
    )
    print("Solving...")
    sw.train_model(max_iters, restore=restore)
    print("done solving")
