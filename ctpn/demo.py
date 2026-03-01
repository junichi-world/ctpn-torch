from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.networks.factory import get_network
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.utils.timer import Timer
from ctpn.tf_runtime import configure_tensorflow_runtime


def find_latest_checkpoint(checkpoint_path):
    candidates = []
    if os.path.isabs(checkpoint_path):
        candidates.append(checkpoint_path)
    else:
        candidates.extend(
            [
                checkpoint_path,
                os.path.join(PROJECT_ROOT, checkpoint_path),
                os.path.join(PROJECT_ROOT, "ctpn", checkpoint_path),
            ]
        )

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        latest = tf.train.latest_checkpoint(norm)
        if latest is not None:
            return latest, norm
    return None, checkpoint_path


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = os.path.basename(image_name)
    stem = os.path.splitext(base_name)[0]

    os.makedirs("data/results", exist_ok=True)

    with open(os.path.join("data", "results", "res_{}.txt".format(stem)), "w") as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            f.write(",".join([str(min_x), str(min_y), str(max_x), str(max_y)]) + "\r\n")

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data", "results", base_name), img)


def ctpn(net, image_name):
    timer = Timer()
    timer.tic()
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(net, img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(("Detection took {:.3f}s for {:d} object proposals").format(timer.total_time, boxes.shape[0]))


if __name__ == "__main__":
    # Enables TensorFlow GPU memory growth via ctpn.tf_runtime.configure_tensorflow_runtime().
    configure_tensorflow_runtime()

    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file("ctpn/text.yml")
    net = get_network("VGGnet_test")
    _ = net(tf.zeros([1, 64, 64, 3], dtype=tf.float32), training=False)

    print(("Loading network {:s}... ".format("VGGnet_test")), end=" ")
    ckpt = tf.train.Checkpoint(model=net)
    latest, resolved_ckpt_dir = find_latest_checkpoint(cfg.TEST.checkpoints_path)
    if latest is None:
        raise RuntimeError("No checkpoint found under {}".format(cfg.TEST.checkpoints_path))
    print("Restoring from {} (resolved from {})...".format(latest, resolved_ckpt_dir), end=" ")
    ckpt.restore(latest).expect_partial()
    print("done")

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for _ in range(2):
        test_ctpn(net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, "demo", "*.png")) + glob.glob(
        os.path.join(cfg.DATA_DIR, "demo", "*.jpg")
    )
    for im_name in im_names:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(("Demo for {:s}".format(im_name)))
        ctpn(net, im_name)
